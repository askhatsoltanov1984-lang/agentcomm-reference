import { ulid } from "ulid";
import { db } from "../db/connection.ts";
import { QoopiaError, nowIso, safeJsonParse } from "../utils/errors.ts";
import { assertNoSecrets } from "../utils/secret-guard.ts";
import { logActivity } from "./activity.ts";
import { scheduleAgentWakeDrain } from "./agent-wake.ts";

export type AgentCommKind = "request" | "ack" | "reply" | "status" | "system";

type AgentRow = { id: string; name: string; type: string; active: number; last_seen: string | null; tool_profile: string };
type SessionRow = { id: string; workspace_id: string; topic: string; status: string; created_by_agent_id: string; metadata: string; created_at: string; updated_at: string; closed_at: string | null };
type MessageRow = { id: string; workspace_id: string; session_id: string; sender_agent_id: string; recipient_agent_id: string; kind: string; body: string; metadata: string; ack_required: number; acked_at: string | null; ack_status: string | null; parent_message_id: string | null; created_at: string; sender_name?: string; recipient_name?: string; topic?: string; session_status?: string };

function parse(s: string): unknown {
  return safeJsonParse(s, {});
}

function resolveAgent(workspace_id: string, nameOrId: string): AgentRow {
  let target = String(nameOrId || "").trim();
  if (!target) throw new QoopiaError("INVALID_INPUT", "target agent is required");
  // Leo identity: canonical name is 'Leo' (one identity, one path). The legacy
  // name 'leo-agentcomm' redirects here so any in-flight reference still resolves.
  if (target.toLowerCase() === "leo-agentcomm") target = "Leo";
  const row = db.prepare(
    `SELECT id, name, type, active, last_seen, tool_profile FROM agents
     WHERE workspace_id = ? AND active = 1 AND (id = ? OR lower(name) = lower(?))
     LIMIT 1`,
  ).get(workspace_id, target, target) as AgentRow | undefined;
  if (!row) throw new QoopiaError("NOT_FOUND", `active agent not found: ${target}`);
  return row;
}

function getSession(workspace_id: string, session_id: string): SessionRow {
  const s = db.prepare(`SELECT * FROM agent_comm_sessions WHERE workspace_id = ? AND id = ?`).get(workspace_id, session_id) as SessionRow | undefined;
  if (!s) throw new QoopiaError("NOT_FOUND", "agent session not found");
  return s;
}

function messageOut(r: MessageRow) {
  return {
    id: r.id,
    session_id: r.session_id,
    topic: r.topic,
    session_status: r.session_status,
    kind: r.kind,
    from: r.sender_name,
    to: r.recipient_name,
    body: r.body,
    metadata: parse(r.metadata),
    ack_required: !!r.ack_required,
    acked_at: r.acked_at,
    ack_status: r.ack_status,
    parent_message_id: r.parent_message_id,
    created_at: r.created_at,
  };
}

function createWake(workspace_id: string, target_agent_id: string, session_id: string, message_id: string) {
  const id = ulid();
  db.prepare(
    `INSERT INTO agent_wake_events (id, workspace_id, target_agent_id, session_id, message_id, payload, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?)`,
  ).run(id, workspace_id, target_agent_id, session_id, message_id, JSON.stringify({ reason: "agent_message" }), nowIso());
  return id;
}

export function agentSessionCreate(p: { workspace_id: string; agent_id: string; to_agent?: string; topic: string; message?: string; metadata?: Record<string, unknown>; ack_required?: boolean }) {
  const topic = String(p.topic || "").trim();
  if (!topic) throw new QoopiaError("INVALID_INPUT", "topic is required");
  assertNoSecrets(topic, "agent_comm.topic");
  if (p.message) assertNoSecrets(p.message, "agent_comm.message");
  const id = ulid();
  const ts = nowIso();
  db.prepare(
    `INSERT INTO agent_comm_sessions (id, workspace_id, topic, status, created_by_agent_id, metadata, created_at, updated_at)
     VALUES (?, ?, ?, 'open', ?, ?, ?, ?)`,
  ).run(id, p.workspace_id, topic, p.agent_id, JSON.stringify(p.metadata || {}), ts, ts);
  logActivity({ workspace_id: p.workspace_id, agent_id: p.agent_id, action: "agent_session_create", entity_type: "agent_comm_session", entity_id: id, project_id: null, summary: `Created agent session: ${topic}`, details: { session_id: id } });
  let initial_message = null as unknown;
  if (p.to_agent && p.message) {
    initial_message = agentSend({ workspace_id: p.workspace_id, agent_id: p.agent_id, to_agent: p.to_agent, session_id: id, body: p.message, metadata: p.metadata, ack_required: p.ack_required });
  }
  return { id, topic, status: "open", initial_message };
}

export function agentSend(p: { workspace_id: string; agent_id: string; to_agent: string; body: string; session_id?: string; topic?: string; metadata?: Record<string, unknown>; ack_required?: boolean; kind?: AgentCommKind; parent_message_id?: string }) {
  const body = String(p.body || "").trim();
  if (!body) throw new QoopiaError("INVALID_INPUT", "message body is required");
  assertNoSecrets(body, "agent_comm.body");
  let target = resolveAgent(p.workspace_id, p.to_agent);
  const sender = resolveAgent(p.workspace_id, p.agent_id);
  let session_id = p.session_id;
  if (session_id) {
    const s = getSession(p.workspace_id, session_id);
    if (s.status !== "open") throw new QoopiaError("CONFLICT", "agent session is closed");
  } else {
    const created = agentSessionCreate({ workspace_id: p.workspace_id, agent_id: p.agent_id, topic: p.topic || `Message to ${target.name}`, metadata: p.metadata });
    session_id = created.id;
  }
  const id = ulid();
  const ts = nowIso();
  const kind: AgentCommKind = p.kind || "request";
  // Reply routing integrity: a reply MUST return to the original requester (the
  // parent message's sender), regardless of a mis-addressed to_agent — this stops
  // verdicts/answers leaking to a passive third agent (e.g. Leo's replies landing
  // on corsair-main instead of the requester).
  let rerouted_from: string | null = null;
  if (kind === "reply" && p.parent_message_id) {
    const parent = db.prepare(`SELECT sender_agent_id FROM agent_comm_messages WHERE workspace_id = ? AND id = ?`).get(p.workspace_id, p.parent_message_id) as { sender_agent_id: string } | undefined;
    if (parent?.sender_agent_id && parent.sender_agent_id !== p.agent_id && parent.sender_agent_id !== target.id) {
      rerouted_from = target.name;
      target = resolveAgent(p.workspace_id, parent.sender_agent_id);
    }
  }
  // Backlog guard: only a genuine request to a working agent should require an ack.
  // reply/status/system/ack are closures or signals; loop/liveness bodies are fire-and-forget.
  const NO_ACK_KINDS = new Set(["reply", "status", "system", "ack"]);
  const LOOP_SIGNAL = /^(REVERSE_WAKE_OK|WAKE_SLO_[A-Z]+|CORSAIR_LOOP_TERMINATE|[A-Z_]*PONG[A-Z_]*)\b/;
  const ackRequired = p.ack_required === false ? false
    : NO_ACK_KINDS.has(kind) ? false
    : LOOP_SIGNAL.test(body) ? false
    : true;
  let wakeId = "";
  db.transaction(() => {
    db.prepare(
      `INSERT INTO agent_comm_messages (id, workspace_id, session_id, sender_agent_id, recipient_agent_id, kind, body, metadata, ack_required, parent_message_id, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(id, p.workspace_id, session_id, p.agent_id, target.id, kind, body, JSON.stringify(p.metadata || {}), ackRequired ? 1 : 0, p.parent_message_id || null, ts);
    // A reply closes its parent request automatically (the recipient has answered).
    if (kind === "reply" && p.parent_message_id) {
      db.prepare(`UPDATE agent_comm_messages SET acked_at = ?, ack_status = 'auto_acked_by_reply' WHERE workspace_id = ? AND id = ? AND recipient_agent_id = ? AND ack_required = 1 AND acked_at IS NULL`).run(ts, p.workspace_id, p.parent_message_id, p.agent_id);
    }
    db.prepare(`UPDATE agent_comm_sessions SET updated_at = ? WHERE id = ?`).run(ts, session_id);
    // Refresh sender's last_seen on any message write, so agents whose runner
    // posts via an in-process / co-located path (not the authenticated HTTP
    // layer that updates last_seen in verifyApiKey) don't look dormant.
    db.prepare(`UPDATE agents SET last_seen = ? WHERE id = ?`).run(ts, p.agent_id);
    wakeId = createWake(p.workspace_id, target.id, session_id!, id);
  })();
  const srow = getSession(p.workspace_id, session_id!);
  scheduleAgentWakeDrain(wakeId);
  logActivity({ workspace_id: p.workspace_id, agent_id: p.agent_id, action: "agent_send", entity_type: "agent_comm_message", entity_id: id, project_id: null, summary: `Agent message ${kind} to ${target.name}`, details: { session_id, to: target.name, kind, ...(rerouted_from ? { rerouted_from } : {}) } });
  return {
    id,
    session_id,
    to: target.name,
    kind,
    ack_required: ackRequired,
    created_at: ts,
    wake: wakeId ? { event_id: wakeId, status: "queued" } : null,
    topic: srow.topic,
  };
}

export function agentInbox(p: { workspace_id: string; agent_id: string; status?: string; limit?: number }) {
  const limit = Math.min(Math.max(p.limit || 20, 1), 100);
  const where = [`m.workspace_id = ?`, `m.recipient_agent_id = ?`];
  const params: any[] = [p.workspace_id, p.agent_id];
  if (p.status === "unacked" || p.status === "needs_ack") where.push(`m.ack_required = 1 AND m.acked_at IS NULL`);
  if (p.status === "acked") where.push(`m.acked_at IS NOT NULL`);
  const rows = db.prepare(
    `SELECT m.*, sa.name as sender_name, ra.name as recipient_name, s.topic, s.status as session_status
     FROM agent_comm_messages m
     JOIN agents sa ON sa.id = m.sender_agent_id
     JOIN agents ra ON ra.id = m.recipient_agent_id
     JOIN agent_comm_sessions s ON s.id = m.session_id
     WHERE ${where.join(" AND ")}
     ORDER BY m.created_at DESC LIMIT ?`,
  ).all(...params, limit) as MessageRow[];
  return { items: rows.map(messageOut), limit };
}

export function agentAck(p: { workspace_id: string; agent_id: string; message_id: string; status?: string; note?: string }) {
  if (p.note) assertNoSecrets(p.note, "agent_comm.ack_note");
  const row = db.prepare(`SELECT * FROM agent_comm_messages WHERE workspace_id = ? AND id = ?`).get(p.workspace_id, p.message_id) as MessageRow | undefined;
  if (!row) throw new QoopiaError("NOT_FOUND", "message not found");
  if (row.recipient_agent_id !== p.agent_id) throw new QoopiaError("FORBIDDEN", "only recipient can ack this message");
  // Idempotent / tolerant: an already-acked (or auto-acked) message is a no-op, no duplicate ack row.
  if (row.acked_at) return { message_id: p.message_id, acked_at: row.acked_at, ack_status: row.ack_status, noop: true };
  const ts = nowIso();
  const status = p.status || "working";
  db.transaction(() => {
    db.prepare(`UPDATE agent_comm_messages SET acked_at = ?, ack_status = ? WHERE id = ?`).run(ts, status, p.message_id);
    db.prepare(
      `INSERT INTO agent_comm_messages (id, workspace_id, session_id, sender_agent_id, recipient_agent_id, kind, body, metadata, ack_required, parent_message_id, created_at)
       VALUES (?, ?, ?, ?, ?, 'ack', ?, ?, 0, ?, ?)`,
    ).run(ulid(), p.workspace_id, row.session_id, p.agent_id, row.sender_agent_id, p.note || status, JSON.stringify({ status }), row.id, ts);
    db.prepare(`UPDATE agent_comm_sessions SET updated_at = ? WHERE id = ?`).run(ts, row.session_id);
    // Refresh acker's last_seen (see agentSend note) — covers in-process ack paths.
    db.prepare(`UPDATE agents SET last_seen = ? WHERE id = ?`).run(ts, p.agent_id);
  })();
  return { message_id: p.message_id, acked_at: ts, ack_status: status };
}

export function agentReply(p: { workspace_id: string; agent_id: string; session_id: string; body: string; to_agent?: string; reply_to_message_id?: string; metadata?: Record<string, unknown>; close?: boolean }) {
  const s = getSession(p.workspace_id, p.session_id);
  if (s.status !== "open" && !p.close) throw new QoopiaError("CONFLICT", "agent session is closed");
  let to = p.to_agent;
  if (!to) {
    const last = db.prepare(
      `SELECT sender_agent_id, recipient_agent_id FROM agent_comm_messages
       WHERE workspace_id = ? AND session_id = ? AND sender_agent_id != ?
       ORDER BY created_at DESC LIMIT 1`,
    ).get(p.workspace_id, p.session_id, p.agent_id) as { sender_agent_id: string; recipient_agent_id: string } | undefined;
    const targetId = last?.sender_agent_id || s.created_by_agent_id;
    const a = db.prepare(`SELECT name FROM agents WHERE id = ?`).get(targetId) as { name: string } | undefined;
    if (!a) throw new QoopiaError("NOT_FOUND", "reply target not found");
    to = a.name;
  }
  const sent = agentSend({ workspace_id: p.workspace_id, agent_id: p.agent_id, to_agent: to, session_id: p.session_id, body: p.body, metadata: p.metadata, kind: "reply", parent_message_id: p.reply_to_message_id, ack_required: false });
  if (p.close) agentSessionClose({ workspace_id: p.workspace_id, agent_id: p.agent_id, session_id: p.session_id, reason: "closed with reply" });
  return sent;
}

export function agentSessionClose(p: { workspace_id: string; agent_id: string; session_id: string; reason?: string }) {
  if (p.reason) assertNoSecrets(p.reason, "agent_comm.close_reason");
  getSession(p.workspace_id, p.session_id);
  const ts = nowIso();
  db.prepare(`UPDATE agent_comm_sessions SET status = 'closed', closed_at = ?, updated_at = ? WHERE workspace_id = ? AND id = ?`).run(ts, ts, p.workspace_id, p.session_id);
  logActivity({ workspace_id: p.workspace_id, agent_id: p.agent_id, action: "agent_session_close", entity_type: "agent_comm_session", entity_id: p.session_id, project_id: null, summary: "Closed agent session", details: { reason: p.reason || "done" } });
  return { session_id: p.session_id, status: "closed", closed_at: ts };
}

export function agentStatus(p: { workspace_id: string; agent?: string; limit?: number }) {
  const where = [`workspace_id = ?`, `active = 1`];
  const params: any[] = [p.workspace_id];
  if (p.agent) { where.push(`lower(name) = lower(?)`); params.push(p.agent); }
  const agents = db.prepare(`SELECT id, name, type, tool_profile, last_seen FROM agents WHERE ${where.join(" AND ")} ORDER BY name LIMIT ?`).all(...params, Math.min(Math.max(p.limit || 50, 1), 100)) as AgentRow[];
  const backlog = db.prepare(
    `SELECT a.name, COUNT(*) as unacked
     FROM agent_comm_messages m JOIN agents a ON a.id = m.recipient_agent_id
     WHERE m.workspace_id = ? AND m.ack_required = 1 AND m.acked_at IS NULL
     GROUP BY a.name`,
  ).all(p.workspace_id) as Array<{ name: string; unacked: number }>;
  const b = new Map(backlog.map(x => [x.name, x.unacked]));
  return { agents: agents.map(a => ({ name: a.name, type: a.type, tool_profile: a.tool_profile, last_seen: a.last_seen, unacked: b.get(a.name) || 0 })) };
}
