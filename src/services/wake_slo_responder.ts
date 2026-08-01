/**
 * Wake SLO responder — Phase 1 item 1 / Leo R2 fix #3+4.
 *
 * Contract: /srv/qoopia/docs/wake-slo-responder-contract.md
 *
 * When an AgentComm message arrives whose topic + body match the probe
 * protocol, this module immediately (in-process, server-side) writes the
 * matching ack + pong reply so the probe metric becomes a measurement of
 * the AgentComm DB + service pipeline rather than LLM session liveness.
 *
 * Identical code runs on every host that hosts a Qoopia instance — when
 * Leo's Qoopia ships, it inherits the same responder. No per-host
 * configuration; the responder always replies as the message's recipient.
 *
 * Idempotency: if a reply row already exists for the probe (by
 * parent_message_id), the responder returns without writing.
 *
 * The responder is invoked synchronously from `agentSend` AFTER the probe
 * row is persisted and BEFORE pushAgentWake fires. The pong reply path
 * therefore never depends on the wake-push transport surviving the host
 * boundary — which is the point.
 */
import { ulid } from "ulid";
import { db } from "../db/connection.ts";

// Probe latency is measured in ms; the rest of agent_comm_messages uses
// second-precision timestamps via nowIso(). Use millisecond precision
// inside the responder so probe latency arithmetic is meaningful — at
// the cost of these specific rows carrying a slightly more precise ISO
// string than their neighbours.
function nowIsoMs(): string {
  return new Date().toISOString();
}

const TOPIC_RE = /^WAKE_SLO_PROBE_(C2L|L2C)_\d+$/;
const BODY_RE = /^WAKE_SLO_PING (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)$/;
const ACK_STATUS = "probe_auto_ack";

export type ProbeRespondInput = {
  workspace_id: string;
  message_id: string;
  session_id: string;
  topic: string;
  body: string;
  sender_agent_id: string;
  recipient_agent_id: string;
};

export type ProbeRespondResult =
  | { matched: false }
  | { matched: true; replied: false; reason: "duplicate" }
  | { matched: true; replied: true; reply_message_id: string; pong_ts: string };

export function maybeRespondToProbe(p: ProbeRespondInput): ProbeRespondResult {
  if (!TOPIC_RE.test(p.topic)) return { matched: false };
  const m = BODY_RE.exec(p.body);
  if (!m) return { matched: false };
  const pong_ts = m[1]!;

  // Idempotency: refuse to write a second reply if one is already on file
  // for this probe (parent_message_id = original probe id, kind='reply').
  const existing = db
    .prepare(
      `SELECT id FROM agent_comm_messages
        WHERE workspace_id = ?
          AND session_id = ?
          AND parent_message_id = ?
          AND kind = 'reply'
        LIMIT 1`,
    )
    .get(p.workspace_id, p.session_id, p.message_id) as { id: string } | undefined;
  if (existing) return { matched: true, replied: false, reason: "duplicate" };

  const ts = nowIsoMs();
  const ack_id = ulid();
  const reply_id = ulid();
  const pong_body = `WAKE_SLO_PONG ${pong_ts}`;

  // Single transaction: ack the probe (mirrors `agentAck`'s side effects)
  // and insert the pong reply. Direct DB writes — calling back through
  // `agentSend`/`agentAck` here would recurse on the same responder hook
  // and is unnecessary because the rows are fully deterministic.
  db.transaction(() => {
    db.prepare(
      `UPDATE agent_comm_messages
          SET acked_at = ?, ack_status = ?
        WHERE id = ?`,
    ).run(ts, ACK_STATUS, p.message_id);
    db.prepare(
      `INSERT INTO agent_comm_messages
         (id, workspace_id, session_id, sender_agent_id, recipient_agent_id,
          kind, body, metadata, ack_required, parent_message_id, created_at)
       VALUES (?, ?, ?, ?, ?, 'ack', ?, ?, 0, ?, ?)`,
    ).run(
      ack_id,
      p.workspace_id,
      p.session_id,
      p.recipient_agent_id,
      p.sender_agent_id,
      ACK_STATUS,
      JSON.stringify({ status: ACK_STATUS, by: "wake_slo_responder" }),
      p.message_id,
      ts,
    );
    db.prepare(
      `INSERT INTO agent_comm_messages
         (id, workspace_id, session_id, sender_agent_id, recipient_agent_id,
          kind, body, metadata, ack_required, parent_message_id, created_at)
       VALUES (?, ?, ?, ?, ?, 'reply', ?, ?, 0, ?, ?)`,
    ).run(
      reply_id,
      p.workspace_id,
      p.session_id,
      p.recipient_agent_id,
      p.sender_agent_id,
      pong_body,
      JSON.stringify({ probe: "wake_slo", by: "wake_slo_responder" }),
      p.message_id,
      ts,
    );
    db.prepare(`UPDATE agent_comm_sessions SET updated_at = ? WHERE id = ?`)
      .run(ts, p.session_id);
  })();

  return { matched: true, replied: true, reply_message_id: reply_id, pong_ts };
}
