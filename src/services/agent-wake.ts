import { createHmac } from "node:crypto";
import { db } from "../db/connection.ts";
import { nowIso } from "../utils/errors.ts";
import { logger } from "../utils/logger.ts";
import { logActivity } from "./activity.ts";

const DELIVERY_TIMEOUT_MS = 5_000;
const WORKER_INTERVAL_MS = 1_000;
export const AGENT_WAKE_MAX_ATTEMPTS = 5;
const STALE_AFTER_MS = 10 * 60_000;
const BATCH_SIZE = 25;

type WakeStatus = "queued" | "delivered" | "failed" | "ignored";

type WakeEventRow = {
  id: string;
  workspace_id: string;
  target_agent_id: string;
  session_id: string;
  message_id: string;
  status: WakeStatus;
  attempt_count: number;
  last_attempt_at: string | null;
  next_attempt_at: string | null;
  created_at: string;
  sender_agent_id: string;
  sender_name: string;
  target_name: string;
  target_active: number;
  target_workspace_id: string;
  sender_workspace_id: string;
  session_workspace_id: string;
  message_workspace_id: string;
  topic: string;
  body: string;
  kind: string;
  message_created_at: string;
};

type WebhookConfig = {
  url: string;
  token: string;
  auth: "bearer" | "hmac";
};

type WebhookResolution =
  | { config: WebhookConfig }
  | { ignored: "no_webhook_config" | "invalid_webhook_url" | "stale_direct_callback_disabled" };

export type WakeDeliveryResult = {
  event_id: string;
  status: Exclude<WakeStatus, "queued">;
  attempted: boolean;
  http_status?: number;
  error?: string;
};

function envPrefixForAgent(targetName: string): string {
  const normalized = targetName.trim().toLowerCase();
  if (normalized === "leo-agentcomm" || normalized === "leo") {
    return "AGENTCOMM_LEO_WEBHOOK";
  }
  return `AGENTCOMM_${normalized
    .toUpperCase()
    .replace(/[^A-Z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")}_WEBHOOK`;
}

function isStaleLeoDirectCallback(targetName: string, url: URL): boolean {
  const normalized = targetName.trim().toLowerCase();
  return (
    (normalized === "leo" || normalized === "leo-agentcomm") &&
    (url.hostname === "127.0.0.1" || url.hostname === "localhost" || url.hostname === "::1") &&
    url.port === "8645"
  );
}

function resolveWebhookConfig(targetName: string): WebhookResolution {
  const prefix = envPrefixForAgent(targetName);
  const rawUrl = process.env[`${prefix}_URL`];
  const token = process.env[`${prefix}_SECRET`] || process.env[`${prefix}_TOKEN`];
  if (!rawUrl || !token) return { ignored: "no_webhook_config" };

  let url: URL;
  try {
    url = new URL(rawUrl);
  } catch {
    return { ignored: "invalid_webhook_url" };
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    return { ignored: "invalid_webhook_url" };
  }
  if (isStaleLeoDirectCallback(targetName, url)) {
    return { ignored: "stale_direct_callback_disabled" };
  }

  const rawAuth = (process.env[`${prefix}_AUTH`] || "hmac").trim().toLowerCase();
  return {
    config: {
      url: url.toString(),
      token,
      auth: rawAuth === "bearer" ? "bearer" : "hmac",
    },
  };
}

function dueRows(eventId?: string, limit = BATCH_SIZE): WakeEventRow[] {
  const now = nowIso();
  const exhaustedFilter = eventId ? "AND id = ?" : "";
  db.prepare(
    `UPDATE agent_wake_events
        SET status = 'failed',
            next_attempt_at = NULL,
            last_error = COALESCE(last_error, 'attempt_ceiling_exhausted')
      WHERE status = 'queued'
        AND attempt_count >= ?
        AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
        ${exhaustedFilter}`,
  ).run(AGENT_WAKE_MAX_ATTEMPTS, now, ...(eventId ? [eventId] : []));

  const eventFilter = eventId ? "AND w.id = ?" : "";
  const params = eventId ? [now, eventId, limit] : [now, limit];
  return db.prepare(
    `SELECT w.id, w.workspace_id, w.target_agent_id, w.session_id, w.message_id,
            w.status, w.attempt_count, w.last_attempt_at, w.next_attempt_at, w.created_at,
            m.sender_agent_id, sender.name AS sender_name, target.name AS target_name,
            target.active AS target_active, target.workspace_id AS target_workspace_id,
            sender.workspace_id AS sender_workspace_id,
            s.workspace_id AS session_workspace_id, m.workspace_id AS message_workspace_id,
            s.topic, m.body, m.kind, m.created_at AS message_created_at
       FROM agent_wake_events w
       JOIN workspaces ws ON ws.id = w.workspace_id
       JOIN agent_comm_sessions s ON s.id = w.session_id
       JOIN agent_comm_messages m ON m.id = w.message_id
       JOIN agents sender ON sender.id = m.sender_agent_id
       JOIN agents target ON target.id = w.target_agent_id
      WHERE (
              (w.status = 'queued' AND w.attempt_count < ?)
              OR (w.status = 'failed' AND w.attempt_count < ?)
            )
        AND (w.next_attempt_at IS NULL OR w.next_attempt_at <= ?)
        ${eventFilter}
      ORDER BY w.created_at ASC, w.id ASC
      LIMIT ?`,
  ).all(AGENT_WAKE_MAX_ATTEMPTS, AGENT_WAKE_MAX_ATTEMPTS, ...params) as WakeEventRow[];
}

function claim(row: WakeEventRow): number | null {
  const attemptedAt = new Date().toISOString();
  const leaseUntil = new Date(Date.now() + DELIVERY_TIMEOUT_MS * 2).toISOString();
  const info = db.prepare(
    `UPDATE agent_wake_events
        SET attempt_count = attempt_count + 1,
            last_attempt_at = ?,
            next_attempt_at = ?
      WHERE id = ?
        AND status = ?
        AND attempt_count = ?
        AND attempt_count < ?
        AND (next_attempt_at IS NULL OR next_attempt_at <= ?)`,
  ).run(
    attemptedAt,
    leaseUntil,
    row.id,
    row.status,
    row.attempt_count,
    AGENT_WAKE_MAX_ATTEMPTS,
    attemptedAt,
  );
  return info.changes === 1 ? row.attempt_count + 1 : null;
}

function retryAt(attempt: number): string | null {
  if (attempt >= AGENT_WAKE_MAX_ATTEMPTS) return null;
  const delayMs = Math.min(60_000, 1_000 * 2 ** Math.max(0, attempt - 1));
  return new Date(Date.now() + delayMs).toISOString();
}

function recordOutcome(row: WakeEventRow, attempt: number, result: WakeDeliveryResult): void {
  const completedAt = new Date().toISOString();
  db.transaction(() => {
    const update = db.prepare(
      `UPDATE agent_wake_events
          SET status = ?,
              delivered_at = ?,
              next_attempt_at = ?,
              last_error = ?
        WHERE id = ? AND attempt_count = ?`,
    ).run(
      result.status,
      result.status === "delivered" ? completedAt : null,
      result.status === "failed" ? retryAt(attempt) : null,
      result.error ?? null,
      row.id,
      attempt,
    );
    if (update.changes !== 1) return;
    logActivity({
      workspace_id: row.workspace_id,
      agent_id: row.sender_agent_id,
      action: `agent_wake_${result.status}`,
      entity_type: "agent_wake_event",
      entity_id: row.id,
      project_id: null,
      summary: `AgentComm wake ${result.status} for ${row.target_name}`,
      details: {
        to: row.target_name,
        session_id: row.session_id,
        message_id: row.message_id,
        attempt,
        attempted: result.attempted,
        ...(result.http_status === undefined ? {} : { http_status: result.http_status }),
        ...(result.error ? { error: result.error } : {}),
      },
    });
  })();
}

async function deliver(row: WakeEventRow): Promise<WakeDeliveryResult> {
  if (Date.now() - new Date(row.created_at).getTime() > STALE_AFTER_MS) {
    return {
      event_id: row.id,
      status: "ignored",
      attempted: false,
      error: "stale_queued_event",
    };
  }
  if (
    row.target_active !== 1 ||
    row.target_workspace_id !== row.workspace_id ||
    row.sender_workspace_id !== row.workspace_id ||
    row.session_workspace_id !== row.workspace_id ||
    row.message_workspace_id !== row.workspace_id
  ) {
    return {
      event_id: row.id,
      status: "ignored",
      attempted: false,
      error: "inactive_or_workspace_mismatch",
    };
  }

  const resolved = resolveWebhookConfig(row.target_name);
  if ("ignored" in resolved) {
    return {
      event_id: row.id,
      status: "ignored",
      attempted: false,
      error: resolved.ignored,
    };
  }

  const payload = JSON.stringify({
    event_type: "agentcomm_wake",
    from_agent: row.sender_name,
    to_agent: row.target_name,
    session_id: row.session_id,
    message_id: row.message_id,
    topic: row.topic,
    body: row.body,
    created_at: row.message_created_at,
    kind: row.kind,
  });
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (resolved.config.auth === "bearer") {
    headers.Authorization = `Bearer ${resolved.config.token}`;
  } else {
    headers["X-Webhook-Signature"] = createHmac("sha256", resolved.config.token)
      .update(payload)
      .digest("hex");
  }

  try {
    const response = await fetch(resolved.config.url, {
      method: "POST",
      headers,
      body: payload,
      signal: AbortSignal.timeout(DELIVERY_TIMEOUT_MS),
    });
    if (response.ok) {
      return {
        event_id: row.id,
        status: "delivered",
        attempted: true,
        http_status: response.status,
      };
    }
    return {
      event_id: row.id,
      status: "failed",
      attempted: true,
      http_status: response.status,
      error: `http_${response.status}`,
    };
  } catch (error) {
    const message = error instanceof Error ? error.name : "transport_error";
    return {
      event_id: row.id,
      status: "failed",
      attempted: true,
      error: message.slice(0, 120),
    };
  }
}

export async function drainAgentWakeQueue(options: {
  eventId?: string;
  limit?: number;
} = {}): Promise<WakeDeliveryResult[]> {
  const rows = dueRows(options.eventId, Math.min(Math.max(options.limit ?? BATCH_SIZE, 1), 100));
  const results: WakeDeliveryResult[] = [];
  for (const row of rows) {
    const attempt = claim(row);
    if (attempt === null) continue;
    const result = await deliver(row);
    recordOutcome(row, attempt, result);
    results.push(result);
  }
  return results;
}

let scheduled = false;

export function scheduleAgentWakeDrain(eventId?: string): void {
  if (scheduled && !eventId) return;
  scheduled = true;
  queueMicrotask(() => {
    void drainAgentWakeQueue({ eventId })
      .catch((error) => logger.warn("AgentComm wake drain failed", { error: String(error) }))
      .finally(() => {
        scheduled = false;
      });
  });
}

let worker: ReturnType<typeof setInterval> | null = null;

export function startAgentWakeWorker(): void {
  if (worker) return;
  scheduleAgentWakeDrain();
  worker = setInterval(() => scheduleAgentWakeDrain(), WORKER_INTERVAL_MS);
  if (typeof worker.unref === "function") worker.unref();
}

export function stopAgentWakeWorker(): void {
  if (!worker) return;
  clearInterval(worker);
  worker = null;
}
