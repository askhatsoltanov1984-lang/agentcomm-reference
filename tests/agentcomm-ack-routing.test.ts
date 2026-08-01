/**
 * agent-comm ack_required + reply-routing integrity (Liam, 2026-06-23).
 * Reviewed/GREEN by Leo. Covers:
 *  - request defaults ack_required=true
 *  - reply is no-ack, auto-acks its parent, and is REROUTED to the requester
 *    even when the sender mis-addresses a third agent (the corsair-main bug)
 *  - explicit ack_required=false stays false
 *  - loop/liveness signals (REVERSE_WAKE_OK / WAKE_SLO_* / *PONG* / CORSAIR_LOOP_TERMINATE) no-ack
 *  - status/system kinds no-ack
 *  - agentAck idempotent (already-acked = no-op, no duplicate ack row)
 *
 * Guard: unset wake-push webhook envs BEFORE importing agent-comm, else the
 * per-test agentSend would issue REAL HTTP POSTs to prod bridges.
 */
delete process.env.AGENTCOMM_CORSAIR_MAIN_WEBHOOK_URL;
delete process.env.AGENTCOMM_CORSAIR_MAIN_WEBHOOK_TOKEN;
delete process.env.AGENTCOMM_LEO_WEBHOOK_URL;
delete process.env.AGENTCOMM_LEO_WEBHOOK_SECRET;
delete process.env.AGENTCOMM_LEO_WEBHOOK_TOKEN;
delete process.env.AGENTCOMM_LIAM_WEBHOOK_URL;
delete process.env.AGENTCOMM_LIAM_WEBHOOK_TOKEN;

import { beforeAll, describe, expect, test } from "bun:test";
import { runMigrations } from "../src/db/migrate.ts";
import { createWorkspace } from "../src/admin/workspaces.ts";
import { createAgent } from "../src/admin/agents.ts";
import { db } from "../src/db/connection.ts";
import { agentSend, agentReply, agentAck, agentInbox } from "../src/services/agent-comm.ts";

let WS = "";
let LIAM = "";
let LEO = "";
let CORSAIR = "";

beforeAll(() => {
  runMigrations();
  const ws = createWorkspace({ name: "ack-routing-test" });
  WS = ws.id;
  LIAM = createAgent({ name: "liam-t", workspaceSlug: ws.slug, type: "steward" }).id;
  LEO = createAgent({ name: "leo-t", workspaceSlug: ws.slug }).id;
  CORSAIR = createAgent({ name: "corsair-t", workspaceSlug: ws.slug }).id;
});

const unacked = (agentId: string) =>
  agentInbox({ workspace_id: WS, agent_id: agentId, status: "unacked", limit: 100 }).items.length;
const row = (id: string): any => db.prepare("SELECT * FROM agent_comm_messages WHERE id = ?").get(id);

describe("agent-comm ack_required + routing", () => {
  test("request defaults ack_required=true", () => {
    const r = agentSend({ workspace_id: WS, agent_id: LIAM, to_agent: "leo-t", body: "do X" });
    expect(r.ack_required).toBe(true);
    expect(row(r.id).ack_required).toBe(1);
  });

  test("reply: no-ack + auto-acks parent + reroutes to requester despite mis-address", () => {
    const req = agentSend({ workspace_id: WS, agent_id: LIAM, to_agent: "leo-t", body: "please review" });
    expect(unacked(LEO)).toBeGreaterThan(0);
    // Leo replies but MIS-ADDRESSES corsair-t (exact prod bug). Server must reroute to LIAM.
    const rep = agentReply({ workspace_id: WS, agent_id: LEO, session_id: req.session_id, to_agent: "corsair-t", reply_to_message_id: req.id, body: "GREEN" });
    expect(rep.ack_required).toBe(false);
    const rr = row(rep.id);
    expect(rr.recipient_agent_id).toBe(LIAM);
    expect(rr.recipient_agent_id).not.toBe(CORSAIR);
    expect(row(req.id).acked_at).not.toBeNull();
    expect(row(req.id).ack_status).toBe("auto_acked_by_reply");
    expect(unacked(CORSAIR)).toBe(0);
  });

  test("explicit ack_required=false stays false", () => {
    const r = agentSend({ workspace_id: WS, agent_id: LIAM, to_agent: "leo-t", body: "fyi", ack_required: false });
    expect(r.ack_required).toBe(false);
  });

  test("loop/liveness signals are no-ack", () => {
    for (const b of ["REVERSE_WAKE_OK 2026-06-23", "WAKE_SLO_PING x", "WAKE_SLO_PONG x", "CORSAIR_LOOP_TERMINATE"]) {
      const r = agentSend({ workspace_id: WS, agent_id: LEO, to_agent: "liam-t", body: b });
      expect(r.ack_required).toBe(false);
    }
  });

  test("status/system kinds are no-ack", () => {
    expect(agentSend({ workspace_id: WS, agent_id: LEO, to_agent: "liam-t", body: "s", kind: "status" }).ack_required).toBe(false);
    expect(agentSend({ workspace_id: WS, agent_id: LEO, to_agent: "liam-t", body: "y", kind: "system" }).ack_required).toBe(false);
  });

  test("agentAck is idempotent (already-acked = no-op)", () => {
    const req = agentSend({ workspace_id: WS, agent_id: LIAM, to_agent: "leo-t", body: "need ack" });
    const a1: any = agentAck({ workspace_id: WS, agent_id: LEO, message_id: req.id, status: "done" });
    expect(a1.acked_at).toBeTruthy();
    const a2: any = agentAck({ workspace_id: WS, agent_id: LEO, message_id: req.id, status: "done" });
    expect(a2.noop).toBe(true);
  });
});
