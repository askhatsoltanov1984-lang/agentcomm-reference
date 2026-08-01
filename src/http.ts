import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { AsyncLocalStorage } from "node:async_hooks";
import crypto from "node:crypto";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { createMcpServer } from "./mcp/server.ts";
import { normalizeAgentProfile, riskOf } from "./mcp/tools.ts";
import {
  handleDashboardApi,
  isHttps,
  checkDashboardAuth,
  originAllowed as dashboardOriginAllowed,
  dashboardOriginDiagnostics,
  type DashboardAuth,
} from "./dashboard-api.ts";
import { authenticate, type AuthContext } from "./auth/middleware.ts";
import { getAllowlist } from "./admin/claude-agents.ts";
import { saveMessage } from "./services/sessions.ts";
import { fileUpload, fileDelete } from "./services/files.ts";
import {
  wellKnownAuthorizationServer,
  wellKnownProtectedResource,
  createAuthorizationCode,
  exchangeCodeForTokens,
  refreshTokens,
  revokeTokenForClient,
  registerClient,
  assertCanRegisterOAuth,
  getClient,
  createConsentTicket,
  getConsentTicket,
  consentTicketStatus,
  approveConsentTicket,
  denyConsentTicket,
  redeemConsentTicket,
  consumeConsentNonce,
  rotateConsentNonce,
  pruneConsentTickets,
  normalizeScope,
  parseGrantedScope,
  describeScope,
  assertValidPkceS256Challenge,
} from "./auth/oauth.ts";
import { QoopiaError } from "./utils/errors.ts";
import { db } from "./db/connection.ts";
import { env } from "./utils/env.ts";
import { logger } from "./utils/logger.ts";
import {
  globalLimiter,
  mcpLimiter,
  ingestLimiter,
  dashboardLimiter,
  authLimiter,
  type RateLimiter,
} from "./utils/rate-limit.ts";
import { audit } from "./utils/audit.ts";

// ADR-017: in-memory consentNonces is gone. Consent is brokered through
// the consent_tickets table; nonces live as `approve_nonce` columns and are
// rotated atomically. The dashboard-side approve POST uses
// consumeConsentNonce() for one-time semantics.
// Short-lived replay cache for OAuth finalize redirects. Browser-mediated
// OAuth can double-hit /oauth/authorize/finalize after a successful consent
// handoff; the ticket must stay single-use, but a duplicate GET should see the
// same redirect instead of surfacing {ticket redeemed} to the operator.
const finalizeRedirectReplay = new Map<string, { location: string; expiresAtMs: number }>();
function rememberFinalizeRedirect(ticketId: string, location: string): void {
  finalizeRedirectReplay.set(ticketId, { location, expiresAtMs: Date.now() + 10 * 60_000 });
}
function replayFinalizeRedirect(ticketId: string): string | null {
  const row = finalizeRedirectReplay.get(ticketId);
  if (!row) return null;
  if (row.expiresAtMs <= Date.now()) {
    finalizeRedirectReplay.delete(ticketId);
    return null;
  }
  return row.location;
}

function oauthAlreadyCompletedHtml(): string {
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Authorization already completed</title>
  <style>
    :root { color-scheme: light dark; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; min-height: 100vh; display: grid; place-items: center; background: Canvas; color: CanvasText; }
    main { max-width: 34rem; padding: 2rem; line-height: 1.5; }
    h1 { font-size: 1.4rem; margin: 0 0 0.75rem; }
    p { margin: 0.5rem 0; opacity: 0.86; }
  </style>
</head>
<body>
  <main>
    <h1>Authorization already completed</h1>
    <p>This OAuth approval link was already used.</p>
    <p>You can close this tab and return to Claude.</p>
    <p>If Claude still does not show the connector, start a new connector flow.</p>
  </main>
</body>
</html>`;
}


// Periodic GC for finalized/expired/denied consent_tickets. Runs on the
// process's main interval; cheap because the SQL is indexed on expires_at.
let _consentTicketGc: ReturnType<typeof setInterval> | null = null;
function startConsentTicketGc(): void {
  if (_consentTicketGc) return;
  _consentTicketGc = setInterval(() => {
    try {
      pruneConsentTickets();
    } catch (err) {
      logger.warn("pruneConsentTickets failed", { error: String(err) });
    }
  }, 60_000);
  // Don't keep the process alive for the GC alone — when the http server
  // closes, the timer should not block test exit.
  if (typeof _consentTicketGc.unref === "function") _consentTicketGc.unref();
}
function stopConsentTicketGc(): void {
  if (_consentTicketGc) {
    clearInterval(_consentTicketGc);
    _consentTicketGc = null;
  }
}

// --- CORS allowlist ---
const ALLOWED_ORIGINS = new Set([
  "https://claude.ai",
  "https://www.claude.ai",
  "https://console.anthropic.com",
  "https://chat.openai.com",
  "https://chatgpt.com",
  "https://www.chatgpt.com",
]);

function getAllowedOrigin(req: IncomingMessage): string {
  const origin = req.headers.origin;
  if (!origin) return "";
  if (ALLOWED_ORIGINS.has(origin)) return origin;
  if (/^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin)) return origin;
  return "";
}

// --- Auth context per-request (no module-level variable, no race condition) ---
const authStorage = new AsyncLocalStorage<AuthContext>();

export function getCurrentAuth(): AuthContext | null {
  return authStorage.getStore() ?? null;
}

// --- Client IP extraction ---
// Trust proxy-hop headers ONLY when TRUST_PROXY=true AND the connection arrives
// from one of TRUSTED_PROXIES (default: loopback). Иначе — socket address, чтобы
// предотвратить header spoofing от сетевого атакующего.
const TRUSTED_PROXIES_SET = new Set(env.TRUSTED_PROXIES);
function getClientIp(req: IncomingMessage): string {
  const remote = req.socket?.remoteAddress || "unknown";
  if (env.TRUST_PROXY && TRUSTED_PROXIES_SET.has(remote)) {
    return (
      (req.headers["cf-connecting-ip"] as string) ||
      (req.headers["x-forwarded-for"] as string)?.split(",")[0]?.trim() ||
      remote
    );
  }
  // Direct connection, untrusted source, or TRUST_PROXY=false — socket only
  return remote;
}

/**
 * Single Node http server hosts:
 *  - /mcp                                (Streamable HTTP MCP, stateless mode)
 *  - /health                             (health check)
 *  - /.well-known/oauth-authorization-server
 *  - /.well-known/oauth-protected-resource
 *  - /oauth/authorize                    (PKCE code flow)
 *  - /oauth/token
 *  - /oauth/revoke
 *
 * We use node:http (via Bun's Node compat) because the MCP SDK's
 * StreamableHTTPServerTransport is built around IncomingMessage/ServerResponse.
 * This keeps the transport layer tiny — we don't reimplement Streamable HTTP.
 */

interface NodeReqWithBody extends IncomingMessage {
  _body?: Buffer;
}

const MAX_BODY_BYTES = 1_048_576; // 1 MB

async function readBody(req: IncomingMessage): Promise<Buffer> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of req) {
    total += (chunk as Buffer).length;
    if (total > MAX_BODY_BYTES) {
      req.destroy();
      throw new Error("payload_too_large");
    }
    chunks.push(chunk as Buffer);
  }
  return Buffer.concat(chunks);
}

const MAX_UPLOAD_BYTES = 104_857_600; // 100 MB per request (dashboard file upload)

async function readBodyLimited(req: IncomingMessage, max: number): Promise<Buffer> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of req) {
    total += (chunk as Buffer).length;
    if (total > max) {
      req.destroy();
      throw new Error("payload_too_large");
    }
    chunks.push(chunk as Buffer);
  }
  return Buffer.concat(chunks);
}

function json(res: ServerResponse, status: number, body: unknown, req?: IncomingMessage) {
  const payload = JSON.stringify(body);
  const headers: Record<string, string> = {
    "content-type": "application/json",
    "content-length": String(Buffer.byteLength(payload)),
    "x-content-type-options": "nosniff",
  };
  if (req) {
    const origin = getAllowedOrigin(req);
    if (origin) {
      headers["access-control-allow-origin"] = origin;
      headers["vary"] = "Origin";
    }
  }
  res.writeHead(status, headers);
  res.end(payload);
}

function text(res: ServerResponse, status: number, body: string, req?: IncomingMessage) {
  const headers: Record<string, string> = {
    "content-type": "text/plain; charset=utf-8",
    "x-content-type-options": "nosniff",
  };
  if (req) {
    const origin = getAllowedOrigin(req);
    if (origin) {
      headers["access-control-allow-origin"] = origin;
      headers["vary"] = "Origin";
    }
  }
  res.writeHead(status, headers);
  res.end(body);
}

function sendHtml(res: ServerResponse, status: number, body: string, req: IncomingMessage) {
  const headers: Record<string, string> = {
    "content-type": "text/html; charset=utf-8",
    "cache-control": "no-store",
    ...securityHeaders(req),
  };
  res.writeHead(status, headers);
  res.end(body);
}

function nodeReqToFetchRequest(req: IncomingMessage, body?: Buffer): Request {
  const url = `http://local${req.url || "/"}`;
  const headers = new Headers();
  for (const [k, v] of Object.entries(req.headers)) {
    if (Array.isArray(v)) headers.set(k, v.join(", "));
    else if (typeof v === "string") headers.set(k, v);
  }
  const init: RequestInit = {
    method: req.method || "GET",
    headers,
  };
  if (body && body.length > 0 && req.method !== "GET" && req.method !== "HEAD") {
    init.body = body;
  }
  return new Request(url, init);
}

// ---------- Dashboard file upload / delete (owner-only) ----------
async function handleFileUpload(req: IncomingMessage, res: ServerResponse) {
  const auth = checkDashboardAuth(req);
  if (!auth) return json(res, 401, { error: "unauthorized" }, req);
  if (!(auth.type === "owner" || auth.type === "steward"))
    return json(res, 403, { error: "forbidden", detail: "owner only" }, req);
  if (!dashboardOriginAllowed(req)) return json(res, 403, { error: "forbidden_origin" }, req);
  let body: Buffer;
  try {
    body = await readBodyLimited(req, MAX_UPLOAD_BYTES);
  } catch {
    return json(res, 413, { error: "payload_too_large", detail: "max 100MB per request" }, req);
  }
  let form: FormData;
  try {
    form = await nodeReqToFetchRequest(req, body).formData();
  } catch {
    return json(res, 400, { error: "bad_multipart" }, req);
  }
  const folderRaw = form.get("folder");
  const folder = (typeof folderRaw === "string" ? folderRaw : "") || "inbox";
  const entries = form.getAll("file").filter((f) => typeof f !== "string") as File[];
  if (!entries.length) return json(res, 400, { error: "no_file", detail: "expected a 'file' field" }, req);
  const uploaded = [];
  for (const file of entries) {
    const buf = Buffer.from(await file.arrayBuffer());
    uploaded.push(
      await fileUpload({
        workspace_id: auth.workspace_id,
        owner_agent_id: auth.agent_id,
        uploaded_by_agent_id: auth.agent_id,
        folder,
        filename: file.name || "file",
        mime: file.type || "application/octet-stream",
        bytes: buf,
      }),
    );
  }
  return json(res, 200, { uploaded }, req);
}

async function handleFileDelete(req: IncomingMessage, res: ServerResponse, id: string) {
  const auth = checkDashboardAuth(req);
  if (!auth) return json(res, 401, { error: "unauthorized" }, req);
  if (!(auth.type === "owner" || auth.type === "steward"))
    return json(res, 403, { error: "forbidden" }, req);
  if (!dashboardOriginAllowed(req)) return json(res, 403, { error: "forbidden_origin" }, req);
  try {
    return json(res, 200, fileDelete({ workspace_id: auth.workspace_id, id, agent_id: auth.agent_id }), req);
  } catch (e) {
    const er = e as { code?: string };
    if (er.code === "NOT_FOUND") return json(res, 404, { error: "not_found" }, req);
    throw e;
  }
}

/**
 * QRERUN-001: refuse to start the HTTP server if /oauth/authorize is reachable
 * but env.ADMIN_SECRET is empty. Without this gate the consent POST handler
 * has no owner-proof to verify, and the previous loopback-fallback was unsafe
 * behind tunnels (cloudflared etc. always present as 127.0.0.1).
 *
 * Throws so launchd surfaces the failure in stderr and the install runbook
 * can prompt the operator to set QOOPIA_ADMIN_SECRET.
 */
export function assertOAuthReady(): void {
  if (env.ADMIN_SECRET) return;
  const msg =
    "QOOPIA_ADMIN_SECRET is not set. /oauth/authorize cannot start without it " +
    "(QRERUN-001 fail-closed). Generate one and add it to your launchd plist " +
    "or shell env: `openssl rand -base64 32`.";
  logger.error(msg);
  throw new Error(msg);
}

export function startHttpServer() {
  assertOAuthReady();
  startConsentTicketGc();
  const httpServer = createServer(async (req, res) => {
    try {
      await handleRequest(req as NodeReqWithBody, res);
    } catch (err) {
      const msg = (err as Error).message || "";
      if (!res.headersSent) {
        if (msg === "payload_too_large") {
          json(res, 413, { error: "payload_too_large", max_bytes: MAX_BODY_BYTES });
        } else {
          logger.error("Request handler failed", { error: String(err) });
          json(res, 500, { error: "internal_error" });
        }
      }
    }
  });

  httpServer.on("close", () => {
    stopConsentTicketGc();
  });

  httpServer.listen(env.PORT, env.HOST, () => {
    const addr = httpServer.address();
    const boundPort =
      addr && typeof addr === "object" ? addr.port : env.PORT;
    logger.info(`Qoopia V3.0 listening on http://${env.HOST}:${boundPort}`);
    if (env.HOST !== "127.0.0.1" && env.HOST !== "::1" && env.HOST !== "localhost") {
      logger.warn(
        `QOOPIA_HOST=${env.HOST} — server is reachable beyond loopback. ` +
          `Ensure firewall/tunnel ACLs are in place; OAuth + Bearer endpoints assume trusted network.`,
      );
    }
  });

  return httpServer;
}

/**
 * Per-route rate-limit guard. Returns true если лимит превышен и 429 уже
 * отправлен — вызывающий должен сразу return.
 */
function rateLimit429(
  limiter: RateLimiter,
  scope: string,
  clientIp: string,
  res: ServerResponse,
): boolean {
  if (limiter.allow(clientIp)) return false;
  audit({ event: "rate_limit_trigger", result: "deny", ip: clientIp, scope });
  res.writeHead(429, {
    "content-type": "application/json",
    "retry-after": String(limiter.retryAfterSec(clientIp)),
  });
  res.end(JSON.stringify({ error: "too_many_requests", scope }));
  return true;
}

async function handleRequest(req: NodeReqWithBody, res: ServerResponse) {
  const rawUrl = req.url || "/";
  // Route by pathname, not the full URL. Browser OAuth consent bounces through
  // /dashboard?next=...; matching against req.url made that valid dashboard
  // URL fall through to {error:"not_found"}.
  let url = rawUrl;
  try {
    url = new URL(rawUrl, "http://local").pathname;
  } catch {
    url = rawUrl.split("?", 1)[0] || "/";
  }
  const method = (req.method || "GET").toUpperCase();
  const clientIp = getClientIp(req);

  // CORS preflight
  if (method === "OPTIONS") {
    const origin = getAllowedOrigin(req);
    const headers: Record<string, string> = {
      "access-control-allow-methods": "GET, POST, DELETE, OPTIONS",
      "access-control-allow-headers": "authorization, content-type, mcp-session-id, mcp-protocol-version",
      "access-control-max-age": "86400",
    };
    if (origin) {
      headers["access-control-allow-origin"] = origin;
      headers["vary"] = "Origin";
    }
    res.writeHead(204, headers);
    res.end();
    return;
  }

  // --- Global safety-net rate limit (1000 req/min per IP) ---
  // Per-route limiters (mcp/ingest/dashboard/auth) срабатывают в своих хэндлерах.
  if (!globalLimiter.allow(clientIp)) {
    res.writeHead(429, {
      "content-type": "application/json",
      "retry-after": String(globalLimiter.retryAfterSec(clientIp)),
    });
    res.end(JSON.stringify({ error: "too_many_requests", scope: "global" }));
    return;
  }

  // --- Health ---
  if (url === "/health") {
    return json(res, 200, {
      status: "ok",
      version: "3.0.0",
      uptime: Math.round(process.uptime()),
    }, req);
  }

  if (url === "/") {
    return text(
      res,
      200,
      `Qoopia V3.0 MCP server\nMCP endpoint: ${env.PUBLIC_URL}/mcp\nHealth: ${env.PUBLIC_URL}/health\nDashboard: ${env.PUBLIC_URL}/dashboard\n`,
      req,
    );
  }

  // --- Dashboard ---
  if (url === "/dashboard") {
    return serveDashboard(req, res);
  }

  // --- OAuth discovery ---
  if (url === "/.well-known/oauth-authorization-server" || url.startsWith("/.well-known/oauth-authorization-server/")) {
    // Some OAuth clients derive RFC 8414 metadata from the protected resource
    // path and request /.well-known/oauth-authorization-server/mcp. The issuer
    // is still the host root, so serve the same metadata instead of 404.
    return json(res, 200, wellKnownAuthorizationServer(), req);
  }
  if (url.startsWith("/.well-known/oauth-protected-resource")) {
    return json(res, 200, wellKnownProtectedResource(), req);
  }

  // --- OAuth endpoints (stricter: 20 req/min per IP) ---
  // ADR-017: /oauth/authorize is now a thin redirect target. It validates
  // params + client + redirect_uri, creates a server-side consent_ticket,
  // and 302s to the dashboard-scoped consent UI. The /oauth/* surface
  // never reads the dashboard cookie (ADR-015 §"the cookie is never
  // attached outside dashboard routes" preserved).
  if (url.startsWith("/oauth/authorize/finalize") && method === "GET") {
    if (rateLimit429(authLimiter, "auth", clientIp, res)) return;
    return handleAuthorizeFinalize(req, res, clientIp);
  }
  if (url.startsWith("/oauth/authorize") && method === "GET") {
    if (rateLimit429(authLimiter, "auth", clientIp, res)) return;
    return handleAuthorizeRedirect(req, res, clientIp);
  }
  if ((url === "/oauth/authorize" || url === "/oauth/authorize/") && method === "POST") {
    // ADR-017: POST /oauth/authorize is gone. Approval lives on the
    // dashboard surface. Explicitly 405 so a stale Claude.ai client or a
    // crawler hitting the old path gets a deterministic error rather than
    // a 404.
    res.writeHead(405, { "content-type": "application/json", allow: "GET" });
    res.end(JSON.stringify({ error: "method_not_allowed" }));
    return;
  }
  if ((url === "/oauth/token" || url === "/oauth/token/") && method === "POST") {
    if (rateLimit429(authLimiter, "auth", clientIp, res)) return;
    const body = await readBody(req);
    return handleToken(req, body, res);
  }
  if ((url === "/oauth/register" || url === "/oauth/register/") && method === "POST") {
    if (rateLimit429(authLimiter, "auth", clientIp, res)) return;
    const body = await readBody(req);
    const fetchReq = nodeReqToFetchRequest(req, body);
    const auth = authenticate(fetchReq);
    if (!auth) {
      if (fetchReq.headers.get("authorization")) {
        audit({ event: "auth_failure", result: "deny", ip: clientIp, scope: "/oauth/register", detail: "invalid Authorization header" });
        return json(res, 401, {
          error: "unauthorized",
          error_description:
            "Bearer api_key required (steward or claude-privileged scope).",
        });
      }
      const publicDcr = resolveTrustedUnauthenticatedDcrAuth(body);
      if (publicDcr) {
        audit({
          event: "oauth_register",
          result: "allow",
          ip: clientIp,
          workspace_id: publicDcr.auth.workspace_id,
          agent_id: publicDcr.auth.agent_id,
          detail: publicDcr.detail,
        });
        return handleRegister(body, res, publicDcr.auth);
      }
      audit({ event: "auth_failure", result: "deny", ip: clientIp, scope: "/oauth/register" });
      return json(res, 401, {
        error: "unauthorized",
        error_description:
          "Bearer api_key required (steward or claude-privileged scope).",
      });
    }
    try {
      assertCanRegisterOAuth(auth);
    } catch (err) {
      if (err instanceof QoopiaError && err.code === "FORBIDDEN") {
        audit({
          event: "oauth_register",
          result: "deny",
          ip: clientIp,
          workspace_id: auth.workspace_id,
          agent_id: auth.agent_id,
          detail: `agent type=${auth.type} cannot register OAuth clients`,
        });
        return json(res, 403, {
          error: "forbidden",
          error_description: err.message,
        });
      }
      throw err;
    }
    audit({
      event: "oauth_register",
      result: "allow",
      ip: clientIp,
      workspace_id: auth.workspace_id,
      agent_id: auth.agent_id,
    });
    return handleRegister(body, res, auth);
  }
  if (url === "/oauth/register" || url === "/oauth/register/") {
    return json(res, 405, { error: "method_not_allowed", allow: "POST" }, req);
  }
  if ((url === "/oauth/token" || url === "/oauth/token/") && method !== "POST") {
    return json(res, 405, { error: "method_not_allowed", allow: "POST" }, req);
  }
  if ((url === "/oauth/revoke" || url === "/oauth/revoke/") && method === "POST") {
    if (rateLimit429(authLimiter, "auth", clientIp, res)) return;
    const body = await readBody(req);
    return handleRevoke(body, res, clientIp);
  }

  // --- Ingest endpoints (ingest-daemon only, 500 req/min per IP) ---
  if (url === "/ingest/allowlist" && method === "GET") {
    if (rateLimit429(ingestLimiter, "ingest", clientIp, res)) return;
    const fetchReq = nodeReqToFetchRequest(req);
    const auth = authenticate(fetchReq);
    if (!auth || auth.type !== "ingest-daemon") {
      audit({ event: "ingest_forbidden", result: "deny", ip: clientIp, scope: "/ingest/allowlist", detail: auth ? `wrong type: ${auth.type}` : "no auth" });
      return json(res, 403, { error: "forbidden", error_description: "ingest-daemon credentials required" }, req);
    }
    // Hard-isolation: каждый tailer получает allowlist только своего workspace.
    return json(res, 200, getAllowlist(auth.workspace_id), req);
  }

  if (url === "/ingest/session" && method === "POST") {
    if (rateLimit429(ingestLimiter, "ingest", clientIp, res)) return;
    const rawBody = await readBody(req);
    const fetchReq = nodeReqToFetchRequest(req, rawBody);
    const auth = authenticate(fetchReq);
    if (!auth || auth.type !== "ingest-daemon") {
      audit({ event: "ingest_forbidden", result: "deny", ip: clientIp, scope: "/ingest/session", detail: auth ? `wrong type: ${auth.type}` : "no auth" });
      return json(res, 403, { error: "forbidden", error_description: "ingest-daemon credentials required" }, req);
    }
    let payload: {
      attributed_agent_id?: string;
      session_id?: string;
      uuid?: string;
      role?: string;
      content?: string;
      timestamp?: string;
      cwd?: string;
      metadata?: Record<string, unknown>;
    };
    try {
      payload = JSON.parse(rawBody.toString("utf8"));
    } catch {
      return json(res, 400, { error: "invalid_json" }, req);
    }

    const { attributed_agent_id, session_id, uuid, role, content } = payload;
    if (!attributed_agent_id || !session_id || !uuid || !role || !content) {
      return json(res, 400, { error: "missing_fields", required: ["attributed_agent_id", "session_id", "uuid", "role", "content"] }, req);
    }
    if (role !== "user" && role !== "assistant") {
      return json(res, 400, { error: "invalid_role", allowed: ["user", "assistant"] }, req);
    }

    // Resolve the target agent's workspace
    const { db: dbConn } = await import("./db/connection.ts");
    const targetAgent = dbConn
      .prepare(`SELECT workspace_id FROM agents WHERE id = ? AND active = 1`)
      .get(attributed_agent_id) as { workspace_id: string } | undefined;
    if (!targetAgent) {
      return json(res, 404, { error: "agent_not_found", agent_id: attributed_agent_id }, req);
    }

    // Hard-isolation guard: ingest-daemon token привязан к своему workspace,
    // запись в чужой workspace запрещена даже если attacker угадал чужой agent ULID.
    if (targetAgent.workspace_id !== auth.workspace_id) {
      audit({
        event: "workspace_mismatch",
        result: "deny",
        ip: clientIp,
        workspace_id: auth.workspace_id,
        agent_id: attributed_agent_id,
        detail: `caller workspace ${auth.workspace_id} tried to write to agent's workspace ${targetAgent.workspace_id}`,
      });
      return json(res, 403, { error: "workspace_mismatch", error_description: "ingest token workspace does not match target agent workspace" }, req);
    }

    try {
      const result = saveMessage({
        workspace_id: targetAgent.workspace_id,
        agent_id: attributed_agent_id,
        session_id,
        role: role as "user" | "assistant",
        content,
        ingest_uuid: uuid,
        metadata: { ingest_cwd: payload.cwd ?? "", ingest_ts: payload.timestamp ?? "", ...(payload.metadata ?? {}) },
      });
      return json(res, 200, result, req);
    } catch (err) {
      const e = err as { code?: string; message?: string };
      if (e.code === "FORBIDDEN") return json(res, 409, { error: "session_conflict", detail: e.message }, req);
      if (e.code === "INVALID_INPUT") return json(res, 400, { error: "invalid_input", detail: e.message }, req);
      throw err;
    }
  }

  // --- Dashboard-scoped OAuth consent bridge (ADR-017) ---
  // These endpoints intentionally live under /api/dashboard so they pick up
  // the qoopia_dash cookie's Path scope. They are intercepted BEFORE
  // handleDashboardApi() because that dispatcher would 404 unknown paths
  // and is not aware of the OAuth-bridge ones.
  if (url.startsWith("/api/dashboard/oauth-consent") && method === "GET") {
    if (rateLimit429(dashboardLimiter, "dashboard", clientIp, res)) return;
    return handleDashboardOAuthConsentGet(req, res);
  }
  if (url === "/api/dashboard/oauth-consent/approve" && method === "POST") {
    if (rateLimit429(dashboardLimiter, "dashboard", clientIp, res)) return;
    const body = await readBody(req);
    return handleDashboardOAuthConsentApprove(req, body, res, clientIp);
  }
  if (url === "/api/dashboard/oauth-consent/deny" && method === "POST") {
    if (rateLimit429(dashboardLimiter, "dashboard", clientIp, res)) return;
    const body = await readBody(req);
    return handleDashboardOAuthConsentDeny(req, body, res, clientIp);
  }
  if (url === "/api/dashboard/oauth/clients" && method === "POST") {
    if (rateLimit429(dashboardLimiter, "dashboard", clientIp, res)) return;
    const body = await readBody(req);
    return handleDashboardRegisterClient(req, body, res, clientIp);
  }

  // --- Dashboard file upload (POST) + delete (DELETE) — owner-only, before GET gate ---
  if (url === "/api/dashboard/files" && method === "POST") {
    if (rateLimit429(dashboardLimiter, "dashboard", clientIp, res)) return;
    return handleFileUpload(req, res);
  }
  {
    const fm = url.match(/^\/api\/dashboard\/files\/([^/]+)$/);
    if (fm && method === "DELETE") {
      if (rateLimit429(dashboardLimiter, "dashboard", clientIp, res)) return;
      return handleFileDelete(req, res, decodeURIComponent(fm[1]!));
    }
  }

  // --- Dashboard API (read-only, 200 req/min per IP) ---
  if (url.startsWith("/api/dashboard")) {
    if (rateLimit429(dashboardLimiter, "dashboard", clientIp, res)) return;
    if (handleDashboardApi(req, res)) return;
  }

  // --- MCP endpoint (300 req/min per IP) ---
  // Accept both /mcp and /mcp/ because some UI clients/browser flows
  // normalize connector URLs by adding a trailing slash.
  if (url === "/mcp" || url === "/mcp/" || url.startsWith("/mcp?") || url.startsWith("/mcp/?")) {
    if (rateLimit429(mcpLimiter, "mcp", clientIp, res)) return;
    return handleMcp(req, res);
  }

  return json(res, 404, { error: "not_found" }, req);
}

// ---------- Dashboard ----------

let dashboardHtml: string | null = null;

/**
 * QSA-G / Codex QSA-007: HTML responses (dashboard + OAuth consent) must
 * carry a hardened CSP and HSTS-on-https so a successful XSS in the rendered
 * page can't exfiltrate or redirect, and downgrade attacks are refused by
 * the browser on subsequent loads.
 *
 * Notes on the policy choices:
 *   - 'unsafe-inline' for script and style is required because dashboard.html
 *     and the consent page both ship a single inline <script> / <style> block.
 *     We trade the script-src strictness for keeping the dashboard a single
 *     self-contained file (no nonces, no extra build step). frame-ancestors
 *     'none' still blocks clickjacking, and form-action 'self' contains
 *     POST exfil via injected <form>.
 *   - HSTS only when isHttps(req) — emitting it on plain http would either
 *     be ignored (per RFC 6797) or, worse, "stick" if the request was
 *     proxied by a TLS-terminating tunnel and break local debugging.
 */
function securityHeaders(req: IncomingMessage, allowNavOrigin?: string): Record<string, string> {
  // OAuth consent fix: the approve form's submission redirects (302 chain:
  // /approve → /oauth/authorize/finalize → client callback) to the registered
  // client redirect_uri, which is cross-origin (e.g. https://claude.ai).
  // Chrome/Safari enforce form-action AND navigate-to across the whole redirect
  // chain, so with a bare 'self' the post-approve cross-origin redirect is
  // SILENTLY BLOCKED — the Approve button appears to do nothing, the client
  // never receives the code, and the browser retries (storm) / shows "already
  // used". The consent page therefore must allow the OAuth client's origin.
  const extra = allowNavOrigin ? ` ${allowNavOrigin}` : "";
  const csp = [
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data:",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'none'",
    "frame-ancestors 'none'",
    `form-action 'self'${extra}`,
    // CSP Level 3 navigation restriction. Browsers that don't implement
    // it ignore the directive (per CSP spec); browsers that do block
    // top-level window.location-style redirects from injected inline
    // script. Allow the OAuth client origin so the post-consent cross-origin
    // redirect is not blocked.
    `navigate-to 'self'${extra}`,
  ].join("; ");
  const headers: Record<string, string> = {
    "content-security-policy": csp,
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
  };
  if (isHttps(req)) {
    headers["strict-transport-security"] =
      "max-age=15552000; includeSubDomains";
  }
  return headers;
}

function serveDashboard(req: IncomingMessage, res: ServerResponse) {
  if (!dashboardHtml) {
    try {
      // Resolve path relative to this source file
      const thisDir = typeof __dirname !== "undefined"
        ? __dirname
        : dirname(fileURLToPath(import.meta.url));
      dashboardHtml = readFileSync(join(thisDir, "public", "dashboard.html"), "utf8");
    } catch {
      res.writeHead(404, { "content-type": "text/plain" });
      res.end("Dashboard not found");
      return;
    }
  }
  res.writeHead(200, {
    "content-type": "text/html; charset=utf-8",
    "cache-control": "no-cache",
    ...securityHeaders(req),
  });
  res.end(dashboardHtml);
}

// ---------- MCP handler ----------

async function handleMcp(req: IncomingMessage, res: ServerResponse) {
  const method = (req.method || "GET").toUpperCase();

  // Authenticate
  const body = method === "GET" || method === "DELETE" ? undefined : await readBody(req);
  const fetchReq = nodeReqToFetchRequest(req, body);
  const auth = authenticate(fetchReq);
  if (!auth) {
    const headers: Record<string, string> = {
      "content-type": "application/json",
      "www-authenticate": `Bearer realm="qoopia", resource_metadata="${env.PUBLIC_URL}/.well-known/oauth-protected-resource"`,
    };
    const origin = getAllowedOrigin(req);
    if (origin) {
      headers["access-control-allow-origin"] = origin;
      headers["access-control-expose-headers"] = "WWW-Authenticate";
      headers["vary"] = "Origin";
    }
    res.writeHead(401, headers);
    res.end(JSON.stringify({ error: "unauthorized" }));
    return;
  }

  // QSA-F / ADR-016: normalize the agent's per-agent tool profile once
  // per request. Unknown / null values are coerced to 'read-only' with
  // a single WARN line, matching the documented fail-closed posture.
  const agentProfile = normalizeAgentProfile(
    auth.tool_profile,
    auth.agent_name,
  );

  // Access log: parse JSON-RPC method/tool name from body for debugging.
  if (body && body.length > 0) {
    try {
      const parsed = JSON.parse(body.toString("utf8"));
      const rpcMethod = parsed.method;
      let detail = "";
      if (rpcMethod === "tools/call" && parsed.params?.name) {
        const toolName = parsed.params.name as string;
        const risk = riskOf(toolName);
        // Risk class makes destructive/admin calls greppable in stderr
        // even when the tool name itself isn't obviously dangerous.
        detail = ` tool=${toolName} risk=${risk ?? "unknown"} profile=${agentProfile}`;
      }
      logger.info(
        `MCP ${rpcMethod || "?"}${detail} agent=${auth.agent_name} (${auth.source})`,
      );
    } catch {
      // ignore — body may be batched or non-json
    }
  }

  // Run inside AsyncLocalStorage so concurrent requests never share auth context
  await authStorage.run(auth, async () => {
    const server = createMcpServer(() => getCurrentAuth(), "full", {
      isSteward: auth.type === "steward",
      agentToolProfile: agentProfile,
      grantedScope: auth.granted_scope,
    });
    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: undefined,
    });
    res.on("close", () => {
      try { transport.close(); } catch {}
      try { server.close(); } catch {}
    });

    await server.connect(transport);
    let parsedBody: unknown;
    if (body && body.length > 0) {
      try {
        parsedBody = JSON.parse(body.toString("utf8"));
      } catch {
        return json(res, 400, { error: "invalid_json", message: "Request body is not valid JSON" }, req);
      }
    }
    await transport.handleRequest(req, res, parsedBody);
  });
}

// ---------- OAuth handlers ----------

// ADR-017: checkAdminSecret() and verifyConsentSecret() are deleted along
// with the consent HTML form. Approval is brokered through a dashboard-side
// POST authenticated by the qoopia_dash cookie (per ADR-015), and OAuth
// client registration is gated on Bearer api_key + steward/claude-priv type.

function parseForm(body: Buffer): Record<string, string> {
  const out: Record<string, string> = {};
  const s = body.toString("utf8");
  for (const pair of s.split("&")) {
    const [k, v = ""] = pair.split("=");
    if (!k) continue;
    let key: string;
    let val: string;
    try {
      key = decodeURIComponent(k);
      val = decodeURIComponent(v.replace(/\+/g, " "));
    } catch {
      // Malformed percent-encoding — throw controlled 400 (caught by callers)
      throw Object.assign(new Error("invalid_request"), { statusCode: 400 });
    }
    out[key] = val;
  }
  return out;
}

/**
 * Parse an application/json token-endpoint body into the same flat string map
 * parseForm produces. Some MCP clients (incl. claude.ai) POST /oauth/token as
 * application/json instead of the RFC6749 x-www-form-urlencoded; without this
 * branch the body was unreadable → grant_type undefined → unsupported_grant_type
 * → no access token (claude.ai connector stuck at code→token exchange).
 */
function parseJsonForm(body: Buffer): Record<string, string> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(body.toString("utf8"));
  } catch {
    throw Object.assign(new Error("invalid_request"), { statusCode: 400 });
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw Object.assign(new Error("invalid_request"), { statusCode: 400 });
  }
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(parsed as Record<string, unknown>)) {
    if (v === undefined || v === null) continue;
    if (typeof v === "string") out[k] = v;
    else if (typeof v === "number" || typeof v === "boolean") out[k] = String(v);
    // nested objects/arrays are ignored — OAuth token params are flat scalars
  }
  return out;
}

/**
 * ADR-017: GET /oauth/authorize.
 *
 * The /oauth/* surface trusts no browser-carried state beyond an opaque
 * ticket id. This handler:
 *   - validates OAuth params + the registered client + the redirect_uri
 *     allowlist (same as before),
 *   - snapshots the parameters into a server-side consent_ticket row,
 *   - 302s the browser to /api/dashboard/oauth-consent?ticket=<id>.
 *
 * The dashboard-scoped consent UI then handles cookie auth, workspace
 * matching, and approval. No HTML is rendered here, no cookies are read.
 */
function handleAuthorizeRedirect(
  req: IncomingMessage,
  res: ServerResponse,
  clientIp: string,
) {
  const u = new URL(req.url || "/", env.PUBLIC_URL);
  const clientId = u.searchParams.get("client_id");
  const redirectUri = u.searchParams.get("redirect_uri");
  const responseType = u.searchParams.get("response_type");
  const codeChallenge = u.searchParams.get("code_challenge");
  const codeChallengeMethod = u.searchParams.get("code_challenge_method") || "S256";
  const state = u.searchParams.get("state") || "";
  let scope = "";
  logger.info(`OAuth authorize ENTER client=${clientId || "<none>"} redirect=${redirectUri || "<none>"} has_state=${state ? "y" : "n"} ua=${(req.headers["user-agent"] || "").slice(0, 40)}`);

  try {
    scope = normalizeScope(u.searchParams.get("scope")).normalized;
  } catch (err) {
    const msg = (err as Error).message || "invalid_scope";
    if (msg.startsWith("invalid_scope:")) {
      return json(res, 400, {
        error: "invalid_scope",
        error_description: `Unknown scope '${msg.slice("invalid_scope:".length)}'`,
      });
    }
    throw err;
  }

  if (!clientId || !redirectUri || responseType !== "code" || !codeChallenge) {
    return json(res, 400, {
      error: "invalid_request",
      error_description:
        "Missing required: client_id, redirect_uri, response_type=code, code_challenge",
    });
  }
  if (codeChallengeMethod !== "S256") {
    return json(res, 400, {
      error: "invalid_request",
      error_description: "Only S256 code_challenge_method is supported",
    });
  }
  try {
    assertValidPkceS256Challenge(codeChallenge);
  } catch {
    return json(res, 400, {
      error: "invalid_request",
      error_description:
        "code_challenge must be a 43-character base64url S256 challenge",
    });
  }

  const client = getClient(clientId);
  if (!client) {
    return json(res, 400, {
      error: "invalid_client",
      error_description: "Unknown client_id — register first via /oauth/register",
    });
  }
  if (!client.redirect_uris.includes(redirectUri)) {
    return json(res, 400, {
      error: "invalid_request",
      error_description: "redirect_uri not registered for this client",
    });
  }
  if (!client.workspace_id) {
    // Legacy oauth_clients row that escaped migration 011's backfill.
    // Refuse rather than silently bind to whatever workspace getClient()
    // would have inferred — fail-closed per ADR-017 §Migration.
    return json(res, 500, {
      error: "server_error",
      error_description:
        "OAuth client is not bound to a workspace. Re-register the client.",
    });
  }

  const ticket = createConsentTicket({
    clientId,
    workspaceId: client.workspace_id,
    redirectUri,
    codeChallenge,
    codeChallengeMethod,
    scope,
    state,
  });

  audit({
    event: "oauth_consent",
    result: "allow",
    ip: clientIp,
    workspace_id: client.workspace_id,
    detail: `client=${clientId} ticket_fp=${fingerprintIdentifier(ticket.id, 12)} created`,
  });

  // Simple ChatGPT connector path: clients that were registered through the
  // public GPT-bound DCR path and are redirecting back to ChatGPT do not need
  // the dashboard consent UI. Auto-approve only this narrow case; arbitrary
  // public DCR callbacks still fall through to the normal dashboard consent.
  if (isPublicDcrAgentId(client.agent_id) && isChatGptRedirectUri(redirectUri)) {
    if (approveConsentTicket(ticket.id, client.agent_id)) {
      audit({
        event: "oauth_consent",
        result: "allow",
        ip: clientIp,
        workspace_id: client.workspace_id,
        agent_id: client.agent_id,
        detail: `client=${clientId} ticket_fp=${fingerprintIdentifier(ticket.id, 12)} auto-approved chatgpt`,
      });
      const target = new URL("/oauth/authorize/finalize", env.PUBLIC_URL);
      target.searchParams.set("ticket", ticket.id);
      res.writeHead(302, {
        location: target.toString(),
        "cache-control": "no-store",
      });
      return res.end();
    }
  }

  const target = new URL("/api/dashboard/oauth-consent", env.PUBLIC_URL);
  target.searchParams.set("ticket", ticket.id);
  res.writeHead(302, {
    location: target.toString(),
    "cache-control": "no-store",
  });
  res.end();
}

/**
 * ADR-017: GET /oauth/authorize/finalize?ticket=...
 *
 * Reads the consent_ticket, requires it to be approved-but-not-redeemed-and-
 * not-expired-and-not-denied, atomically marks redeemed=1, emits the OAuth
 * code, and 302s to client.redirect_uri. Single-use: replaying the URL
 * returns 400.
 *
 * Cookies are NOT read here. The only state trusted is the ticket row,
 * whose `approved_by_agent_id` was set by the dashboard-side approve POST.
 */
function handleAuthorizeFinalize(
  req: IncomingMessage,
  res: ServerResponse,
  clientIp: string,
) {
  const u = new URL(req.url || "/", env.PUBLIC_URL);
  const ticketId = u.searchParams.get("ticket") || "";
  const ticketFp = ticketId ? fingerprintIdentifier(ticketId, 12) : "";
  logger.info(`OAuth finalize ENTER ticket_fp=${ticketFp} ua=${(req.headers["user-agent"] || "").slice(0, 50)}`);
  if (!ticketId) {
    return json(res, 400, {
      error: "invalid_request",
      error_description: "Missing ticket parameter",
    });
  }
  const ticket = getConsentTicket(ticketId);
  if (!ticket) {
    return json(res, 400, {
      error: "invalid_request",
      error_description: "ticket not found",
    });
  }
  if (ticket.redeemed) {
    const replayLocation = replayFinalizeRedirect(ticket.id);
    if (replayLocation) {
      logger.info(`OAuth finalize REPLAY-redirect ticket_fp=${ticketFp}`);
      res.writeHead(302, {
        location: replayLocation,
        "cache-control": "no-store",
      });
      res.end();
      return;
    }
    logger.warn(`OAuth finalize ALREADY-USED (redeemed, no replay entry) ticket_fp=${ticketFp}`);
    return sendHtml(res, 400, oauthAlreadyCompletedHtml(), req);
  }
  if (ticket.denied) {
    return json(res, 400, {
      error: "access_denied",
      error_description: "ticket denied",
    });
  }
  if (ticket.expires_at <= nowIsoUtc()) {
    return json(res, 400, {
      error: "invalid_request",
      error_description: "ticket expired",
    });
  }
  if (!ticket.approved_by_agent_id) {
    return json(res, 400, {
      error: "invalid_request",
      error_description: "ticket not approved",
    });
  }
  const client = getClient(ticket.client_id);
  if (!client) {
    return json(res, 400, { error: "invalid_client" });
  }
  // Codex HIGH #1 (2026-04-28): re-verify the *approving agent's current
  // workspace* still matches the ticket's workspace, not the registering
  // client owner's. The previous version called clientWorkspace(client_id),
  // which resolves the registrar — that doesn't catch the actual drift case
  // (approver moves workspaces between approve and finalize). Also reject
  // if the approver was deactivated in the gap.
  const approver = db
    .prepare(
      `SELECT id, workspace_id, active FROM agents WHERE id = ?`,
    )
    .get(ticket.approved_by_agent_id) as
    | { id: string; workspace_id: string; active: number }
    | undefined;
  if (!approver || !approver.active) {
    audit({
      event: "oauth_consent",
      result: "deny",
      ip: clientIp,
      workspace_id: ticket.workspace_id,
      agent_id: ticket.approved_by_agent_id,
      detail: `finalize: approver missing/inactive ticket_fp=${ticketFp}`,
    });
    return json(res, 400, {
      error: "invalid_request",
      error_description: "approving agent is no longer active",
    });
  }
  if (approver.workspace_id !== ticket.workspace_id) {
    audit({
      event: "workspace_mismatch",
      result: "deny",
      ip: clientIp,
      workspace_id: ticket.workspace_id,
      agent_id: ticket.approved_by_agent_id,
      detail: `finalize: approver moved workspaces ticket=${ticket.workspace_id} approver_now=${approver.workspace_id}`,
    });
    return json(res, 400, {
      error: "invalid_request",
      error_description: "approving agent's workspace no longer matches the ticket",
    });
  }

  // Atomic redeem; fails if a parallel request already redeemed OR if the
  // approver/workspace state drifted between the pre-check SELECT above
  // and this UPDATE. The redeemConsentTicket UPDATE folds the approver
  // active+workspace predicates into the same statement (Codex HIGH #1
  // round 2, 2026-04-28), so this is fully atomic in SQLite. If the
  // pre-check passed but redeem failed, log the race-loss for forensics —
  // the only ways for that to happen are (a) parallel finalize won the
  // race, or (b) approver state flipped in the gap.
  if (!redeemConsentTicket(ticket.id)) {
    logger.warn(`OAuth finalize REDEEM-RACE-LOST (parallel finalize won or approver drift) ticket_fp=${ticketFp}`);
    audit({
      event: "oauth_consent",
      result: "deny",
      ip: clientIp,
      workspace_id: ticket.workspace_id,
      agent_id: ticket.approved_by_agent_id,
      detail: `finalize: redeem race lost or approver drift after pre-check ticket_fp=${ticketFp}`,
    });
    return sendHtml(res, 400, oauthAlreadyCompletedHtml(), req);
  }

  const code = createAuthorizationCode({
    clientId: ticket.client_id,
    // Bind the OAuth code to the *approving* agent's id so the resulting
    // token is workspace-scoped to the operator who approved (ADR-017 §4).
    agentId: ticket.approved_by_agent_id,
    workspaceId: ticket.workspace_id,
    codeChallenge: ticket.code_challenge,
    codeChallengeMethod: ticket.code_challenge_method,
    redirectUri: ticket.redirect_uri,
    grantedScope: ticket.scope,
  });

  audit({
    event: "oauth_consent",
    result: "allow",
    ip: clientIp,
    workspace_id: ticket.workspace_id,
    agent_id: ticket.approved_by_agent_id,
    detail: `client=${ticket.client_id} ticket_fp=${ticketFp} finalized`,
  });

  const url = new URL(ticket.redirect_uri);
  url.searchParams.set("code", code);
  if (ticket.state) url.searchParams.set("state", ticket.state);
  // RFC 9207 Authorization Server Issuer Identification — strict OAuth 2.1 / MCP
  // clients (incl. claude.ai) require `iss` in the authorization response and
  // silently drop the callback (no token exchange) when it is absent.
  url.searchParams.set("iss", env.OAUTH_ISSUER);
  const redirectLocation = url.toString();
  rememberFinalizeRedirect(ticket.id, redirectLocation);
  logger.info(`OAuth finalize SUCCESS → 302 host=${url.host} path=${url.pathname} has_state=${ticket.state ? "y" : "n"} ticket_fp=${ticketFp}`);
  res.writeHead(302, {
    location: redirectLocation,
    "cache-control": "no-store",
  });
  res.end();
}

const CLAUDE_AI_REDIRECT_URI = "https://claude.ai/api/mcp/auth_callback";
const CLAUDE_PRIVILEGED_AGENT_ID = "01CLAUDE0CODE0AGENT0000001";
const PUBLIC_DCR_AGENT_NAME = process.env.QOOPIA_PUBLIC_DCR_AGENT_NAME || "GPT";
const PUBLIC_DCR_WORKSPACE_SLUG = process.env.QOOPIA_PUBLIC_DCR_WORKSPACE_SLUG || "default";
const CHATGPT_DCR_REDIRECT_HOSTS = new Set([
  "chat.openai.com",
  "chatgpt.com",
  "www.chatgpt.com",
]);

function stringArrayEquals(actual: unknown, expected: string[]): boolean {
  return (
    Array.isArray(actual) &&
    actual.length === expected.length &&
    actual.every((value, index) => value === expected[index])
  );
}

function stringArraySubsetOf(actual: unknown, allowed: string[]): boolean {
  return (
    actual === undefined ||
    (Array.isArray(actual) &&
      actual.length > 0 &&
      actual.every((value) => typeof value === "string" && allowed.includes(value)))
  );
}

function isChatGptRedirectUri(value: string): boolean {
  try {
    const url = new URL(value);
    return (
      url.protocol === "https:" &&
      CHATGPT_DCR_REDIRECT_HOSTS.has(url.hostname.toLowerCase())
    );
  } catch {
    return false;
  }
}

function isChatGptRedirectArray(actual: unknown): boolean {
  return (
    Array.isArray(actual) &&
    actual.length > 0 &&
    actual.every((value) => typeof value === "string" && isChatGptRedirectUri(value))
  );
}

function isPublicDcrAgentId(agentId: string): boolean {
  const row = db
    .prepare(
      `SELECT a.id
         FROM agents a
         JOIN workspaces w ON w.id = a.workspace_id
        WHERE a.name = ? AND w.slug = ? AND a.active = 1 AND a.type = 'claude-privileged'
        LIMIT 1`,
    )
    .get(PUBLIC_DCR_AGENT_NAME, PUBLIC_DCR_WORKSPACE_SLUG) as
    | { id: string }
    | undefined;
  return row?.id === agentId;
}

function authContextFromAgentRow(row: {
  id: string;
  name: string;
  workspace_id: string;
  type: string;
  tool_profile?: string | null;
}): AuthContext {
  return {
    agent_id: row.id,
    agent_name: row.name,
    workspace_id: row.workspace_id,
    type: row.type,
    source: "api-key",
    tool_profile: row.tool_profile ?? null,
  };
}

function resolveChatGptUnauthenticatedDcrAuth(parsed: Record<string, unknown>): AuthContext | null {
  if (!isChatGptRedirectArray(parsed.redirect_uris)) return null;
  if (
    parsed.token_endpoint_auth_method !== undefined &&
    parsed.token_endpoint_auth_method !== "none" &&
    parsed.token_endpoint_auth_method !== "client_secret_post" &&
    parsed.token_endpoint_auth_method !== "client_secret_basic"
  ) {
    return null;
  }
  if (!stringArraySubsetOf(parsed.grant_types, ["authorization_code", "refresh_token"])) {
    return null;
  }
  if (!stringArraySubsetOf(parsed.response_types, ["code"])) return null;

  const row = db
    .prepare(
      `SELECT a.id, a.name, a.workspace_id, a.type, a.tool_profile
         FROM agents a
         JOIN workspaces w ON w.id = a.workspace_id
        WHERE a.name = ? AND w.slug = ? AND a.active = 1 AND a.type = 'claude-privileged'
        LIMIT 1`,
    )
    .get(PUBLIC_DCR_AGENT_NAME, PUBLIC_DCR_WORKSPACE_SLUG) as
    | { id: string; name: string; workspace_id: string; type: string; tool_profile?: string | null }
    | undefined;

  return row ? authContextFromAgentRow(row) : null;
}

function resolvePublicDcrAgentAuth(): AuthContext | null {
  const row = db
    .prepare(
      `SELECT a.id, a.name, a.workspace_id, a.type, a.tool_profile
         FROM agents a
         JOIN workspaces w ON w.id = a.workspace_id
        WHERE a.name = ? AND w.slug = ? AND a.active = 1 AND a.type = 'claude-privileged'
        LIMIT 1`,
    )
    .get(PUBLIC_DCR_AGENT_NAME, PUBLIC_DCR_WORKSPACE_SLUG) as
    | { id: string; name: string; workspace_id: string; type: string; tool_profile?: string | null }
    | undefined;

  return row ? authContextFromAgentRow(row) : null;
}

type TrustedDcrAuth = { auth: AuthContext; detail: string };

function resolveTrustedUnauthenticatedDcrAuth(body: Buffer): TrustedDcrAuth | null {
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(body.toString("utf8"));
  } catch {
    return null;
  }

  const claudeAiAuth = resolveClaudeAiUnauthenticatedDcrAuthParsed(parsed);
  if (claudeAiAuth) {
    return { auth: claudeAiAuth, detail: "claude.ai restricted unauthenticated DCR" };
  }

  const chatGptAuth = resolveChatGptUnauthenticatedDcrAuth(parsed);
  if (chatGptAuth) {
    return { auth: chatGptAuth, detail: "chatgpt restricted unauthenticated DCR" };
  }

  // Simple public DCR path for external clients such as ChatGPT.
  // It only creates an OAuth client bound to the dedicated GPT agent; MCP access
  // still requires the normal OAuth authorization/token flow.
  const publicAuth = resolvePublicDcrAgentAuth();
  if (publicAuth) {
    return { auth: publicAuth, detail: "public unauthenticated DCR bound to GPT agent" };
  }

  return null;
}

/**
 * Claude.ai still relies on RFC 7591 Dynamic Client Registration when adding
 * a custom MCP connector. ADR-017 correctly closed generic unauthenticated DCR,
 * but that also broke Claude.ai self-registration. Keep the unauthenticated
 * exception intentionally tiny: only Claude's fixed callback URL, public or
 * client_secret_post auth, authorization_code/refresh_token grants, code response, and only bound
 * to the default claude-privileged agent.
 */
function resolveClaudeAiUnauthenticatedDcrAuthParsed(parsed: Record<string, unknown>): AuthContext | null {
  if (!stringArrayEquals(parsed.redirect_uris, [CLAUDE_AI_REDIRECT_URI])) return null;
  if (
    parsed.token_endpoint_auth_method !== undefined &&
    parsed.token_endpoint_auth_method !== "none" &&
    parsed.token_endpoint_auth_method !== "client_secret_post"
  ) {
    return null;
  }
  if (!stringArraySubsetOf(parsed.grant_types, ["authorization_code", "refresh_token"])) {
    return null;
  }
  if (!stringArraySubsetOf(parsed.response_types, ["code"])) return null;

  const row = db
    .prepare(
      `SELECT id, name, workspace_id, type, tool_profile
         FROM agents
        WHERE id = ? AND active = 1 AND type = 'claude-privileged'
        LIMIT 1`,
    )
    .get(CLAUDE_PRIVILEGED_AGENT_ID) as
    | { id: string; name: string; workspace_id: string; type: string; tool_profile?: string | null }
    | undefined;

  const fallback = row ?? (db
    .prepare(
      `SELECT id, name, workspace_id, type, tool_profile
         FROM agents
        WHERE active = 1 AND type = 'claude-privileged'
        ORDER BY created_at ASC
        LIMIT 1`,
    )
    .get() as
    | { id: string; name: string; workspace_id: string; type: string; tool_profile?: string | null }
    | undefined);

  if (!fallback) return null;
  return authContextFromAgentRow(fallback);
}

function handleRegister(
  body: Buffer,
  res: ServerResponse,
  auth: AuthContext,
) {
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(body.toString("utf8"));
  } catch {
    return json(res, 400, {
      error: "invalid_request",
      error_description: "Body must be JSON",
    });
  }
  try {
    const out = registerClient(
      {
        client_name: parsed.client_name as string | undefined,
        redirect_uris: parsed.redirect_uris as string[],
        token_endpoint_auth_method:
          parsed.token_endpoint_auth_method as string | undefined,
        grant_types: parsed.grant_types as string[] | undefined,
        response_types: parsed.response_types as string[] | undefined,
      },
      auth,
    );
    logger.info(
      `OAuth register client_id=${out.client_id} name="${out.client_name}" auth=${out.token_endpoint_auth_method} workspace=${auth.workspace_id}`,
    );
    res.writeHead(201, {
      "content-type": "application/json",
      "cache-control": "no-store",
    });
    res.end(
      JSON.stringify({
        ...out,
        client_id_issued_at: Math.floor(Date.now() / 1000),
        ...(out.client_secret ? { client_secret_expires_at: 0 } : {}),
      }),
    );
  } catch (err) {
    return json(res, 400, {
      error: "invalid_request",
      error_description: err instanceof Error ? err.message : String(err),
    });
  }
}

// nowIsoUtc trims sub-second precision to align with `nowIso()` in
// auth/oauth.ts so SQL string compares (`expires_at <= ?`) match.
function nowIsoUtc(): string {
  return new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
}

// ---------- Dashboard-scoped OAuth consent bridge (ADR-017) ----------

function escapeHtmlSafe(s: string): string {
  return escapeHtml(s);
}

/**
 * Codex QSA-H (2026-04-28): the consent surface must reject:
 *
 *   1) OAuth access tokens — checkDashboardAuth() falls back to Bearer when
 *      the cookie is absent, and authenticate() accepts both api-key bearers
 *      AND OAuth access tokens. If we let an OAuth bearer in here, an existing
 *      OAuth client could fetch the consent page, read the rotated nonce, and
 *      self-approve a new ticket — defeating the bridge pattern entirely.
 *
 *   2) standard agents — registration is restricted to steward/claude-priv
 *      (assertCanRegisterOAuth), so consent (which trusts a connector with the
 *      caller's full agent surface) must be at least as restrictive. A standard
 *      agent in the same workspace approving a ticket would mint OAuth tokens
 *      bound to itself.
 *
 * Returns null if eligible, else a short reason code for the caller to
 * translate into the right HTTP shape.
 */
function oauthConsentRejection(
  auth: DashboardAuth,
): "oauth_token_not_accepted" | "consent_requires_admin" | null {
  if (auth.source === "oauth") return "oauth_token_not_accepted";
  if (!auth.isAdmin) return "consent_requires_admin";
  return null;
}

/**
 * GET /api/dashboard/oauth-consent?ticket=<id>
 *
 * The browser was 302'd here from /oauth/authorize. The dashboard cookie
 * auto-attaches because the path is /api/dashboard/*. We:
 *   - look up the ticket; reject if missing/expired/finalized,
 *   - require a verified dashboard cookie (no Bearer fallback) — the
 *     primary user is a browser session,
 *   - check cookie.workspace_id === ticket.workspace_id; mismatch renders
 *     a "wrong workspace" page with no approve button,
 *   - else rotate `approve_nonce` and render the consent UI.
 */
function handleDashboardOAuthConsentGet(
  req: IncomingMessage,
  res: ServerResponse,
) {
  const u = new URL(req.url || "/", env.PUBLIC_URL);
  const ticketId = u.searchParams.get("ticket") || "";
  if (!ticketId) {
    return json(res, 400, {
      error: "invalid_request",
      error_description: "Missing ticket parameter",
    });
  }
  const ticket = getConsentTicket(ticketId);
  const status = consentTicketStatus(ticket);
  if (status === "not_found") {
    return json(res, 404, {
      error: "not_found",
      error_description: "ticket not found",
    });
  }
  if (status !== "ok") {
    if (status === "redeemed") {
      return sendHtml(res, 400, oauthAlreadyCompletedHtml(), req);
    }
    return json(res, 400, {
      error: "invalid_request",
      error_description: `ticket ${status}`,
    });
  }

  const auth = checkDashboardAuth(req);
  if (!auth) {
    // Not logged into dashboard → bounce through /dashboard?next=...
    const next = `/api/dashboard/oauth-consent?ticket=${encodeURIComponent(ticketId)}`;
    const target = `/dashboard?next=${encodeURIComponent(next)}`;
    res.writeHead(302, { location: target, "cache-control": "no-store" });
    return res.end();
  }
  // QSA-H: reject OAuth bearers and standard agents — see oauthConsentRejection().
  const reject = oauthConsentRejection(auth);
  if (reject) {
    audit({
      event: "oauth_consent",
      result: "deny",
      workspace_id: auth.workspace_id,
      agent_id: auth.agent_id,
      detail: `oauth-consent GET rejected reason=${reject} source=${auth.source} type=${auth.type}`,
    });
    return json(res, 403, {
      error: "forbidden",
      error_description:
        reject === "oauth_token_not_accepted"
          ? "OAuth access tokens are not accepted on the consent surface; sign in to the dashboard with a static API key."
          : "OAuth client consent requires a steward or claude-privileged agent.",
    });
  }

  const t = ticket!;
  const client = getClient(t.client_id);
  const safeClientName = escapeHtmlSafe(client?.name || "Unknown client");
  const requestedScopes = parseGrantedScope(t.scope) || [];
  const scopeRows = requestedScopes.length > 0
    ? requestedScopes
        .map(
          (scope) =>
            `<li><strong>${escapeHtmlSafe(scope)}</strong> — ${escapeHtmlSafe(describeScope(scope))}</li>`,
        )
        .join("")
    : `<li><strong>legacy/full agent profile</strong> — this client did not request explicit OAuth scopes, so the token inherits the approving agent's MCP tool profile.</li>`;

  const sharedCss = `
    body { font-family: -apple-system, system-ui, sans-serif; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; background: #0a0a0f; color: #e0e0e0; }
    .card { background: #1a1a2e; border-radius: 16px; padding: 40px; max-width: 480px; width: 90%; text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.4); }
    .logo { font-size: 32px; margin-bottom: 16px; }
    h1 { font-size: 20px; margin: 0 0 8px; }
    .client { color: #7c8aff; font-weight: 600; }
    .info { color: #888; font-size: 14px; margin: 16px 0 24px; }
    .info.warn { color: #ffb37c; }
    .scope-list { text-align: left; margin: 20px 0; padding-left: 20px; color: #c7c7d9; }
    .scope-list li { margin: 8px 0; }
    .actions { display: flex; gap: 12px; justify-content: center; margin-top: 16px; }
    button, .btn { padding: 12px 32px; border-radius: 8px; border: none; font-size: 16px; font-weight: 600; cursor: pointer; }
    .approve { background: #7c8aff; color: #fff; }
    .deny { background: #2a2a3e; color: #888; }
    form { display: inline; }
  `;

  if (auth.workspace_id !== t.workspace_id) {
    audit({
      event: "workspace_mismatch",
      result: "deny",
      workspace_id: auth.workspace_id,
      agent_id: auth.agent_id,
      detail: `oauth-consent GET cookie=${auth.workspace_id} ticket=${t.workspace_id}`,
    });
    const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Wrong workspace — Qoopia</title>
  <style>${sharedCss}</style>
</head>
<body>
  <div class="card">
    <div class="logo">⛔</div>
    <h1>Wrong workspace</h1>
    <p class="info warn">You are signed in to one workspace, but <span class="client">${safeClientName}</span> belongs to a different workspace.</p>
    <p class="info">Sign out, then sign in as the agent that registered this connector.</p>
  </div>
</body>
</html>`;
    res.writeHead(403, {
      "content-type": "text/html; charset=utf-8",
      "cache-control": "no-store",
      ...securityHeaders(req),
    });
    return res.end(html);
  }

  // Match — rotate the approve_nonce so each render gets a fresh value.
  const fresh = rotateConsentNonce(t.id);
  if (!fresh) {
    return json(res, 400, {
      error: "invalid_request",
      error_description: "ticket no longer in-flight",
    });
  }

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Authorize — Qoopia</title>
  <style>${sharedCss}</style>
</head>
<body>
  <div class="card">
    <div class="logo">🔑</div>
    <h1>Authorize access</h1>
    <p><span class="client">${safeClientName}</span> wants to connect to your workspace.</p>
    <p class="info">Approving will let this client read and act through your agent.</p>
    <ul class="scope-list">${scopeRows}</ul>
    <div class="actions">
      <form method="POST" action="/api/dashboard/oauth-consent/deny">
        <input type="hidden" name="ticket" value="${escapeHtmlSafe(t.id)}">
        <input type="hidden" name="nonce" value="${escapeHtmlSafe(fresh)}">
        <button type="submit" class="deny">Deny</button>
      </form>
      <form method="POST" action="/api/dashboard/oauth-consent/approve">
        <input type="hidden" name="ticket" value="${escapeHtmlSafe(t.id)}">
        <input type="hidden" name="nonce" value="${escapeHtmlSafe(fresh)}">
        <button type="submit" class="approve">Approve</button>
      </form>
    </div>
  </div>
</body>
</html>`;
  // Allow the OAuth client's callback origin in form-action/navigate-to so the
  // approve form's cross-origin redirect to the client is not CSP-blocked.
  let clientOrigin: string | undefined;
  try {
    clientOrigin = new URL(t.redirect_uri).origin;
  } catch {
    clientOrigin = undefined;
  }
  res.writeHead(200, {
    "content-type": "text/html; charset=utf-8",
    "cache-control": "no-store",
    ...securityHeaders(req, clientOrigin),
  });
  res.end(html);
}

/**
 * POST /api/dashboard/oauth-consent/approve
 *
 * Cookie auth + Origin/Referer match + nonce one-time consume + workspace
 * re-check (defense in depth even though the GET hides the button on
 * mismatch). On success, marks ticket approved by cookie.agent_id and 302s
 * to /oauth/authorize/finalize?ticket=...
 */
function handleDashboardOAuthConsentApprove(
  req: IncomingMessage,
  body: Buffer,
  res: ServerResponse,
  clientIp: string,
) {
  if (!dashboardOriginAllowed(req)) {
    logger.warn("Dashboard OAuth consent approve Origin guard denied", dashboardOriginDiagnostics(req));
    return json(res, 403, {
      error: "forbidden",
      error_description: "Origin not allowed.",
    });
  }
  let form: Record<string, string>;
  try {
    form = parseForm(body);
  } catch {
    return json(res, 400, {
      error: "invalid_request",
      error_description: "malformed form body",
    });
  }
  const ticketId = form.ticket || "";
  const nonce = form.nonce || "";
  if (!ticketId || !nonce) {
    return json(res, 400, {
      error: "invalid_request",
      error_description: "ticket and nonce required",
    });
  }
  logger.info(`OAuth consent APPROVE received ticket_fp=${fingerprintIdentifier(ticketId, 12)}`);

  const auth = checkDashboardAuth(req);
  if (!auth) {
    return json(res, 401, {
      error: "unauthorized",
      error_description: "Dashboard session required.",
    });
  }
  // QSA-H: reject OAuth bearers and standard agents BEFORE any state read.
  const reject = oauthConsentRejection(auth);
  if (reject) {
    audit({
      event: "oauth_consent",
      result: "deny",
      ip: clientIp,
      workspace_id: auth.workspace_id,
      agent_id: auth.agent_id,
      detail: `oauth-consent approve rejected reason=${reject} source=${auth.source} type=${auth.type}`,
    });
    return json(res, 403, {
      error: "forbidden",
      error_description:
        reject === "oauth_token_not_accepted"
          ? "OAuth access tokens are not accepted on the consent surface."
          : "OAuth client consent requires a steward or claude-privileged agent.",
    });
  }
  const ticket = getConsentTicket(ticketId);
  const status = consentTicketStatus(ticket);
  if (status === "not_found") {
    return json(res, 404, {
      error: "not_found",
      error_description: "ticket not found",
    });
  }
  if (status !== "ok") {
    if (status === "redeemed") {
      // Idempotent double-submit recovery. claude.ai's browser POSTs this approve
      // more than once (duplicate consent-form submit). The first approve already
      // ran finalize → redeemed the ticket, minted the code, and stored the
      // finalize redirect. Returning a dead-end "already used" page on the second
      // POST strands the OAuth client: the duplicate POST is the navigation the
      // browser actually displays, so it never follows the code-bearing redirect
      // to the client callback and never exchanges the code (the exact claude.ai
      // failure). Re-issue the SAME finalize redirect (302 → client callback with
      // the code) so whichever approve the browser lands on reaches the callback.
      const replay = replayFinalizeRedirect(ticketId);
      if (replay) {
        logger.info(`OAuth consent APPROVE on redeemed ticket → replay redirect ticket_fp=${fingerprintIdentifier(ticketId, 12)}`);
        res.writeHead(302, { location: replay, "cache-control": "no-store" });
        res.end();
        return;
      }
      logger.warn(`OAuth consent APPROVE on redeemed ticket, no replay entry → already-used ticket_fp=${fingerprintIdentifier(ticketId, 12)}`);
      return sendHtml(res, 400, oauthAlreadyCompletedHtml(), req);
    }
    return json(res, 400, {
      error: "invalid_request",
      error_description: `ticket ${status}`,
    });
  }
  if (auth.workspace_id !== ticket!.workspace_id) {
    audit({
      event: "workspace_mismatch",
      result: "deny",
      ip: clientIp,
      workspace_id: auth.workspace_id,
      agent_id: auth.agent_id,
      detail: `oauth-consent approve cookie=${auth.workspace_id} ticket=${ticket!.workspace_id}`,
    });
    return json(res, 403, {
      error: "forbidden",
      error_description: "Workspace mismatch.",
    });
  }

  // Atomic single-use nonce consume.
  if (!consumeConsentNonce(ticket!.id, nonce)) {
    return json(res, 403, {
      error: "forbidden",
      error_description: "Invalid or expired nonce.",
    });
  }

  if (!approveConsentTicket(ticket!.id, auth.agent_id)) {
    // Lost the race — ticket was approved/denied/redeemed/expired between
    // status check and approve.
    return json(res, 400, {
      error: "invalid_request",
      error_description: "ticket no longer in-flight",
    });
  }

  const ticketFp = fingerprintIdentifier(ticket!.id, 12);
  audit({
    event: "oauth_consent",
    result: "allow",
    ip: clientIp,
    workspace_id: auth.workspace_id,
    agent_id: auth.agent_id,
    detail: `client=${ticket!.client_id} ticket_fp=${ticketFp} approved`,
  });

  const target = new URL("/oauth/authorize/finalize", env.PUBLIC_URL);
  target.searchParams.set("ticket", ticket!.id);
  logger.info(`OAuth consent APPROVED → 302 finalize ticket_fp=${fingerprintIdentifier(ticket!.id, 12)}`);
  res.writeHead(302, {
    location: target.toString(),
    "cache-control": "no-store",
  });
  res.end();
}

/**
 * POST /api/dashboard/oauth-consent/deny
 *
 * Cookie auth + Origin + nonce. Sets denied=1; 302s to client.redirect_uri
 * with error=access_denied&state=... so the OAuth client sees a clean RFC
 * 6749 §4.1.2.1 deny.
 */
function handleDashboardOAuthConsentDeny(
  req: IncomingMessage,
  body: Buffer,
  res: ServerResponse,
  clientIp: string,
) {
  if (!dashboardOriginAllowed(req)) {
    logger.warn("Dashboard OAuth consent deny Origin guard denied", dashboardOriginDiagnostics(req));
    return json(res, 403, {
      error: "forbidden",
      error_description: "Origin not allowed.",
    });
  }
  let form: Record<string, string>;
  try {
    form = parseForm(body);
  } catch {
    return json(res, 400, {
      error: "invalid_request",
      error_description: "malformed form body",
    });
  }
  const ticketId = form.ticket || "";
  const nonce = form.nonce || "";
  if (!ticketId || !nonce) {
    return json(res, 400, {
      error: "invalid_request",
      error_description: "ticket and nonce required",
    });
  }
  const auth = checkDashboardAuth(req);
  if (!auth) {
    return json(res, 401, {
      error: "unauthorized",
      error_description: "Dashboard session required.",
    });
  }
  // QSA-H: same eligibility gate as approve. A standard agent / OAuth bearer
  // can't approve, so they shouldn't be able to deny either — denying still
  // burns the ticket and signals intent into the audit log.
  const reject = oauthConsentRejection(auth);
  if (reject) {
    audit({
      event: "oauth_consent",
      result: "deny",
      ip: clientIp,
      workspace_id: auth.workspace_id,
      agent_id: auth.agent_id,
      detail: `oauth-consent deny rejected reason=${reject} source=${auth.source} type=${auth.type}`,
    });
    return json(res, 403, {
      error: "forbidden",
      error_description:
        reject === "oauth_token_not_accepted"
          ? "OAuth access tokens are not accepted on the consent surface."
          : "OAuth client consent requires a steward or claude-privileged agent.",
    });
  }
  const ticket = getConsentTicket(ticketId);
  const status = consentTicketStatus(ticket);
  if (status === "not_found") {
    return json(res, 404, {
      error: "not_found",
      error_description: "ticket not found",
    });
  }
  if (status !== "ok") {
    return json(res, 400, {
      error: "invalid_request",
      error_description: `ticket ${status}`,
    });
  }
  if (auth.workspace_id !== ticket!.workspace_id) {
    // Codex MED #5 (2026-04-28): audit cross-workspace deny attempts the same
    // way GET/approve mismatches are audited. A wrong-workspace deny is
    // security-relevant — it could be reconnaissance or a confused agent.
    audit({
      event: "workspace_mismatch",
      result: "deny",
      ip: clientIp,
      workspace_id: auth.workspace_id,
      agent_id: auth.agent_id,
      detail: `oauth-consent deny cookie=${auth.workspace_id} ticket=${ticket!.workspace_id}`,
    });
    return json(res, 403, {
      error: "forbidden",
      error_description: "Workspace mismatch.",
    });
  }
  if (!consumeConsentNonce(ticket!.id, nonce)) {
    return json(res, 403, {
      error: "forbidden",
      error_description: "Invalid or expired nonce.",
    });
  }
  if (!denyConsentTicket(ticket!.id)) {
    return json(res, 400, {
      error: "invalid_request",
      error_description: "ticket no longer in-flight",
    });
  }
  const ticketFp = fingerprintIdentifier(ticket!.id, 12);
  audit({
    event: "oauth_consent",
    result: "deny",
    ip: clientIp,
    workspace_id: auth.workspace_id,
    agent_id: auth.agent_id,
    detail: `client=${ticket!.client_id} ticket_fp=${ticketFp} denied`,
  });

  const url = new URL(ticket!.redirect_uri);
  url.searchParams.set("error", "access_denied");
  if (ticket!.state) url.searchParams.set("state", ticket!.state);
  res.writeHead(302, {
    location: url.toString(),
    "cache-control": "no-store",
  });
  res.end();
}

/**
 * POST /api/dashboard/oauth/clients
 *
 * Browser-initiated client registration. Cookie auth required; delegates to
 * registerClient() with the cookie's AuthContext. Pure ergonomic shim — same
 * steward/claude-priv check as /oauth/register.
 */
function handleDashboardRegisterClient(
  req: IncomingMessage,
  body: Buffer,
  res: ServerResponse,
  clientIp: string,
) {
  if (!dashboardOriginAllowed(req)) {
    logger.warn("Dashboard OAuth connect Origin guard denied", dashboardOriginDiagnostics(req));
    return json(res, 403, {
      error: "forbidden",
      error_description: "Origin not allowed.",
    });
  }
  const dauth = checkDashboardAuth(req);
  if (!dauth) {
    return json(res, 401, {
      error: "unauthorized",
      error_description: "Dashboard session required.",
    });
  }
  // Codex CRITICAL #1 (2026-04-28 round 2): explicitly reject OAuth bearers
  // here. checkDashboardAuth() accepts both cookie sessions and Bearer
  // tokens; Bearer can be either api_key or an OAuth access token. The
  // dashboard shim is intended for cookie-driven browser flows, with
  // api_key Bearer accepted for parity with /oauth/register. OAuth bearers
  // must NOT be able to register new clients via this shim, otherwise the
  // source-rejection in assertCanRegisterOAuth() at /oauth/register is
  // trivially bypassable. The forge to source="api-key" below is preserved
  // so the legitimate cookie path (source="cookie") still passes
  // assertCanRegisterOAuth's api-key-only check.
  if (dauth.source === "oauth") {
    audit({
      event: "oauth_register",
      result: "deny",
      ip: clientIp,
      workspace_id: dauth.workspace_id,
      agent_id: dauth.agent_id,
      detail: "dashboard register: OAuth bearer source not accepted",
    });
    return json(res, 403, {
      error: "forbidden",
      error_description:
        "OAuth access tokens cannot register new clients via the dashboard. Use a static API key or browser cookie session.",
    });
  }
  // Build a minimal AuthContext to feed registerClient. We only need
  // {agent_id, workspace_id, type} for the registerClient + assertCanRegister
  // path — the dashboard cookie auth carries exactly those fields. Source
  // is forged to "api-key" here so cookie sessions pass the api-key-only
  // check in assertCanRegisterOAuth(); OAuth bearers were already rejected
  // above.
  const auth: AuthContext = {
    agent_id: dauth.agent_id,
    agent_name: "",
    workspace_id: dauth.workspace_id,
    type: dauth.type,
    source: "api-key",
  };
  try {
    assertCanRegisterOAuth(auth);
  } catch (err) {
    if (err instanceof QoopiaError && err.code === "FORBIDDEN") {
      audit({
        event: "oauth_register",
        result: "deny",
        ip: clientIp,
        workspace_id: auth.workspace_id,
        agent_id: auth.agent_id,
        detail: `dashboard register: agent type=${auth.type}`,
      });
      return json(res, 403, {
        error: "forbidden",
        error_description: err.message,
      });
    }
    throw err;
  }
  audit({
    event: "oauth_register",
    result: "allow",
    ip: clientIp,
    workspace_id: auth.workspace_id,
    agent_id: auth.agent_id,
    detail: "via dashboard",
  });
  return handleRegister(body, res, auth);
}

function jsonToken(res: ServerResponse, status: number, body: unknown) {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": String(Buffer.byteLength(payload)),
    "cache-control": "no-store",
    "pragma": "no-cache",
    "x-content-type-options": "nosniff",
  });
  res.end(payload);
}

function fingerprintIdentifier(value: string, length = 16): string {
  return crypto
    .createHash("sha256")
    .update(value)
    .digest("hex")
    .slice(0, length);
}

function parseTokenBasicAuth(req: IncomingMessage): { client_id: string; client_secret: string } | null {
  const h = req.headers.authorization || "";
  if (!h.toLowerCase().startsWith("basic ")) return null;
  try {
    const decoded = Buffer.from(h.slice(6).trim(), "base64").toString("utf8");
    const idx = decoded.indexOf(":");
    if (idx <= 0) return null;
    return {
      client_id: decodeURIComponent(decoded.slice(0, idx)),
      client_secret: decodeURIComponent(decoded.slice(idx + 1)),
    };
  } catch {
    return null;
  }
}

function handleToken(req: IncomingMessage, body: Buffer, res: ServerResponse) {
  let form: Record<string, string>;
  const contentType = (req.headers["content-type"] || "").toLowerCase();
  try {
    form = contentType.includes("application/json")
      ? parseJsonForm(body)
      : parseForm(body);
  } catch {
    logger.warn("OAuth token invalid_request: malformed body");
    return jsonToken(res, 400, { error: "invalid_request" });
  }
  const basic = parseTokenBasicAuth(req);
  if (basic) {
    form.client_id ||= basic.client_id;
    form.client_secret ||= basic.client_secret;
  }
  const grantType = form.grant_type;
  const formKeys = Object.keys(form).filter((key) => key !== "client_secret" && key !== "code" && key !== "refresh_token").sort();
  logger.info(
    `OAuth token request grant_type=${grantType || "<missing>"} auth=${basic ? "client_secret_basic" : form.client_secret ? "client_secret_post" : "none"} content_type=${req.headers["content-type"] || "<missing>"} keys=${formKeys.join(",")}`,
  );
  try {
    if (grantType === "authorization_code") {
      if (!form.code || !form.code_verifier || !form.redirect_uri || !form.client_id) {
        logger.warn("OAuth token authorization_code invalid_request: missing required field");
        return jsonToken(res, 400, { error: "invalid_request" });
      }
      const out = exchangeCodeForTokens({
        code: form.code,
        codeVerifier: form.code_verifier,
        redirectUri: form.redirect_uri,
        clientId: form.client_id,
        clientSecret: form.client_secret,
      });
      return jsonToken(res, 200, {
        access_token: out.access,
        refresh_token: out.refresh,
        token_type: "Bearer",
        expires_in: out.expiresInSec,
        ...(out.grantedScope ? { scope: out.grantedScope } : {}),
      });
    }
    if (grantType === "refresh_token") {
      if (!form.refresh_token || !form.client_id) {
        logger.warn("OAuth token refresh invalid_request: missing required field");
        return jsonToken(res, 400, { error: "invalid_request" });
      }
      const out = refreshTokens({
        refreshToken: form.refresh_token,
        clientId: form.client_id,
        clientSecret: form.client_secret,
      });
      return jsonToken(res, 200, {
        access_token: out.access,
        refresh_token: out.refresh,
        token_type: "Bearer",
        expires_in: out.expiresInSec,
        ...(out.grantedScope ? { scope: out.grantedScope } : {}),
      });
    }
    logger.warn(`OAuth token unsupported_grant_type grant_type=${grantType || "<missing>"}`);
    return jsonToken(res, 400, { error: "unsupported_grant_type" });
  } catch (err) {
    const msg = (err as Error).message || "invalid_grant";
    logger.warn(`OAuth token failed error=${msg}`);
    return jsonToken(res, 400, { error: msg });
  }
}

function handleRevoke(body: Buffer, res: ServerResponse, clientIp: string) {
  let form: Record<string, string>;
  try {
    form = parseForm(body);
  } catch {
    return json(res, 400, { error: "invalid_request", error_description: "malformed form body" });
  }
  const token = form.token;
  const clientId = form.client_id;
  if (!token) return json(res, 400, { error: "invalid_request", error_description: "token is required" });
  if (!clientId) return json(res, 400, { error: "invalid_request", error_description: "client_id is required" });
  // Revoke audit logs intentionally use a short deterministic fingerprint,
  // not the raw OAuth client_id, so token-shaped ids never land in audit.log.
  const clientFingerprint = fingerprintIdentifier(clientId);
  // Per RFC 7009: confidential clients must supply client_secret.
  // revokeTokenForClient validates the secret and throws "invalid_client" if wrong.
  try {
    revokeTokenForClient(token, clientId, form.client_secret);
    audit({
      event: "oauth_revoke",
      result: "allow",
      ip: clientIp,
      detail: `client_fp=${clientFingerprint}`,
    });
    // RFC 7009 §2.2: always return 200 even if token was not found (avoid enumeration)
    return json(res, 200, {});
  } catch (err) {
    const msg = (err as Error).message || "";
    if (msg === "invalid_client") {
      audit({
        event: "auth_failure",
        result: "deny",
        ip: clientIp,
        scope: "/oauth/revoke",
        detail: `client_fp=${clientFingerprint} authentication failed`,
      });
      return json(res, 401, { error: "invalid_client", error_description: "Client authentication failed" });
    }
    return json(res, 500, { error: "server_error" });
  }
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
