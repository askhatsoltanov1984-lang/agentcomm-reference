import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import {
  registerTools,
  type ToolProfile,
  type AgentToolProfile,
} from "./tools.ts";
import type { OAuthScope } from "../auth/oauth.ts";
import type { AuthContext } from "../auth/middleware.ts";

export function createMcpServer(
  authProvider: () => AuthContext | null,
  profile: ToolProfile = "full",
  opts?: {
    isSteward?: boolean;
    agentToolProfile?: AgentToolProfile;
    grantedScope?: OAuthScope[];
  },
): McpServer {
  const server = new McpServer({
    name: "qoopia",
    version: "3.0.0",
  });
  registerTools(server, authProvider, profile, opts);
  return server;
}
