/**
 * Dashboard files feature — service layer (migration 024 + services/files.ts).
 * Covers: text upload + inline read, binary upload + download_path, folders,
 * list, delete, and re-upload overwrite (UNIQUE workspace+folder+filename).
 */
import { beforeAll, describe, expect, test } from "bun:test";
import { runMigrations } from "../src/db/migrate.ts";
import { createWorkspace } from "../src/admin/workspaces.ts";
import { createAgent } from "../src/admin/agents.ts";
import { fileUpload, fileGet, fileList, fileListByFolder, fileListFolders, fileGetForDownload, fileDelete } from "../src/services/files.ts";

let WS = "";
let OWNER = "";

beforeAll(() => {
  runMigrations();
  const ws = createWorkspace({ name: "files-test" });
  WS = ws.id;
  OWNER = createAgent({ name: "owner-t", workspaceSlug: ws.slug }).id;
});

async function up(folder: string, filename: string, mime: string, body: Buffer) {
  return fileUpload({ workspace_id: WS, owner_agent_id: OWNER, uploaded_by_agent_id: OWNER, folder, filename, mime, bytes: body });
}

describe("files service", () => {
  test("text upload is readable inline via file_get", async () => {
    await up("inbox", "plan.md", "text/markdown", Buffer.from("# Plan\nhello world"));
    const g: any = fileGet({ workspace_id: WS, folder: "inbox", filename: "plan.md" });
    expect(g.content).toContain("hello world");
    expect(g.mime).toBe("text/markdown");
  });

  test("octet-stream .txt detected as text by extension", async () => {
    await up("inbox", "note.txt", "application/octet-stream", Buffer.from("just text"));
    const g: any = fileGet({ workspace_id: WS, folder: "inbox", filename: "note.txt" });
    expect(g.content).toBe("just text");
  });

  test("binary file → no inline content, has download_path", async () => {
    await up("inbox", "pic.png", "image/png", Buffer.from([0x89, 0x50, 0x4e, 0x47, 1, 2, 3]));
    const g: any = fileGet({ workspace_id: WS, folder: "inbox", filename: "pic.png" });
    expect(g.content).toBeNull();
    expect(g.download_path).toContain("/api/dashboard/files/");
  });

  test("fileGet by id works", async () => {
    const m: any = await up("docs", "a.txt", "text/plain", Buffer.from("byid"));
    const g: any = fileGet({ workspace_id: WS, id: m.id });
    expect(g.content).toBe("byid");
  });

  test("folders list reflects uploads", () => {
    const f: any = fileListFolders({ workspace_id: WS });
    const names = f.folders.map((x: any) => x.folder);
    expect(names).toContain("inbox");
    expect(names).toContain("docs");
  });

  test("list by folder + MCP fileList", () => {
    const byFolder: any = fileListByFolder({ workspace_id: WS, folder: "inbox" });
    expect(byFolder.files.length).toBeGreaterThanOrEqual(3);
    const mcp: any = fileList({ workspace_id: WS, folder: "inbox" });
    expect(mcp.count).toBeGreaterThanOrEqual(3);
    expect(mcp.files[0]).toHaveProperty("filename");
  });

  test("download returns bytes", async () => {
    const m: any = await up("dl", "blob.bin", "application/octet-stream", Buffer.from("rawbytes"));
    const d = fileGetForDownload({ workspace_id: WS, id: m.id });
    expect(d).not.toBeNull();
    expect(d!.content.toString()).toBe("rawbytes");
  });

  test("re-upload same folder+filename overwrites (no duplicate)", async () => {
    await up("inbox", "plan.md", "text/markdown", Buffer.from("v2 content"));
    const list: any = fileListByFolder({ workspace_id: WS, folder: "inbox" });
    const plans = list.files.filter((f: any) => f.filename === "plan.md");
    expect(plans.length).toBe(1);
    const g: any = fileGet({ workspace_id: WS, folder: "inbox", filename: "plan.md" });
    expect(g.content).toBe("v2 content");
  });

  test("delete removes the file", async () => {
    const m: any = await up("tmp", "gone.txt", "text/plain", Buffer.from("x"));
    fileDelete({ workspace_id: WS, id: m.id, agent_id: OWNER });
    expect(() => fileGet({ workspace_id: WS, id: m.id })).toThrow();
  });

  test("workspace isolation: other workspace cannot read", async () => {
    const m: any = await up("inbox", "secret.txt", "text/plain", Buffer.from("mine"));
    const other = createWorkspace({ name: "files-other" });
    const list: any = fileListByFolder({ workspace_id: other.id, folder: "inbox" });
    expect(list.files.length).toBe(0);
    expect(() => fileGet({ workspace_id: other.id, id: m.id })).toThrow();
  });
});
