/**
 * Embedding store — DB layer for notes_embeddings (migration 012).
 *
 * recall.ts hybrid path loads the full set of vectors for a workspace,
 * computes cosine similarity in JS against the query vector, and feeds
 * the top-N into RRF fusion against the FTS5 channel. With 737 notes ×
 * 1024 floats this is ~3 MiB of memory per query and ~10ms of CPU —
 * acceptable without sqlite-vec or any extension.
 */
import { db } from "../db/connection.ts";
import {
  EMBED_DIM,
  EMBED_MODEL,
  deserializeEmbedding,
  embedText,
  serializeEmbedding,
  textHash,
} from "./embeddings.ts";
import { logger } from "../utils/logger.ts";

interface EmbeddingRow {
  note_id: string;
  workspace_id: string;
  embedding: Buffer;
  dim: number;
  model: string;
  embedded_at: string;
  text_hash: string;
}

/**
 * Compute and upsert the embedding for a note. Idempotent on
 * (note_id, text_hash, model) — skips the Ollama call when the text
 * hash matches the stored row. Never throws — failures are logged and
 * swallowed so the calling write path (createNote/updateNote) doesn't
 * block on the embedder. The vector channel simply lacks this note
 * until backfill or the next update fixes it.
 */
export async function upsertNoteEmbedding(
  note_id: string,
  workspace_id: string,
  text: string,
): Promise<{ embedded: boolean; skipped: "hash" | null; error?: string }> {
  try {
    const hash = await textHash(text);
    const existing = db
      .prepare(
        `SELECT text_hash, model FROM notes_embeddings WHERE note_id = ? LIMIT 1`,
      )
      .get(note_id) as { text_hash: string; model: string } | undefined;
    if (
      existing &&
      existing.text_hash === hash &&
      existing.model === EMBED_MODEL
    ) {
      return { embedded: false, skipped: "hash" };
    }

    const vec = await embedText(text);
    const blob = serializeEmbedding(vec);
    db.prepare(
      `INSERT INTO notes_embeddings (note_id, workspace_id, embedding, dim, model, text_hash)
       VALUES (?, ?, ?, ?, ?, ?)
       ON CONFLICT(note_id) DO UPDATE SET
         embedding = excluded.embedding,
         dim = excluded.dim,
         model = excluded.model,
         text_hash = excluded.text_hash,
         embedded_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')`,
    ).run(note_id, workspace_id, blob, EMBED_DIM, EMBED_MODEL, hash);
    return { embedded: true, skipped: null };
  } catch (e: any) {
    const msg = e?.message || String(e);
    logger.warn(
      `upsertNoteEmbedding(${note_id}) failed (note still saved): ${msg}`,
    );
    return { embedded: false, skipped: null, error: msg };
  }
}

/** Remove an embedding row — used by tests and admin cleanup. */
export function deleteNoteEmbedding(note_id: string): void {
  db.prepare(`DELETE FROM notes_embeddings WHERE note_id = ?`).run(note_id);
}

/**
 * Load all embeddings for a workspace into memory. Returns rows with
 * Float32Array decoded once so the caller can run cosineSim in a tight
 * loop. Filter on the active model — stale rows from a prior model
 * sit in the table until backfill rebuilds them.
 */
export function loadWorkspaceEmbeddings(workspace_id: string): Array<{
  note_id: string;
  vector: Float32Array;
}> {
  const rows = db
    .prepare(
      `SELECT note_id, embedding FROM notes_embeddings
        WHERE workspace_id = ? AND model = ? AND dim = ?`,
    )
    .all(workspace_id, EMBED_MODEL, EMBED_DIM) as Array<{
    note_id: string;
    embedding: Buffer;
  }>;
  return rows.map((r) => ({
    note_id: r.note_id,
    vector: deserializeEmbedding(r.embedding),
  }));
}

/**
 * Load embeddings across ALL workspaces — used by the
 * cross_workspace+privileged recall path (steward audit).
 */
export function loadAllEmbeddings(): Array<{
  note_id: string;
  workspace_id: string;
  vector: Float32Array;
}> {
  const rows = db
    .prepare(
      `SELECT note_id, workspace_id, embedding FROM notes_embeddings
        WHERE model = ? AND dim = ?`,
    )
    .all(EMBED_MODEL, EMBED_DIM) as Array<{
    note_id: string;
    workspace_id: string;
    embedding: Buffer;
  }>;
  return rows.map((r) => ({
    note_id: r.note_id,
    workspace_id: r.workspace_id,
    vector: deserializeEmbedding(r.embedding),
  }));
}

/**
 * Compute and upsert the embedding for an entity page (Phase 2 Item C,
 * migration 021). Mirrors upsertNoteEmbedding — same idempotency key
 * (text_hash + model), same swallow-and-log failure mode, separate
 * destination table (entity_embeddings) because the FK target is
 * entity_pages(id), not notes(id).
 *
 * The caller is expected to pass a concatenated "title + summary +
 * slug" string so the embedder sees enough surface for bge-m3 to land
 * a useful vector on short entity rows (a one-line title alone
 * collapses to corpus-mean — same problem the noise filter v2 solved
 * for short project labels).
 */
export async function upsertEntityEmbedding(
  entity_id: string,
  workspace_id: string,
  text: string,
): Promise<{ embedded: boolean; skipped: "hash" | null; error?: string }> {
  try {
    const hash = await textHash(text);
    const existing = db
      .prepare(
        `SELECT text_hash, model FROM entity_embeddings WHERE entity_id = ? LIMIT 1`,
      )
      .get(entity_id) as { text_hash: string; model: string } | undefined;
    if (
      existing &&
      existing.text_hash === hash &&
      existing.model === EMBED_MODEL
    ) {
      return { embedded: false, skipped: "hash" };
    }
    const vec = await embedText(text);
    const blob = serializeEmbedding(vec);
    db.prepare(
      `INSERT INTO entity_embeddings (entity_id, workspace_id, embedding, dim, model, text_hash)
       VALUES (?, ?, ?, ?, ?, ?)
       ON CONFLICT(entity_id) DO UPDATE SET
         embedding = excluded.embedding,
         dim = excluded.dim,
         model = excluded.model,
         text_hash = excluded.text_hash,
         embedded_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')`,
    ).run(entity_id, workspace_id, blob, EMBED_DIM, EMBED_MODEL, hash);
    return { embedded: true, skipped: null };
  } catch (e: any) {
    const msg = e?.message || String(e);
    logger.warn(
      `upsertEntityEmbedding(${entity_id}) failed (entity still saved): ${msg}`,
    );
    return { embedded: false, skipped: null, error: msg };
  }
}

/**
 * Load all entity-page embeddings for a workspace (active model only).
 * Mirrors loadWorkspaceEmbeddings for the entity channel; consumed by
 * recall.ts hybrid fusion.
 */
export function loadWorkspaceEntityEmbeddings(
  workspace_id: string,
): Array<{ entity_id: string; vector: Float32Array }> {
  const rows = db
    .prepare(
      `SELECT entity_id, embedding FROM entity_embeddings
        WHERE workspace_id = ? AND model = ? AND dim = ?`,
    )
    .all(workspace_id, EMBED_MODEL, EMBED_DIM) as Array<{
    entity_id: string;
    embedding: Buffer;
  }>;
  return rows.map((r) => ({
    entity_id: r.entity_id,
    vector: deserializeEmbedding(r.embedding),
  }));
}

/** Cross-workspace entity embedding load — privileged recall only. */
export function loadAllEntityEmbeddings(): Array<{
  entity_id: string;
  workspace_id: string;
  vector: Float32Array;
}> {
  const rows = db
    .prepare(
      `SELECT entity_id, workspace_id, embedding FROM entity_embeddings
        WHERE model = ? AND dim = ?`,
    )
    .all(EMBED_MODEL, EMBED_DIM) as Array<{
    entity_id: string;
    workspace_id: string;
    embedding: Buffer;
  }>;
  return rows.map((r) => ({
    entity_id: r.entity_id,
    workspace_id: r.workspace_id,
    vector: deserializeEmbedding(r.embedding),
  }));
}

/** Stats for /healthz and the dashboard — how much of the corpus is embedded. */
export function embeddingCoverage(workspace_id?: string): {
  total_notes: number;
  embedded: number;
  ratio: number;
  model: string;
} {
  const totalSql = workspace_id
    ? `SELECT COUNT(*) AS c FROM notes WHERE deleted_at IS NULL AND workspace_id = ?`
    : `SELECT COUNT(*) AS c FROM notes WHERE deleted_at IS NULL`;
  const embeddedSql = workspace_id
    ? `SELECT COUNT(*) AS c FROM notes_embeddings WHERE workspace_id = ? AND model = ?`
    : `SELECT COUNT(*) AS c FROM notes_embeddings WHERE model = ?`;
  const total = workspace_id
    ? (db.prepare(totalSql).get(workspace_id) as { c: number }).c
    : (db.prepare(totalSql).get() as { c: number }).c;
  const embedded = workspace_id
    ? (
        db.prepare(embeddedSql).get(workspace_id, EMBED_MODEL) as { c: number }
      ).c
    : (db.prepare(embeddedSql).get(EMBED_MODEL) as { c: number }).c;
  return {
    total_notes: total,
    embedded,
    ratio: total === 0 ? 0 : Number((embedded / total).toFixed(3)),
    model: EMBED_MODEL,
  };
}
