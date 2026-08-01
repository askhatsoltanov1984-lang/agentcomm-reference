import { db } from "../db/connection.ts";
import { QoopiaError, safeJsonParse } from "../utils/errors.ts";
import { logger } from "../utils/logger.ts";
import { cosineSim, embedText } from "./embeddings.ts";
import {
  loadAllEmbeddings,
  loadAllEntityEmbeddings,
  loadWorkspaceEmbeddings,
  loadWorkspaceEntityEmbeddings,
} from "./embedding-store.ts";
import { redactQuery } from "./recall_log_redaction.ts";

const MAX_QUERY_CHARS = 1000;

/**
 * RRF (Reciprocal Rank Fusion) constant. k=60 is the value Cormack et
 * al. (2009) report as well-behaved across heterogeneous IR systems
 * and the value most production hybrid-retrieval systems converge on.
 * Larger k flattens the score; smaller k sharpens the top-rank bias.
 *
 * Read at CALL TIME (rrfK()) not module load — capturing at import
 * locks the value for the whole process and breaks tests that mutate
 * QOOPIA_RRF_K. Same precedent as embedEndpoint() in embeddings.ts
 * and getRecallMode() above.
 */
function rrfK(): number {
  return Number(process.env.QOOPIA_RRF_K || 60);
}

/**
 * Number of candidates each channel (FTS5, vector) contributes to RRF
 * fusion before re-ranking. Lowered from 50 → 20 (2026-05-13, noise-
 * filter PR) because the smoke test on prod showed the vector channel's
 * long tail polluting fusion: bge-m3 produces cosine ≈0.40–0.45 for
 * short project labels ("FLOCK", "LOCATIONS") on almost any query, so
 * the bottom half of a 50-row vector channel was noise that still
 * earned a non-trivial RRF weight via 1/(k+rank).
 */
function hybridChannelTopN(): number {
  return Number(process.env.QOOPIA_HYBRID_CHANNEL_TOPN || 20);
}

/**
 * Minimum cosine similarity for a vector-channel candidate to survive.
 * bge-m3 vectors are not centred — random pairs land around 0.30–0.40
 * cosine, so anything below this threshold is statistical noise rather
 * than topical relevance. See docs/cosine-distribution-2026-05-13.txt:
 * the 5 rescued queries ("крашнулся", "упал", "починили", "сломалась",
 * "postmortem") all have their genuinely relevant top results in the
 * 0.44–0.55 band, with the noise tail starting around 0.41–0.42.
 * 0.42 is the conservative midpoint — keeps every legitimate match
 * observed in the probe, cuts the long tail that fed noise into RRF.
 */
function vectorCosineThreshold(): number {
  return Number(process.env.QOOPIA_VECTOR_COSINE_THRESHOLD || 0.42);
}

/**
 * Project-label notes are 1-few-word labels ("FLOCK", "OPENCLAW IS",
 * "LOCATIONS"). bge-m3 averages token vectors, so very-short text
 * collapses near a corpus-mean point and lands at high cosine to almost
 * any query — pure noise in the vector channel. We drop type='project'
 * notes shorter than this many characters BEFORE RRF fusion so they
 * cannot poison the fused ranking. FTS5 is untouched: a user typing the
 * literal label ("FLOCK") will still get the project via the FTS path.
 *
 * 30 chars matches the user's specification — short legitimate notes of
 * other types ("Saule day off Friday", "watchdog node v25 path") pass
 * because they are not type='project'.
 *
 * NOTE: superseded in practice by getVectorTypeFilterMaxLen() below (v2
 * filter — also covers type='task'). Kept for backward compat with the
 * env knob shipped in PR #25; no longer referenced by the filter path.
 */
function projectLabelMinChars(): number {
  return Number(process.env.QOOPIA_PROJECT_LABEL_MIN_CHARS || 30);
}

/**
 * Noise filter v2 (refs PR #25). After PR #25 shipped the type='project'
 * + len<30 filter, re-running the 5 rescued queries showed two queries
 * ("крашнулся", "починили") still 0/3 relevant in top-3 — noise was now
 * coming from short type='task' completion-labels ("Deploy staging
 * health checks", "[Новая] Караганда — приложение", "GitHub Sync
 * настроен") that bge-m3 collapses to corpus-mean just like project
 * labels. Extend the filter to type IN ('project','task') AND
 * length(text) < this knob.
 *
 * Default 40 — probe on prod ~/.qoopia/data/qoopia.db at length<40:
 * task=38, project=7, memory=4. The 45 task+project rows under the
 * threshold were manually reviewed and none carry critical signal that
 * needs vector recall (FTS5 keyword channel still surfaces them).
 *
 * type='memory' is INTENTIONALLY excluded — memory is the primary
 * recall target, and short anchor notes ("Aidan port 18789",
 * "Tailscale IP 100.81.108.26") must remain reachable via vector
 * similarity. Memory hygiene is a separate task.
 *
 * Read at CALL TIME (not module load) — same pattern as rrfK() and
 * getRecallMode(); a module-load const would freeze the value and
 * break per-call env overrides in tests and benchmarks.
 */
function getVectorTypeFilterMaxLen(): number {
  return Number(process.env.QOOPIA_VECTOR_TYPE_FILTER_MAX_LEN || 40);
}

/** Note types subject to the short-length vector-channel filter. */
const VECTOR_TYPE_FILTER_TYPES = new Set(["project", "task"]);

/**
 * Two-tier cross-encoder rerank. Both run in sidecar processes that
 * expose the same {query, documents[]} → {results: [{index, score}]}
 * HTTP contract, so the recall pipeline stays backend-agnostic.
 *
 *  - `deep` (default OFF unless QOOPIA_RERANK_DEFAULT_DEEP=1 or
 *    deep=true per-call) hits the JinaAI cross-encoder sidecar —
 *    bge-reranker-class quality at ~1.5s/call for n=20 RU/EN.
 *    The right balance for interactive recall when enabled.
 *  - `deep_llm` (opt-in per call) hits Claude Haiku via SDK — slower
 *    (~5-15s) but sharper on tricky mono-term / cross-language queries
 *    where pure cross-encoder semantics drift.
 *
 * Both stages are strictly additive: any sidecar failure (timeout, 5xx,
 * network) falls back to the prior RRF ordering. Recall is bounded by
 * QOOPIA_RERANK_{,LLM_}TIMEOUT_MS, then falls back — same contract as
 * the vector channel.
 *
 * Legacy QOOPIA_RERANK_ENABLED is honoured: when set to "1" without any
 * specific {jina, llm} endpoint env, falls back to the historical single
 * endpoint configured via QOOPIA_RERANK_ENDPOINT.
 */
function rerankEnabledLegacy(): boolean {
  return process.env.QOOPIA_RERANK_ENABLED === "1";
}
function rerankDefaultDeep(): boolean {
  // OFF by default; explicit opt-in via QOOPIA_RERANK_DEFAULT_DEEP=1.
  // Safety: a bare deployment with no env vars and no sidecar would
  // otherwise eat QOOPIA_RERANK_TIMEOUT_MS per recall waiting for a
  // missing endpoint before falling back. Default-off keeps backward
  // compat zero-cost; operators opt in once a sidecar is wired up.
  return process.env.QOOPIA_RERANK_DEFAULT_DEEP === "1";
}
function jinaRerankEndpoint(): string {
  return (
    process.env.QOOPIA_RERANK_JINA_ENDPOINT ||
    process.env.QOOPIA_RERANK_ENDPOINT ||
    "http://jina-reranker:8790/rerank"
  );
}
function llmRerankEndpoint(): string {
  return (
    process.env.QOOPIA_RERANK_LLM_ENDPOINT ||
    "http://host.docker.internal:8789/rerank"
  );
}
function rerankTopK(): number {
  return Number(process.env.QOOPIA_RERANK_TOP_K || 30);
}
function rerankTimeoutMs(): number {
  return Number(process.env.QOOPIA_RERANK_TIMEOUT_MS || 5000);
}
function llmRerankTimeoutMs(): number {
  // LLM-judge rerank is materially slower than the cross-encoder; give
  // it its own ceiling so a slow Haiku call doesn't trip the standard
  // (jina) timeout if a caller flips between them.
  return Number(process.env.QOOPIA_RERANK_LLM_TIMEOUT_MS || 45000);
}
function rerankMaxDocChars(): number {
  return Number(process.env.QOOPIA_RERANK_MAX_DOC_CHARS || 2000);
}

export type RecallMode = "fts5" | "hybrid";

/**
 * Read the recall mode from the environment. `fts5` (default) preserves
 * the existing FTS5-only behaviour; `hybrid` adds a bge-m3 vector
 * channel fused with FTS5 via RRF(k=60). Hybrid still requires Ollama
 * up; if the query embedding fails, recall() degrades to FTS5-only and
 * sets `mode='fts5-fallback'` in the response so callers can observe.
 */
export function getRecallMode(): RecallMode {
  const v = (process.env.QOOPIA_RECALL_MODE || "fts5").toLowerCase();
  return v === "hybrid" ? "hybrid" : "fts5";
}

/**
 * Sanitize a free-text query into an FTS5 MATCH expression.
 * Rules:
 *  - Strip FTS5 operators that confuse users (AND/OR/NOT/NEAR when not in quotes)
 *  - Escape double quotes
 *  - Each term gets prefix match (word*)
 *  - Terms joined with OR (any term matches), ranking handled by bm25 via
 *    ORDER BY rank in the SQL caller. Prior behaviour was implicit AND which
 *    zero'd out multi-word recall on morphology/synonym mismatches — see
 *    docs/recall-baseline.txt for the 20-query measurement.
 *  - Truncate to MAX_QUERY_CHARS
 */
export function sanitizeFtsQuery(query: string): string {
  if (!query || !query.trim()) {
    throw new QoopiaError("INVALID_INPUT", "query is required");
  }
  let raw = query.slice(0, MAX_QUERY_CHARS).trim();

  // Remove characters that break FTS5 parsing
  raw = raw.replace(/["`]/g, " ");
  raw = raw.replace(/[()[\]{}]/g, " ");

  // Drop boolean operators (uppercase) so plain typing works. We apply our
  // own OR-join below; user-typed AND/OR/NOT/NEAR are not honoured (yet).
  const terms = raw
    .split(/\s+/)
    .filter((tok) => tok.length > 0)
    .filter((tok) => !/^(AND|OR|NOT|NEAR)$/i.test(tok))
    .map((tok) => tok.toLowerCase())
    .filter((t) => t.length >= 2);

  if (terms.length === 0) {
    throw new QoopiaError("INVALID_INPUT", "query has no usable terms");
  }

  // Prefix match each term, join with OR. Single-term queries collapse to
  // `"term"*` (no OR), unchanged from the prior AND-join behaviour for
  // mono-term cases.
  return terms.map((t) => `"${t}"*`).join(" OR ");
}

export interface RecallParams {
  workspace_id: string;
  /** QRERUN-003 / ADR-014: agent_id of the caller — needed to surface
   *  their own private notes alongside workspace-visibility ones. */
  caller_agent_id: string;
  /** QRERUN-003 / ADR-014: true for steward/claude-privileged; bypasses
   *  the private-note filter. Distinct from `privileged` below, which
   *  controls cross-workspace search. */
  is_admin: boolean;
  query: string;
  limit?: number;
  scope?: "notes" | "activity" | "sessions" | "entities" | "all";
  type?: string;
  project_id?: string;
  cross_workspace?: boolean;
  privileged?: boolean;
  /** Include notes whose metadata.status = 'archived'. Default false — archived
   *  rows are hidden from recall to keep results focused on live state. */
  include_archived?: boolean;
  /**
   * Override the env-derived recall mode. Used by tests and the
   * benchmark harness to force fts5 / hybrid per call without
   * mutating process.env.
   */
  mode?: RecallMode;
  /**
   * Cross-encoder rerank stage on top of the hybrid pool.
   * Default OFF unless QOOPIA_RERANK_DEFAULT_DEEP=1 or this param is
   * set to true per call. Set false to skip rerank entirely (~50ms
   * hybrid only). Ignored when mode='fts5'.
   */
  deep?: boolean;
  /**
   * LLM-judge rerank (Claude Haiku via SDK sidecar). Opt-in per call —
   * use for tricky / high-precision recall where the cross-encoder
   * misses (mono-term cross-language, intent-heavy queries). Slower
   * (~5-15s). When true, overrides `deep` (LLM replaces the
   * cross-encoder rather than running both).
   */
  deep_llm?: boolean;
}

interface ResultRow {
  id: string;
  type: string;
  text: string;
  metadata: unknown;
  project_id: string | null;
  created_at: string;
  workspace_id: string;
  rank: number;
  source?: "notes" | "activity" | "sessions" | "entity";
  /**
   * Phase 2 Item C — populated on `source='entity'` rows so callers can
   * resolve the canonical entity by slug without a second lookup.
   * Undefined for notes/activity/sessions rows.
   */
  slug?: string;
}

/**
 * Run the FTS5 notes query and return candidate rows in bm25 rank
 * order. Factored out of recall() so the hybrid path can call it
 * symmetrically with the vector path.
 */
function ftsNoteCandidates(
  p: RecallParams,
  sanitized: string,
  topN: number,
): Array<ResultRow & { rowid_rank: number }> {
  const canCrossWorkspace = !!(p.cross_workspace && p.privileged);
  const includeArchived = !!p.include_archived;
  const where: string[] = [`notes_fts MATCH ?`, `n.deleted_at IS NULL`];
  const params: any[] = [sanitized];
  if (!canCrossWorkspace) {
    where.push(`n.workspace_id = ?`);
    params.push(p.workspace_id);
  }
  where.push(`(n.visibility = 'workspace' OR n.agent_id = ? OR ? = 1)`);
  params.push(p.caller_agent_id, p.is_admin ? 1 : 0);
  if (p.type) {
    where.push(`n.type = ?`);
    params.push(p.type);
  }
  if (p.project_id) {
    where.push(`n.project_id = ?`);
    params.push(p.project_id);
  }
  if (!includeArchived) {
    where.push(
      `(json_extract(n.metadata, '$.status') IS NULL OR json_extract(n.metadata, '$.status') != 'archived')`,
    );
  }
  const sql = `
    SELECT n.id, n.type, n.text, n.metadata, n.project_id, n.created_at, n.workspace_id, n.agent_id, n.visibility, rank
    FROM notes_fts f
    JOIN notes n ON n.rowid = f.rowid
    WHERE ${where.join(" AND ")}
    ORDER BY rank
    LIMIT ?
  `;
  const rows = db.prepare(sql).all(...params, topN) as Array<{
    id: string;
    type: string;
    text: string;
    metadata: string;
    project_id: string | null;
    created_at: string;
    workspace_id: string;
    agent_id: string | null;
    visibility: string | null;
    rank: number;
  }>;
  return rows.map((r, idx) => ({
    id: r.id,
    type: r.type,
    text: r.text,
    metadata: safeJsonParse(r.metadata, {} as Record<string, unknown>),
    project_id: r.project_id,
    created_at: r.created_at,
    workspace_id: r.workspace_id,
    rank: r.rank,
    source: "notes" as const,
    rowid_rank: idx + 1, // 1-based rank for RRF
  }));
}

/**
 * Run the vector channel for `scope='notes'` (and the notes portion of
 * `scope='all'`). Returns up to topN note_ids ordered by cosine
 * similarity to the query vector. Falls back silently to an empty list
 * if Ollama is unreachable or no embeddings exist for the workspace.
 */
async function vectorNoteCandidates(
  p: RecallParams,
  topN: number,
): Promise<Array<{ note_id: string; sim: number; vector_rank: number }>> {
  try {
    const queryVec = await embedText(p.query);
    const canCrossWorkspace = !!(p.cross_workspace && p.privileged);
    const rows = canCrossWorkspace
      ? loadAllEmbeddings()
      : loadWorkspaceEmbeddings(p.workspace_id);
    if (rows.length === 0) return [];

    // Noise filter step 1 (v2, refs PR #25): drop short project/task
    // notes BEFORE scoring. bge-m3 collapses 1-few-word labels and short
    // completion-strings to corpus-mean-ish vectors that score 0.40+
    // against almost any query, poisoning RRF. type='memory' is NOT
    // covered — short memory anchors ("Aidan port 18789") must remain
    // reachable via vector. Lookup (type, length) for the embedding ids
    // in one round-trip.
    const minLen = getVectorTypeFilterMaxLen();
    const ids = rows.map((r) => r.note_id);
    const placeholders = ids.map(() => "?").join(",");
    const metaRows =
      ids.length === 0
        ? []
        : (db
            .prepare(
              `SELECT id, type, length(text) AS len FROM notes
                WHERE id IN (${placeholders})`,
            )
            .all(...ids) as Array<{ id: string; type: string; len: number }>);
    const dropIds = new Set<string>();
    for (const m of metaRows) {
      if (VECTOR_TYPE_FILTER_TYPES.has(m.type) && m.len < minLen) {
        dropIds.add(m.id);
      }
    }
    const filteredRows = rows.filter((r) => !dropIds.has(r.note_id));

    // Score every surviving candidate. 700 × cosine over 1024-d = ~3ms.
    const scored = filteredRows.map((r) => ({
      note_id: r.note_id,
      sim: cosineSim(queryVec, r.vector),
    }));
    scored.sort((a, b) => b.sim - a.sim);

    // Noise filter step 2: drop everything below the cosine threshold.
    // Applied AFTER the project-label filter and BEFORE topN slicing —
    // so the threshold trims the long tail rather than the head, and
    // topN sees only signal candidates.
    const threshold = vectorCosineThreshold();
    const passed = scored.filter((s) => s.sim >= threshold);

    return passed.slice(0, topN).map((r, idx) => ({
      ...r,
      vector_rank: idx + 1,
    }));
  } catch (e: any) {
    logger.warn(
      `vector channel skipped (FTS5 still returns): ${e?.message || String(e)}`,
    );
    return [];
  }
}

/**
 * RRF (Reciprocal Rank Fusion). For each candidate note that appears
 * in either channel, sum 1/(k + rank) across channels. Higher score
 * wins. Notes that appear in only one channel still rank — the union
 * is what fixes the "вектор знал, FTS5 не знал" zero-recall cases.
 */
function rrfFuse(
  fts: Array<{ id: string; rowid_rank: number }>,
  vec: Array<{ note_id: string; vector_rank: number }>,
  k: number = rrfK(),
): Map<string, number> {
  const scores = new Map<string, number>();
  for (const r of fts) {
    scores.set(r.id, (scores.get(r.id) || 0) + 1 / (k + r.rowid_rank));
  }
  for (const r of vec) {
    scores.set(
      r.note_id,
      (scores.get(r.note_id) || 0) + 1 / (k + r.vector_rank),
    );
  }
  return scores;
}

/**
 * Hydrate the notes portion of a result set: take the fused note_ids,
 * load the rows from `notes`, apply the same visibility / archived /
 * type / project_id filters as ftsNoteCandidates so a row reaching
 * recall() via the vector channel still respects the boundary.
 */
function hydrateFusedNotes(
  p: RecallParams,
  rankedIds: Array<{ id: string; rrf: number }>,
  limit: number,
): ResultRow[] {
  if (rankedIds.length === 0) return [];
  const includeArchived = !!p.include_archived;
  const canCrossWorkspace = !!(p.cross_workspace && p.privileged);
  // Build a parameterised IN clause. SQLite has no array binding, so
  // generate `?,?,?...` placeholders.
  const ids = rankedIds.map((r) => r.id);
  const placeholders = ids.map(() => "?").join(",");
  const where: string[] = [
    `n.id IN (${placeholders})`,
    `n.deleted_at IS NULL`,
  ];
  const params: any[] = [...ids];
  if (!canCrossWorkspace) {
    where.push(`n.workspace_id = ?`);
    params.push(p.workspace_id);
  }
  where.push(`(n.visibility = 'workspace' OR n.agent_id = ? OR ? = 1)`);
  params.push(p.caller_agent_id, p.is_admin ? 1 : 0);
  if (p.type) {
    where.push(`n.type = ?`);
    params.push(p.type);
  }
  if (p.project_id) {
    where.push(`n.project_id = ?`);
    params.push(p.project_id);
  }
  if (!includeArchived) {
    where.push(
      `(json_extract(n.metadata, '$.status') IS NULL OR json_extract(n.metadata, '$.status') != 'archived')`,
    );
  }
  const rows = db
    .prepare(
      `SELECT n.id, n.type, n.text, n.metadata, n.project_id, n.created_at, n.workspace_id
       FROM notes n WHERE ${where.join(" AND ")}`,
    )
    .all(...params) as Array<{
    id: string;
    type: string;
    text: string;
    metadata: string;
    project_id: string | null;
    created_at: string;
    workspace_id: string;
  }>;
  const byId = new Map(rows.map((r) => [r.id, r]));
  const out: ResultRow[] = [];
  for (const { id, rrf } of rankedIds) {
    const r = byId.get(id);
    if (!r) continue; // filtered out by visibility/archived/type
    out.push({
      id: r.id,
      type: r.type,
      text: r.text,
      metadata: safeJsonParse(r.metadata, {} as Record<string, unknown>),
      project_id: r.project_id,
      created_at: r.created_at,
      workspace_id: r.workspace_id,
      // Carry the RRF score in `rank` for transparency. Negate so the
      // existing "smaller rank == better" caller contract holds.
      rank: -rrf,
      source: "notes",
    });
    if (out.length >= limit) break;
  }
  return out;
}

/**
 * Phase 2 Item C — entity channel. Mirrors the notes channel structure:
 * FTS5 candidates from entity_pages_fts, optional vector candidates from
 * entity_embeddings, fused via the same RRF (k=60) when mode='hybrid'.
 * Returns up to `limit` rows tagged `source='entity'` and
 * `type='entity:<entity_type>'` so downstream callers can branch on the
 * type marker without touching the table.
 *
 * Behaviour matches the notes channel:
 *  - Hybrid + zero vector candidates → fts5-fallback (silent, returns
 *    FTS-only ordering).
 *  - FTS5-only mode → pure FTS path.
 *  - Failure of the vector channel (Ollama down) is swallowed; FTS
 *    results still flow.
 *
 * Archived/deprecated entities are excluded unless `include_archived=true`.
 */
async function entityCandidates(
  p: RecallParams,
  sanitized: string,
  topN: number,
): Promise<ResultRow[]> {
  if (topN <= 0) return [];
  const canCrossWorkspace = !!(p.cross_workspace && p.privileged);
  const includeArchived = !!p.include_archived;
  const requestedMode = p.mode || getRecallMode();

  const where: string[] = [`entity_pages_fts MATCH ?`];
  const params: any[] = [sanitized];
  if (!canCrossWorkspace) {
    where.push(`e.workspace_id = ?`);
    params.push(p.workspace_id);
  }
  if (!includeArchived) {
    where.push(`e.status = 'active'`);
  }
  const ftsSql = `
    SELECT e.id, e.workspace_id, e.type, e.slug, e.title, e.summary, e.status,
           e.metadata, e.created_at, e.updated_at, rank
      FROM entity_pages_fts f
      JOIN entity_pages e ON e.rowid = f.rowid
     WHERE ${where.join(" AND ")}
     ORDER BY rank
     LIMIT ?
  `;
  type EntityFtsRow = {
    id: string;
    workspace_id: string;
    type: string;
    slug: string;
    title: string;
    summary: string | null;
    status: string;
    metadata: string;
    created_at: string;
    updated_at: string;
    rank: number;
  };
  const ftsRows = db
    .prepare(ftsSql)
    .all(...params, topN) as EntityFtsRow[];

  // Materialize FTS rows as ResultRow shape so we can return them as-is
  // when hybrid is off, or after RRF when hybrid is on.
  const toResult = (
    r: EntityFtsRow,
    rank: number,
  ): ResultRow & { rowid_rank: number } => ({
    id: r.id,
    type: `entity:${r.type}`,
    text: r.summary ? `${r.title}\n\n${r.summary}` : r.title,
    metadata: {
      ...safeJsonParse(r.metadata, {} as Record<string, unknown>),
      entity_slug: r.slug,
      entity_status: r.status,
      entity_type: r.type,
    },
    project_id: null,
    created_at: r.created_at,
    workspace_id: r.workspace_id,
    rank,
    source: "entity",
    slug: r.slug,
    rowid_rank: 0, // filled below
  });

  if (requestedMode !== "hybrid") {
    return ftsRows.map((r) => {
      const { rowid_rank: _rr, ...clean } = toResult(r, r.rank);
      return clean;
    });
  }

  // Hybrid: also pull entity vector candidates.
  let vecHits: Array<{ entity_id: string; sim: number; vector_rank: number }> = [];
  try {
    const queryVec = await embedText(p.query);
    const rows = canCrossWorkspace
      ? loadAllEntityEmbeddings()
      : loadWorkspaceEntityEmbeddings(p.workspace_id);
    if (rows.length > 0) {
      const scored = rows
        .map((r) => ({
          entity_id: r.entity_id,
          sim: cosineSim(queryVec, r.vector),
        }))
        .filter((x) => x.sim >= vectorCosineThreshold())
        .sort((a, b) => b.sim - a.sim)
        .slice(0, topN);
      vecHits = scored.map((s, idx) => ({
        entity_id: s.entity_id,
        sim: s.sim,
        vector_rank: idx + 1,
      }));
    }
  } catch (e: any) {
    logger.warn(
      `entity vector channel skipped (FTS5 still returns): ${e?.message || String(e)}`,
    );
  }

  if (vecHits.length === 0) {
    return ftsRows.map((r) => {
      const { rowid_rank: _rr, ...clean } = toResult(r, r.rank);
      return clean;
    });
  }

  // RRF fusion on entity ids.
  const k = rrfK();
  const scores = new Map<string, number>();
  ftsRows.forEach((r, idx) => {
    const rrf = 1 / (k + idx + 1);
    scores.set(r.id, (scores.get(r.id) ?? 0) + rrf);
  });
  for (const v of vecHits) {
    const rrf = 1 / (k + v.vector_rank);
    scores.set(v.entity_id, (scores.get(v.entity_id) ?? 0) + rrf);
  }
  const ranked = [...scores.entries()]
    .map(([id, rrf]) => ({ id, rrf }))
    .sort((a, b) => b.rrf - a.rrf)
    .slice(0, topN);

  // Hydrate the fused set (only those not already in ftsRows).
  const ftsIndex = new Map(ftsRows.map((r) => [r.id, r]));
  const missingIds = ranked
    .map((r) => r.id)
    .filter((id) => !ftsIndex.has(id));
  if (missingIds.length > 0) {
    const placeholders = missingIds.map(() => "?").join(",");
    const hydrateSql = `
      SELECT e.id, e.workspace_id, e.type, e.slug, e.title, e.summary,
             e.status, e.metadata, e.created_at, e.updated_at, 0 AS rank
        FROM entity_pages e
       WHERE e.id IN (${placeholders})
         ${canCrossWorkspace ? "" : "AND e.workspace_id = ?"}
         ${includeArchived ? "" : "AND e.status = 'active'"}
    `;
    const hydrateParams = canCrossWorkspace
      ? missingIds
      : [...missingIds, p.workspace_id];
    const more = db.prepare(hydrateSql).all(...hydrateParams) as EntityFtsRow[];
    for (const r of more) ftsIndex.set(r.id, r);
  }

  const out: ResultRow[] = [];
  for (const r of ranked) {
    const row = ftsIndex.get(r.id);
    if (!row) continue;
    // Carry RRF as a NEGATIVE rank so the cross-source "smaller =
    // better" contract holds against bm25 ranks (negative) and the
    // notes-channel `-rrf` convention used by hydrateFusedNotes.
    // Without this negation, hybrid entity hits would always sort
    // BELOW any FTS-only note that came with a negative bm25 rank,
    // re-introducing the appended-after-notes bug from R1.
    const { rowid_rank: _rr, ...clean } = toResult(row, -r.rrf);
    out.push(clean);
    if (out.length >= topN) break;
  }
  return out;
}

/**
 * Cross-encoder / LLM rerank stage. Generic over backend — the caller
 * picks an endpoint URL and timeout per request. Input: hydrated
 * candidates already shortlisted by RRF. Output: same rows reordered
 * by pair-wise relevance. On any sidecar error, returns the input
 * unchanged (mode='rerank-fallback') so the caller stays oblivious.
 */
async function rerankResults(
  query: string,
  candidates: ResultRow[],
  endpoint: string,
  timeoutMs: number,
  backendLabel: string,
): Promise<{ rows: ResultRow[]; mode: "rerank" | "rerank-fallback" }> {
  if (candidates.length <= 1) return { rows: candidates, mode: "rerank" };
  const maxChars = rerankMaxDocChars();
  const documents = candidates.map((c) => c.text.slice(0, maxChars));
  const t0 = performance.now();
  try {
    const resp = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: AbortSignal.timeout(timeoutMs),
      body: JSON.stringify({ query, documents }),
    });
    if (!resp.ok) {
      logger.warn(
        `rerank[${backendLabel}]: sidecar HTTP ${resp.status}; falling back to RRF order`,
      );
      return { rows: candidates, mode: "rerank-fallback" };
    }
    const data = (await resp.json()) as {
      results: Array<{ index: number; score: number }>;
      latency_ms?: number;
    };
    if (!Array.isArray(data.results) || data.results.length === 0) {
      return { rows: candidates, mode: "rerank-fallback" };
    }
    const wallMs = performance.now() - t0;
    logger.info(
      `rerank[${backendLabel}]: n=${candidates.length} sidecar=${data.latency_ms ?? "?"}ms wall=${wallMs.toFixed(0)}ms`,
    );
    const reordered: ResultRow[] = [];
    for (const r of data.results) {
      const src = candidates[r.index];
      if (!src) continue;
      reordered.push({
        ...src,
        // Carry rerank score in `rank` for transparency; negate so the
        // existing "smaller rank == better" caller contract holds.
        rank: -r.score,
      });
    }
    return { rows: reordered, mode: "rerank" };
  } catch (e: any) {
    logger.warn(
      `rerank[${backendLabel}]: failed (${e?.message ?? e}); falling back to RRF order`,
    );
    return { rows: candidates, mode: "rerank-fallback" };
  }
}

/**
 * Decide which rerank backend (if any) to use for this call.
 *
 * Precedence:
 *   1. `deep_llm=true` → Claude Haiku sidecar (opt-in, slow but precise)
 *   2. `deep=true` (or QOOPIA_RERANK_DEFAULT_DEEP=1) → JinaAI cross-encoder
 *   3. Otherwise → null (no rerank, raw RRF order)
 *
 * Legacy QOOPIA_RERANK_ENABLED=1 keeps the historical "always-on with the
 * single env-configured endpoint" behaviour, even when the caller didn't
 * pass `deep`. This lets existing deployments upgrade without coordinated
 * call-site changes.
 */
function chooseRerankBackend(p: RecallParams): {
  endpoint: string;
  timeoutMs: number;
  label: "jina" | "llm" | "legacy";
} | null {
  if (p.deep_llm === true) {
    return {
      endpoint: llmRerankEndpoint(),
      timeoutMs: llmRerankTimeoutMs(),
      label: "llm",
    };
  }
  const deepRequested = p.deep ?? rerankDefaultDeep();
  if (deepRequested) {
    return {
      endpoint: jinaRerankEndpoint(),
      timeoutMs: rerankTimeoutMs(),
      label: "jina",
    };
  }
  if (rerankEnabledLegacy() && p.deep !== false) {
    return {
      endpoint: jinaRerankEndpoint(),
      timeoutMs: rerankTimeoutMs(),
      label: "legacy",
    };
  }
  return null;
}

export async function recall(p: RecallParams) {
  const t0 = performance.now();
  let recallBackendUsed: "fts" | "vector" | "hybrid" | "deep" | "deep_llm" = "fts";
  const limit = Math.min(Math.max(p.limit || 10, 1), 50);
  const sanitized = sanitizeFtsQuery(p.query);
  const scope = p.scope || "notes";
  const canCrossWorkspace = !!(p.cross_workspace && p.privileged);
  const requestedMode = p.mode || getRecallMode();
  // `effectiveMode` tracks what actually ran — it may degrade to
  // `fts5-fallback` if the vector channel was requested but produced
  // no candidates (Ollama down, no embeddings). Visible in the
  // response so callers can detect silent degradation.
  let effectiveMode: RecallMode | "fts5-fallback" = requestedMode;
  // Phase 1 item 6 (recall_log): track which rerank backend the call
  // chose and whether it actually returned (vs fell back to RRF order).
  // Resolved post-hoc just before the recall_log insert.
  let rerankBackendLabel: "jina" | "llm" | "legacy" | null = null;
  let rerankSucceeded = false;

  const results: ResultRow[] = [];
  let tokensReturned = 0;

  // Phase 2 Item C R2 — feature flag. Default OFF until Leo final PASS
  // + Asхат explicit GO (Phase 2 plan §"Item C — Rollback point").
  // When false, the entity channel short-circuits — recall behaves
  // exactly as it did pre-Item-C.
  const entityPagesEnabled = process.env.QOOPIA_ENTITY_PAGES === "true";

  // Phase 2 Item C R2 — cross-source global ranking. notes and
  // entities are collected into separate buffers and merged into a
  // single rank-sorted pool BEFORE slicing to `limit`. R1 appended
  // entities after notes which let abundant notes shut out higher-
  // scoring entity hits (Leo R1 BLOCK §2). All channels carry `rank`
  // with the "smaller = better" contract: bm25 is negative, RRF is
  // stored as `-rrf`, rerank score is stored as `-score`.
  const noteResults: ResultRow[] = [];
  const entityResults: ResultRow[] = [];

  if (scope === "notes" || scope === "all") {
    if (requestedMode === "hybrid") {
      // Pull more candidates per channel than the final limit so RRF
      // has room to promote items ranked low in one channel but high
      // in the other.
      const channelTopN = hybridChannelTopN();
      const [ftsRows, vecRows] = await Promise.all([
        Promise.resolve(ftsNoteCandidates(p, sanitized, channelTopN)),
        vectorNoteCandidates(p, channelTopN),
      ]);
      if (vecRows.length === 0 && ftsRows.length >= 0) {
        // Pure FTS — record the fallback for observability but still
        // use the FTS-only ordering.
        effectiveMode = "fts5-fallback";
        recallBackendUsed = "fts";
        for (const r of ftsRows.slice(0, limit)) {
          // Drop the internal rowid_rank field.
          const { rowid_rank: _rr, ...clean } = r;
          noteResults.push(clean);
        }
      } else {
        recallBackendUsed = "hybrid";
        const scores = rrfFuse(ftsRows, vecRows);
        const ranked = [...scores.entries()]
          .map(([id, rrf]) => ({ id, rrf }))
          .sort((a, b) => b.rrf - a.rrf);
        // When reranker is enabled, hydrate a deeper pool so the
        // backend has room to surface items RRF ranked below `limit`.
        // Otherwise keep the original behaviour (hydrate just what the
        // caller will see).
        const backend = chooseRerankBackend(p);
        const hydrateN = backend ? Math.max(limit, rerankTopK()) : limit;
        const hydrated = hydrateFusedNotes(p, ranked, hydrateN);
        let finalRows = hydrated;
        if (backend) {
          recallBackendUsed = backend.label === "llm" ? "deep_llm" : "deep";
          rerankBackendLabel = backend.label;
          const { rows, mode: rerankMode } = await rerankResults(
            p.query,
            hydrated,
            backend.endpoint,
            backend.timeoutMs,
            backend.label,
          );
          rerankSucceeded = rerankMode === "rerank";
          finalRows = rows.slice(0, limit);
        }
        for (const r of finalRows) noteResults.push(r);
      }
    } else {
      // Pure FTS5 path — original behaviour.
      const ftsRows = ftsNoteCandidates(p, sanitized, limit);
      for (const r of ftsRows) {
        const { rowid_rank: _rr, ...clean } = r;
        noteResults.push(clean);
      }
    }
  }

  if (
    entityPagesEnabled &&
    (scope === "entities" || scope === "notes" || scope === "all")
  ) {
    // Phase 2 Item C — entity channel. Pulls up to `limit` entity
    // candidates so global merge has full visibility (the R1 cap of
    // min(limit, 5) made entities second-class). Hybrid (FTS5 +
    // bge-m3) mirrors the notes pipeline with the same RRF fusion;
    // FTS-only fallback when the vector channel returns nothing.
    const entityRows = await entityCandidates(p, sanitized, limit);
    for (const r of entityRows) entityResults.push(r);
  }

  // Merge notes + entities into a single sorted pool and slice to
  // `limit`. Smaller rank wins (bm25 / -RRF / -rerank-score all share
  // this contract). Tie-breaker: notes before entities to preserve
  // backward-compatible ordering for tests that don't seed entities.
  if (scope === "notes" || scope === "all" || scope === "entities") {
    // Cross-source FTS-only merge: raw bm25 values from different FTS5
    // tables are not comparable. When both notes + entities are present
    // and we did not go through hybrid/RRF already, normalize each
    // channel to an ordinal RRF-shaped score before the global merge.
    // This preserves within-channel order while making the cross-source
    // comparison coherent.
    if (requestedMode !== "hybrid" && noteResults.length > 0 && entityResults.length > 0) {
      const k = rrfK();
      noteResults.forEach((row, idx) => {
        row.rank = -(1 / (k + idx + 1));
      });
      entityResults.forEach((row, idx) => {
        row.rank = -(1 / (k + idx + 1));
      });
    }

    const pool: ResultRow[] = [];
    if (scope !== "entities") pool.push(...noteResults);
    pool.push(...entityResults);
    pool.sort((a, b) => {
      if (a.rank !== b.rank) return a.rank - b.rank;
      // Stable-ish tie-break: notes ahead of entities when scores match.
      if (a.source === "notes" && b.source === "entity") return -1;
      if (a.source === "entity" && b.source === "notes") return 1;
      return 0;
    });
    for (const r of pool.slice(0, limit)) {
      results.push(r);
      tokensReturned += Math.ceil(r.text.length / 4);
    }
  }

  if (scope === "activity" || scope === "all") {
    // Migration 009 added activity_fts (mirrors notes_fts pattern).
    // No vector channel for activity — summaries are short, FTS suffices.
    const where: string[] = [`activity_fts MATCH ?`];
    const params: any[] = [sanitized];
    if (!canCrossWorkspace) {
      where.push(`a.workspace_id = ?`);
      params.push(p.workspace_id);
    }
    // QTHIRD-001: hide activity rows tied to sibling private notes.
    where.push(`(a.visibility = 'workspace' OR a.agent_id = ? OR ? = 1)`);
    params.push(p.caller_agent_id, p.is_admin ? 1 : 0);
    const sql = `
      SELECT a.id, 'activity' as type, a.summary as text, a.details as metadata,
             a.project_id, a.created_at, a.workspace_id, rank
      FROM activity_fts f
      JOIN activity a ON a.rowid = f.rowid
      WHERE ${where.join(" AND ")}
      ORDER BY rank
      LIMIT ?
    `;
    const rows = db.prepare(sql).all(...params, limit) as Array<{
      id: string;
      type: string;
      text: string;
      metadata: string;
      project_id: string | null;
      created_at: string;
      workspace_id: string;
      rank: number;
    }>;
    for (const r of rows) {
      results.push({
        id: r.id,
        type: r.type,
        text: r.text,
        metadata: safeJsonParse(r.metadata, {} as Record<string, unknown>),
        project_id: r.project_id,
        created_at: r.created_at,
        workspace_id: r.workspace_id,
        rank: r.rank,
        source: "activity",
      });
      tokensReturned += Math.ceil(r.text.length / 4);
    }
  }

  if (scope === "sessions" || scope === "all") {
    // session_messages_fts — same private-message-only policy.
    const where: string[] = [`session_messages_fts MATCH ?`];
    const params: any[] = [sanitized];
    if (!canCrossWorkspace) {
      where.push(`m.workspace_id = ?`);
      params.push(p.workspace_id);
    }
    where.push(`(m.agent_id = ? OR ? = 1)`);
    params.push(p.caller_agent_id, p.is_admin ? 1 : 0);
    const sql = `
      SELECT m.id, m.role as type, m.content as text, m.metadata, NULL as project_id,
             m.created_at, m.workspace_id, m.session_id, rank
      FROM session_messages_fts f
      JOIN session_messages m ON m.id = f.rowid
      WHERE ${where.join(" AND ")}
      ORDER BY rank
      LIMIT ?
    `;
    const rows = db.prepare(sql).all(...params, limit) as Array<{
      id: number;
      type: string;
      text: string;
      metadata: string;
      project_id: null;
      created_at: string;
      workspace_id: string;
      session_id: string;
      rank: number;
    }>;
    for (const r of rows) {
      const metadata = safeJsonParse(r.metadata, {} as Record<string, unknown>);
      (metadata as Record<string, unknown>).session_id = r.session_id;
      results.push({
        id: String(r.id),
        type: `session_message:${r.type}`,
        text: r.text,
        metadata,
        project_id: null,
        created_at: r.created_at,
        workspace_id: r.workspace_id,
        rank: r.rank,
        source: "sessions",
      });
      tokensReturned += Math.ceil(r.text.length / 4);
    }
  }

  // Rough cost metric: compare to a naive full scan estimate (avg note 200
  // chars × rows). Good enough to demonstrate savings to the agent.
  const totalRow = db
    .prepare(
      `SELECT COUNT(*) as c, COALESCE(SUM(length(text)),0) as total_chars FROM notes WHERE workspace_id = ? AND deleted_at IS NULL`,
    )
    .get(p.workspace_id) as { c: number; total_chars: number };
  const fullScanTokens = Math.ceil(totalRow.total_chars / 4);
  const savings =
    fullScanTokens > 0 ? 1 - tokensReturned / fullScanTokens : 0;

  // Phase 1 item 6 — recall_log telemetry. Insert AFTER computing the
  // response so latency reflects the full call, BEFORE returning to the
  // caller. Wrapped in try/catch: insertion failures (table missing on
  // an un-migrated DB, FK violation, etc.) MUST NOT fail the recall
  // itself. The redaction step is mandatory and runs synchronously
  // here — query_text NEVER goes into recall_log raw.
  try {
    const latencyMs = Math.max(1, Math.round(performance.now() - t0));
    let backendPath: "fts" | "vector" | "hybrid" | "deep" | "deep_llm";
    if (effectiveMode === "fts5" || effectiveMode === "fts5-fallback") {
      backendPath = "fts";
    } else if (rerankBackendLabel === "llm") {
      backendPath = "deep_llm";
    } else if (rerankBackendLabel === "jina" || rerankBackendLabel === "legacy") {
      backendPath = "deep";
    } else {
      backendPath = "hybrid";
    }
    const deepUsed = rerankSucceeded ? 1 : 0;
    db.prepare(
      `INSERT INTO recall_log
         (caller_agent, workspace_id, query_text, top_k, result_ids,
          result_scores, latency_ms, backend_path, scope, deep_used, error_class)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)`,
    ).run(
      p.caller_agent_id,
      p.workspace_id,
      redactQuery(p.query),
      limit,
      JSON.stringify(results.map((r) => r.id)),
      JSON.stringify(
        results.map((r) =>
          Number.isFinite(r.rank) ? Number(r.rank.toFixed(6)) : 0,
        ),
      ),
      latencyMs,
      backendPath,
      scope,
      deepUsed,
    );
  } catch (e: any) {
    // Swallow — telemetry must never break the recall path. The most
    // common cause on an un-migrated DB is "no such table: recall_log".
    logger.warn(`recall_log insert skipped: ${e?.message ?? String(e)}`);
  }

  return {
    results,
    total_found: results.length,
    query: p.query,
    sanitized_query: sanitized,
    mode: effectiveMode,
    cost: {
      tokens_returned: tokensReturned,
      tokens_full_scan_estimate: fullScanTokens,
      savings_ratio: Math.max(0, Math.min(1, Number(savings.toFixed(3)))),
    },
  };
}
