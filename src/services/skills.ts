/**
 * Fat skills layer — typed convention on top of entity_pages
 * type='skill' (Phase 2 Item E; migration 023).
 *
 * Plan note: 01KSC0E9F1WFWJMSP58KS34H2A §"Item E — Fat skills layer".
 * Reuses entity_pages per Q-P2-5: no new table. The skills helpers
 * wrap upsertEntity / getEntity / searchEntities and add metadata
 * schema validation plus a markdown runbook renderer.
 *
 * Required metadata fields are enforced at write time by
 * skill_upsert. The shape mirrors migration 023's documentation
 * block — if you change one, change the other.
 */
import {
  upsertEntity,
  getEntity,
  searchEntities,
  type Entity,
  type EntityStatus,
  type SearchHit,
  type UpsertResult,
} from "./entities.ts";
import { QoopiaError } from "../utils/errors.ts";
import { assertNoSecrets } from "../utils/secret-guard.ts";

/**
 * Fields the skill_upsert caller MUST provide in metadata. Adding a
 * field here is non-breaking for read paths (skill_get still works
 * on older rows) but write paths will reject inputs missing the new
 * field — bump migration 023 if you add one.
 */
export const SKILL_METADATA_REQUIRED = [
  "skill_version",
  "owner_agent",
  "trigger_conditions",
  "scope",
  "prerequisites",
  "exact_steps",
  "verification_gates",
  "failure_modes",
  "rollback",
  "related_code_paths",
  "related_incidents",
] as const;

const SKILL_METADATA_OPTIONAL = ["last_tested", "tester_agent"] as const;

/** Fields whose value must be a non-empty string array. */
const STRING_ARRAY_FIELDS = new Set<string>([
  "trigger_conditions",
  "prerequisites",
  "exact_steps",
  "verification_gates",
  "failure_modes",
  "related_code_paths",
  "related_incidents",
]);

/** Fields whose value must be a non-empty string. */
const STRING_FIELDS = new Set<string>([
  "skill_version",
  "owner_agent",
  "scope",
  "rollback",
]);

const MAX_STEP_LEN = 4_000;
const MAX_STEP_COUNT = 100;

export interface SkillMetadata {
  skill_version: string;
  owner_agent: string;
  trigger_conditions: string[];
  scope: string;
  prerequisites: string[];
  exact_steps: string[];
  verification_gates: string[];
  failure_modes: string[];
  rollback: string;
  related_code_paths: string[];
  related_incidents: string[];
  last_tested?: string;
  tester_agent?: string;
  [key: string]: unknown;
}

export interface SkillUpsertInput {
  workspace_id: string;
  slug: string;
  title: string;
  summary: string;
  status?: EntityStatus;
  metadata: SkillMetadata;
}

/**
 * Validate skill metadata. Throws QoopiaError on first violation —
 * callers do not get partial saves. Validation is intentionally
 * strict: missing required field is a write-time failure, not a
 * silent default, because a half-encoded runbook is worse than no
 * runbook (it lies about being verified).
 */
export function validateSkillMetadata(metadata: unknown): void {
  if (!metadata || typeof metadata !== "object" || Array.isArray(metadata)) {
    throw new QoopiaError(
      "INVALID_INPUT",
      "skill metadata must be an object",
    );
  }
  const m = metadata as Record<string, unknown>;

  for (const field of SKILL_METADATA_REQUIRED) {
    if (!(field in m)) {
      throw new QoopiaError(
        "INVALID_INPUT",
        `skill metadata missing required field '${field}' (see SKILL_METADATA_REQUIRED)`,
      );
    }
  }

  for (const field of STRING_FIELDS) {
    const v = m[field];
    if (typeof v !== "string" || v.length === 0) {
      throw new QoopiaError(
        "INVALID_INPUT",
        `skill metadata.${field} must be a non-empty string`,
      );
    }
  }

  for (const field of STRING_ARRAY_FIELDS) {
    const v = m[field];
    if (!Array.isArray(v)) {
      throw new QoopiaError(
        "INVALID_INPUT",
        `skill metadata.${field} must be an array of strings`,
      );
    }
    if (v.length === 0) {
      throw new QoopiaError(
        "INVALID_INPUT",
        `skill metadata.${field} must contain at least one entry`,
      );
    }
    if (v.length > MAX_STEP_COUNT) {
      throw new QoopiaError(
        "SIZE_LIMIT",
        `skill metadata.${field} exceeds ${MAX_STEP_COUNT} entries — split the skill`,
      );
    }
    for (let i = 0; i < v.length; i++) {
      const entry = v[i];
      if (typeof entry !== "string" || entry.length === 0) {
        throw new QoopiaError(
          "INVALID_INPUT",
          `skill metadata.${field}[${i}] must be a non-empty string`,
        );
      }
      if (entry.length > MAX_STEP_LEN) {
        throw new QoopiaError(
          "SIZE_LIMIT",
          `skill metadata.${field}[${i}] exceeds ${MAX_STEP_LEN} chars`,
        );
      }
    }
  }

  if (m.last_tested !== undefined) {
    if (typeof m.last_tested !== "string" || m.last_tested.length === 0) {
      throw new QoopiaError(
        "INVALID_INPUT",
        "skill metadata.last_tested must be an ISO timestamp string",
      );
    }
    const ts = Date.parse(m.last_tested);
    if (Number.isNaN(ts)) {
      throw new QoopiaError(
        "INVALID_INPUT",
        "skill metadata.last_tested must parse as an ISO timestamp",
      );
    }
  }
  if (m.tester_agent !== undefined) {
    if (typeof m.tester_agent !== "string" || m.tester_agent.length === 0) {
      throw new QoopiaError(
        "INVALID_INPUT",
        "skill metadata.tester_agent must be a non-empty string when set",
      );
    }
  }
}

/**
 * Upsert a skill entity. Wraps entity_upsert with type='skill' and
 * runs validateSkillMetadata first so the row is never written with
 * a malformed runbook.
 */
export function skillUpsert(input: SkillUpsertInput): UpsertResult {
  if (!input.summary || input.summary.length === 0) {
    throw new QoopiaError(
      "INVALID_INPUT",
      "skill summary is required (renders as the runbook intro)",
    );
  }
  validateSkillMetadata(input.metadata);
  // Defensive: the entity layer also runs assertNoSecrets, but skills
  // carry exact_steps and verification_gates that are the most likely
  // place a secret would slip in. Check the joined steps explicitly so
  // the error pins the offending field rather than 'entity.metadata'.
  const stepsBlob = [
    ...(input.metadata.exact_steps ?? []),
    ...(input.metadata.verification_gates ?? []),
    ...(input.metadata.failure_modes ?? []),
    input.metadata.rollback ?? "",
  ].join("\n");
  assertNoSecrets(stepsBlob, "skill.steps");

  return upsertEntity({
    workspace_id: input.workspace_id,
    type: "skill",
    slug: input.slug,
    title: input.title,
    summary: input.summary,
    status: input.status,
    metadata: input.metadata,
  });
}

export interface SkillGetParams {
  workspace_id: string;
  id?: string;
  slug?: string;
}

export interface Skill extends Entity {
  metadata: SkillMetadata;
}

/**
 * Fetch a skill entity. Throws NOT_FOUND if the entity exists but is
 * not type='skill' — the caller asked for a skill, not a person who
 * happens to share a slug.
 */
export function skillGet(p: SkillGetParams): Skill {
  const ent = getEntity({
    workspace_id: p.workspace_id,
    id: p.id,
    slug: p.slug,
  });
  if (ent.type !== "skill") {
    throw new QoopiaError(
      "NOT_FOUND",
      `entity ${p.id ?? p.slug} exists but type='${ent.type}', not 'skill'`,
    );
  }
  return { ...ent, metadata: ent.metadata as SkillMetadata };
}

export interface SkillSearchParams {
  workspace_id: string;
  query?: string;
  status?: EntityStatus;
  limit?: number;
}

/**
 * Search skill entities. Wraps entity_search with type='skill' pinned
 * so callers cannot accidentally widen the search across types.
 */
export function skillSearch(p: SkillSearchParams): SearchHit[] {
  return searchEntities({
    workspace_id: p.workspace_id,
    type: "skill",
    query: p.query,
    status: p.status,
    limit: p.limit,
  });
}

export interface SkillRunbook {
  id: string;
  slug: string;
  title: string;
  markdown: string;
  last_tested: string | null;
}

/**
 * Render a skill as an actionable markdown runbook. Sections:
 * Trigger, Prerequisites, Steps, Verification, Failure modes,
 * Rollback, Code paths, Related incidents, Metadata. Operators
 * paste the rendered runbook into a session log when they execute
 * the skill — so the rendered form must be self-contained.
 */
export function skillRenderRunbook(p: SkillGetParams): SkillRunbook {
  const s = skillGet(p);
  const m = s.metadata;

  const lines: string[] = [];
  lines.push(`# ${s.title}`);
  lines.push("");
  lines.push(
    `*skill_version:* \`${m.skill_version}\` &middot; *owner:* \`${m.owner_agent}\` &middot; *status:* \`${s.status}\``,
  );
  if (m.last_tested) {
    lines.push(
      `*last_tested:* ${m.last_tested}${m.tester_agent ? ` by \`${m.tester_agent}\`` : ""}`,
    );
  } else {
    lines.push(`*last_tested:* never`);
  }
  lines.push(`*scope:* ${m.scope}`);
  lines.push("");
  if (s.summary) {
    lines.push(s.summary);
    lines.push("");
  }

  lines.push("## Trigger");
  for (const t of m.trigger_conditions) lines.push(`- ${t}`);
  lines.push("");

  lines.push("## Prerequisites");
  for (const p of m.prerequisites) lines.push(`- ${p}`);
  lines.push("");

  lines.push("## Steps");
  m.exact_steps.forEach((step, i) => {
    lines.push(`${i + 1}. ${step}`);
  });
  lines.push("");

  lines.push("## Verification");
  for (const g of m.verification_gates) lines.push(`- [ ] ${g}`);
  lines.push("");

  lines.push("## Failure modes");
  for (const f of m.failure_modes) lines.push(`- ${f}`);
  lines.push("");

  lines.push("## Rollback");
  lines.push(m.rollback);
  lines.push("");

  if (m.related_code_paths.length > 0) {
    lines.push("## Code paths");
    for (const cp of m.related_code_paths) lines.push(`- \`${cp}\``);
    lines.push("");
  }

  if (m.related_incidents.length > 0) {
    lines.push("## Related incidents");
    for (const inc of m.related_incidents) lines.push(`- ${inc}`);
    lines.push("");
  }

  const markdown = lines.join("\n");
  assertNoSecrets(markdown, "skill.runbook");

  return {
    id: s.id,
    slug: s.slug,
    title: s.title,
    markdown,
    last_tested: m.last_tested ?? null,
  };
}

export interface SkillMarkTestedInput {
  workspace_id: string;
  id?: string;
  slug?: string;
  tested_at: string;
  tester_agent: string;
}

/**
 * Mark a skill as verified at a point in time. Updates
 * metadata.last_tested and metadata.tester_agent via skillUpsert so
 * the validation path still runs on the merged metadata — the rest
 * of the runbook can't drift into an invalid state during a mark.
 */
export function skillMarkTested(input: SkillMarkTestedInput): UpsertResult {
  const ts = Date.parse(input.tested_at);
  if (Number.isNaN(ts)) {
    throw new QoopiaError(
      "INVALID_INPUT",
      "tested_at must parse as an ISO timestamp",
    );
  }
  if (!input.tester_agent || input.tester_agent.length === 0) {
    throw new QoopiaError(
      "INVALID_INPUT",
      "tester_agent is required",
    );
  }
  const existing = skillGet({
    workspace_id: input.workspace_id,
    id: input.id,
    slug: input.slug,
  });
  const merged: SkillMetadata = {
    ...existing.metadata,
    last_tested: input.tested_at,
    tester_agent: input.tester_agent,
  };
  return skillUpsert({
    workspace_id: input.workspace_id,
    slug: existing.slug,
    title: existing.title,
    summary: existing.summary ?? "",
    status: existing.status,
    metadata: merged,
  });
}
