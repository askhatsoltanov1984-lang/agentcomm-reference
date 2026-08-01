# SoT — MAINMEMORY → entity pages migration

**Status**: design + tooling (Phase 2 Item F). One-shot migration is
agent-side, gated on Asхат GO. Refresh hook deployment is a separate
GO from the tooling acceptance.

**Plan reference**: Phase 2 plan note `01KSC0E9F1WFWJMSP58KS34H2A`,
Item F. Built on Phase 2 Item C R3 (entity_pages MVP, commit
`7882c290`), which is built on Item B (`2ec5ff43`) and the Phase 1
bundle (`fa8b238`).

---

## Rationale (SoT rule)

Phase 0 architecture decision §Q10 (decision note
`01KSBJVWFKVPX9P4R2DB8R2TX1`) locks this:

> **Qoopia entity pages are the source of truth for durable facts**
> about persons, agents, machines, services, projects, protocols, and
> knowledge topics. The per-agent `workspace/memory_system/MAINMEMORY.md`
> file is an **ephemeral cache** rehydrated at session start.

Before this migration, every Ductor agent kept the same durable facts
in three places: its own `MAINMEMORY.md`, `~/.ductor/SHAREDMEMORY.md`,
and ad-hoc Qoopia notes. The three drift; recall against any one of
them returns stale or contradictory text. SoT consolidation makes
entity pages the single canonical source — agents read them via
`recall()` or `entity_page_render()` and never have to reconcile.

`MAINMEMORY.md` is not deleted. It survives as a thin **session-scope
cache**: it carries (a) a top-line SoT marker pointing at the entity
slugs the agent wants rehydrated, and (b) the ephemeral sections
(in-progress task state, recent incident logs, rolling phase scores)
that are deliberately short-lived and never belong in entity_pages.

SHARED KNOWLEDGE blocks (delimited by `--- SHARED KNOWLEDGE START ---`
/ `--- SHARED KNOWLEDGE END ---`) are owned by
`tools/agent_tools/edit_shared_knowledge.py` and the Supervisor sync.
The migration tool is hard-coded to skip them.

---

## Tooling overview

Two scripts ship in this PR. Both are pure Bun TypeScript and run in
the qoopia-corsair container; neither touches the production agent
process or its system prompt.

| File | Role |
| --- | --- |
| `scripts/mainmemory_migrate.ts` | One-shot migration runner — parses a `MAINMEMORY.md`, classifies sections, upserts durable ones to `entity_pages`, writes back a stub MAINMEMORY. Dry-run by default; `--apply` required to mutate. |
| `scripts/mainmemory_refresh.ts` | Session-start helper — reads a slug list, renders each entity, emits markdown to stdout for prepending to the agent system prompt. |
| `tests/mainmemory_migration.test.ts` | Parser + classifier + apply + refresh tests; 27 cases, all on synthetic `/tmp` MAINMEMORYs. |

The migration tool tags every entity it creates with:

```json
{
  "migrated_from": "MAINMEMORY.md",
  "migrated_at": "<ISO ms timestamp>",
  "original_section": "<original H2 header>",
  "source_path": "<absolute source path>",
  "original_start_line": <int>,
  "original_end_line": <int>
}
```

so the entity row alone tells you where it came from. Reverse audit
is `entity_page_render(slug) → metadata.migrated_from`.

---

## Migration procedure (agent-side; Asхат GO required to `--apply`)

1. **Dry-run** (zero side effects). Run inside the qoopia-corsair
   container with the source path bind-mounted into the container.

   ```
   bun run /app/scripts/mainmemory_migrate.ts \
     --source=/mnt/agent/MAINMEMORY.md \
     --workspace-slug=default
   ```

   The script prints a JSON report with classification counts, durable
   slugs/types, ephemeral headers carried forward, and stale headers
   to be dropped. No DB writes, no filesystem writes.

2. **Review** the JSON output. Spot-check every durable slug for
   correctness; entities created here are workspace-unique on
   `(workspace_id, slug)`, so a typo creates a permanent
   misnamed row. Verify the ephemeral header list is what you expect
   to survive in the stub.

3. **Pre-mutation snapshot** (mandatory). Copy the source MAINMEMORY
   to a `.bak.<UTC>` sibling before `--apply`:

   ```
   cp /path/to/MAINMEMORY.md /path/to/MAINMEMORY.md.bak.$(date -u +%Y%m%dT%H%M%SZ)
   ```

   Also snapshot the target DB (or rely on the daily
   `pull-latest-snapshot.sh` cron if its retention covers the migration
   window).

4. **Apply** (writes DB + filesystem). Re-run with `--apply`:

   ```
   bun run /app/scripts/mainmemory_migrate.ts \
     --source=/mnt/agent/MAINMEMORY.md \
     --workspace-slug=default \
     --apply
   ```

   The script:
   - upserts every durable section into `entity_pages` (idempotent on
     `(workspace_id, slug)`)
   - writes a stub `MAINMEMORY.md` at the source path (or `--out=`
     target) containing the SoT marker + ephemeral sections verbatim.

5. **Verify**. `entity_page_render(slug)` on every migrated slug;
   `recall()` against the durable facts; manually read the stub file
   and confirm only ephemeral sections survived.

### Hard rules

- **NEVER** run `--apply` against
  `/home/askhat/.ductor-corsairmain/workspace/memory_system/MAINMEMORY.md`
  from a code review iteration. Item F PR does design + tooling +
  tests only — the real migration is a separate gated operation.
- The script never reads or writes the production agent's system
  prompt. Refresh hook wiring into the system prompt is itself a
  separate Asхат GO (see "Refresh hook integration" below).
- Workspace must exist before `--apply` — `resolveWorkspaceId()` aborts
  cleanly if the slug is unknown.

---

## Refresh hook integration

Once an agent's MAINMEMORY is a stub, its system prompt must
re-establish the durable facts at session start. The mechanism is:

1. Maintain a slug list at
   `~/.ductor-corsairmain/workspace/memory_system/mainmemory_refresh_slugs.txt`
   (one slug per line; `#`-prefixed lines ignored). The default Item F
   list mirrors the durable slugs from the audit (see Phase 2 Item F
   preflight note for the canonical inventory).

2. Bind-mount the slugs file into the qoopia-corsair container at
   `/etc/qoopia/mainmemory_refresh_slugs.txt` (or pass an explicit
   `--slugs-file=...` arg).

3. Run the refresh helper at session start:

   ```
   bun run /app/scripts/mainmemory_refresh.ts \
     --slugs-file=/etc/qoopia/mainmemory_refresh_slugs.txt \
     --workspace-slug=default
   ```

   stdout contains the concatenated entity_page_render markdown for
   every slug. The agent's bootstrap script prepends this output to its
   system prompt (in front of, or replacing, the historic
   `MAINMEMORY.md` body block).

4. Cap: `--max-bytes=<N>` (default 64 KiB) limits total output. Above
   the cap the script truncates with a marker; the agent sees the
   truncation in-band.

Missing slugs are recorded as in-band `<!-- mainmemory_refresh: missing
slug='...' -->` markers, not failures. This lets the agent detect
schema drift without losing the rest of the refresh.

### What is NOT in this PR

- The actual `corsairmain-ductor.service` wiring that calls the
  refresh helper at startup. That's a deploy step under Asхат GO; the
  service unit change is small but it modifies the production agent
  prompt path.
- The host-side slugs file. The path is documented; the file is
  created at deploy time alongside the service unit change.
- Any modification of the live `MAINMEMORY.md`. The tooling supports
  it; this PR does not perform it.

---

## Rollback

Both halves of the migration are reversible:

| Artifact | Rollback |
| --- | --- |
| `MAINMEMORY.md` (stubbed) | `cp MAINMEMORY.md.bak.<UTC> MAINMEMORY.md` |
| Migrated entity rows | `UPDATE entity_pages SET status='archived' WHERE json_extract(metadata, '$.migrated_from') = 'MAINMEMORY.md' AND json_extract(metadata, '$.source_path') = '<path>'` (keeps audit trail; entities are scoped per workspace, so this never touches Item C R3 seeded entities in a different workspace). |

The migration is **not destructive**: stale sections are dropped from
the stub but were never copied into entity_pages, so they exist only in
the `.bak` snapshot. Recover them from there if classification was wrong.

The entity-side rollback uses `status='archived'` rather than `DELETE`
to preserve the audit trail. If a slug needs to be re-used after
rollback, archive then upsert with the same slug — the next migration
run will reactivate it (the service does not filter on status when
upserting).

---

## Operational risks

1. **Slug collisions**. Two MAINMEMORY sections with H2 headers that
   kebab-case to the same slug will collide: the second upsert
   overwrites the first. The dry-run report makes this visible; resolve
   by editing one of the headers in the source before `--apply`.

2. **Classifier misfires**. The keyword heuristic is conservative
   (unknown headers default to ephemeral, not durable) but it can
   wrongly bucket a durable section as ephemeral. Dry-run review
   catches this; classification can be overridden by editing the source
   header to include a known durable keyword.

3. **Per-agent scope creep**. This PR's tooling is generic but the
   Item F brief scopes the actual `--apply` to corsair-main only.
   Other agents' MAINMEMORYs are Phase 3+ (see Phase 2 plan note
   "out-of-scope for Phase 2").

4. **Refresh size bloat**. As the slug list grows, refresh output
   grows linearly. The `--max-bytes` cap is a safety net; the agent
   bootstrap should reject a stub whose refresh consistently truncates.

---

## Test coverage

`tests/mainmemory_migration.test.ts` (27 cases / all on synthetic
`/tmp` MAINMEMORYs, never the real one):

- Parser: H2 boundary split, pre-H2 intro detection, SHARED KNOWLEDGE
  block exclusion, empty + intro-only edge cases.
- Classifier: durable/ephemeral/stale bucketing for the canonical
  Corsair-main section types, stub-marker recognition for idempotency,
  default-ephemeral for unknown headers.
- Slug derivation: null header, date-qualifier stripping, em-dash
  stripping, kebab-case correctness, entity-shape validation.
- Planner: bucket counts + slug/type assignment for durables.
- Apply: dry-run report shape; `--apply` produces correct upserts +
  stub file shape (SoT marker on line 1, ephemeral sections preserved,
  stale dropped, durable headers removed); idempotency on re-runs
  (stub re-run = 0 durables; original re-run = all `created=false`).
- Refresh: slugs file parsing (comments / blank lines), missing-slug
  marker behaviour, multi-entity render output, oversize truncation.
