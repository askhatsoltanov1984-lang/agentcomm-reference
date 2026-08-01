#!/usr/bin/env bun
/**
 * Shadow sync CLI — invoked by /srv/qoopia/scripts/shadow_sync.sh.
 *
 * Phase 2 Item B (plan note 01KSC0E9F1WFWJMSP58KS34H2A).
 *
 * Dry-run (default) is STRICTLY read-only against both DBs. The only
 * side-effect is writing the markdown report to /logs/.
 *
 * --apply mode is GATED. Requires ALL of:
 *   - env SHADOW_SYNC_APPLY_AUTHORIZED set non-empty
 *   - --apply CLI flag
 *   - --leo-review-id <id>
 *   - --askhat-go-marker <text>
 * Missing any of them → engine refuses with HOLD diagnostic.
 *
 * Phase 2 ships dry-run; --apply path is wired through the same engine but
 * not exercised against prod data until Leo OK + Askhat GO. The first
 * real run is a dry-run against a Mac-mini snapshot.
 *
 * Usage:
 *   bun /app/scripts/shadow_sync.ts \
 *       --mac-db /backups/mac-mini-snapshot-<ts>.db \
 *       --cor-db /data/qoopia.db \
 *       [--report-out /logs/shadow-sync-dryrun-<ts>.md] \
 *       [--apply --leo-review-id <id> --askhat-go-marker <text>]
 */
import {
  buildPlan,
  renderReport,
  writeReport,
  reportBodyHash,
  checkApplyGate,
  applyPlan,
  renderApplyReport,
} from "../src/services/shadow_sync.ts";

type Args = {
  macDb: string;
  corDb: string;
  reportOut?: string;
  apply: boolean;
  leoReviewId?: string;
  askhatGoMarker?: string;
};

function parseArgs(argv: string[]): Args {
  const out: Args = {
    macDb: "",
    corDb: "/data/qoopia.db",
    apply: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    switch (a) {
      case "--mac-db":
        out.macDb = argv[++i];
        break;
      case "--cor-db":
        out.corDb = argv[++i];
        break;
      case "--report-out":
        out.reportOut = argv[++i];
        break;
      case "--apply":
        out.apply = true;
        break;
      case "--dry-run":
        out.apply = false;
        break;
      case "--leo-review-id":
        out.leoReviewId = argv[++i];
        break;
      case "--askhat-go-marker":
        out.askhatGoMarker = argv[++i];
        break;
      case "-h":
      case "--help":
        printHelp();
        process.exit(0);
      default:
        console.error(`unknown arg: ${a}`);
        printHelp();
        process.exit(2);
    }
  }
  return out;
}

function printHelp() {
  console.error(
    [
      "usage: shadow_sync.ts --mac-db <path> [--cor-db <path>] [--report-out <path>]",
      "                     [--apply --leo-review-id <id> --askhat-go-marker <text>]",
      "",
      "  --mac-db          path to Mac mini DB snapshot (read-only)",
      "  --cor-db          path to Corsair DB (default: /data/qoopia.db, read-only in dry-run)",
      "  --report-out      where to write the markdown report",
      "                    (default: /logs/shadow-sync-dryrun-<utc>.md)",
      "  --apply           BLOCKED unless ALL of: env SHADOW_SYNC_APPLY_AUTHORIZED,",
      "                    --leo-review-id, --askhat-go-marker.",
      "  --dry-run         default; strictly read-only against both DBs.",
    ].join("\n"),
  );
}

function utcTsLabel(): string {
  const d = new Date();
  const iso = d.toISOString().replace(/[:.]/g, "").replace(/-/g, "");
  return iso.slice(0, 15) + "Z"; // YYYYMMDDTHHMMSSZ
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.macDb) {
    console.error("error: --mac-db is required");
    printHelp();
    process.exit(2);
  }

  // --apply gate: enforce belt-and-braces (Q-P2-7, Leo R1 OK).
  const gateErr = checkApplyGate({
    apply: args.apply,
    envToken: process.env.SHADOW_SYNC_APPLY_AUTHORIZED,
    leoReviewId: args.leoReviewId,
    askhatGoMarker: args.askhatGoMarker,
  });
  if (gateErr) {
    console.error(gateErr);
    process.exit(3);
  }
  const tsLabel = utcTsLabel();

  if (args.apply) {
    // Phase 3 Item C — real M2C apply path. Gated above (env + leo + askhat).
    const plan = buildPlan(args.macDb, args.corDb);

    // Conflict refusal: conflict_queue must be 0 for the targeted direction.
    // Only `notes` can conflict (activity is an append-only union).
    const conflicts = plan.notes.filter((c) => c.direction === "conflict").length;
    if (conflicts > 0) {
      console.error(
        `HOLD: --apply refused; ${conflicts} conflict(s) in notes — resolve conflict_queue to 0 first`,
      );
      process.exit(5);
    }

    const applyReportOut =
      args.reportOut ?? `/logs/shadow-sync-apply-${tsLabel}.md`;
    const result = applyPlan(plan, {
      macDbPath: args.macDb,
      corDbPath: args.corDb,
      direction: "M2C",
    });

    // Post-apply verify: re-run the planner and assert the applied direction is
    // now empty (parity). Mac source is read-only throughout.
    const verifyPlan = buildPlan(args.macDb, args.corDb);
    const residualNotesM2c = verifyPlan.notes.filter((c) => c.direction === "M2C").length;
    const residualActivityM2c = verifyPlan.activity.filter((c) => c.direction === "M2C").length;
    const parityOk = residualNotesM2c === 0 && residualActivityM2c === 0;

    const applyReport = renderApplyReport(result, tsLabel, {
      residualNotesM2c,
      residualActivityM2c,
    });
    writeReport(applyReport, applyReportOut);

    console.log(
      `apply ${parityOk ? "OK" : "PARITY_FAIL"} report=${applyReportOut} ` +
        `direction=M2C tx=${result.transaction} ` +
        `notes_inserted=${result.notes.inserted} notes_updated=${result.notes.updated} ` +
        `notes_hashes=${result.notes.hashesRecorded} notes_skipped=${result.notes.skippedIdempotent} ` +
        `act_inserted=${result.activity.inserted} act_hashes=${result.activity.hashesRecorded} ` +
        `act_skipped=${result.activity.skippedIdempotent} ` +
        `post_apply_m2c_notes=${residualNotesM2c} post_apply_m2c_activity=${residualActivityM2c}`,
    );
    if (!parityOk) process.exit(6);
    return;
  }

  const reportOut = args.reportOut ?? `/logs/shadow-sync-dryrun-${tsLabel}.md`;

  const plan = buildPlan(args.macDb, args.corDb);
  const report = renderReport(plan, tsLabel);
  writeReport(report, reportOut);

  // Stdout summary for the operator + cron logs (no body bytes per rubric §7).
  const m2cN = plan.notes.filter((c) => c.direction === "M2C").length;
  const c2mN = plan.notes.filter((c) => c.direction === "C2M").length;
  const conflicts = plan.notes.filter((c) => c.direction === "conflict").length;
  const m2cA = plan.activity.filter((c) => c.direction === "M2C").length;
  const c2mA = plan.activity.filter((c) => c.direction === "C2M").length;
  const noopN = plan.notes.filter((c) => c.direction === "noop").length;
  const bodyHash = reportBodyHash(report);
  console.log(
    `dry-run OK report=${reportOut} notes_m2c=${m2cN} notes_c2m=${c2mN} notes_conflict=${conflicts} notes_noop=${noopN} act_m2c=${m2cA} act_c2m=${c2mA} report_body_sha256=${bodyHash}`,
  );
}

if (import.meta.main) main();
