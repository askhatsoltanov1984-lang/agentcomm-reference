/**
 * Reject text that appears to contain secrets before persisting it into
 * notes, session messages, summaries, or activity logs.
 */
import { QoopiaError } from "./errors.ts";

interface SecretDetector {
  label: string;
  re: RegExp;
}

const BASE_SECRET_DETECTORS: SecretDetector[] = [
  {
    label: "qoopia-token",
    re: /(?:q_|qa_|qr_|qc_|qcs_)[A-Za-z0-9_\-]{16,}/,
  },
  {
    label: "github-pat",
    re: /gh[pousr]_[A-Za-z0-9]{36,}/,
  },
  {
    label: "openai-key",
    re: /sk-[A-Za-z0-9]{20,}/,
  },
  {
    label: "openai-project-key",
    re: /sk-proj-[A-Za-z0-9]{40,}/,
  },
  {
    label: "aws-access-key-id",
    re: /AKIA[0-9A-Z]{16}/,
  },
  {
    label: "aws-secret-access-key",
    re: /aws_secret_access_key[^A-Za-z0-9]{0,20}[A-Za-z0-9/+=]{40}/i,
  },
  {
    label: "jwt",
    re: /eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+/,
  },
  {
    label: "anthropic-key",
    re: /sk-ant-[A-Za-z0-9_-]{40,}/,
  },
  {
    label: "slack-token",
    re: /xox[baprs]-[0-9A-Za-z-]{20,}/,
  },
  {
    label: "stripe-secret-key",
    re: /sk_live_[A-Za-z0-9]{24,}/,
  },
  {
    label: "stripe-public-key",
    re: /pk_live_[A-Za-z0-9]{24,}/,
  },
  {
    label: "ssh-private-key",
    re: /-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----/,
  },
];

function warnSecretPattern(message: string): void {
  console.warn(`[secret-guard] ${message}`);
}

function compileSecretDetector(
  label: string,
  pattern: string,
): SecretDetector | null {
  try {
    return {
      label,
      re: new RegExp(pattern),
    };
  } catch (err) {
    warnSecretPattern(
      `Skipping invalid QOOPIA_SECRET_PATTERNS_EXTRA detector '${label}': ${err}`,
    );
    return null;
  }
}

/**
 * Custom secret detectors from env.
 *
 * Preferred format:
 * `QOOPIA_SECRET_PATTERNS_EXTRA='[{"label":"foo","pattern":"foo[a,b]bar"}]'`
 *
 * Migration path:
 * - if the value starts with `[` -> parse JSON array of `{label, pattern}`
 * - otherwise -> treat as legacy comma-separated pattern list, emit a
 *   deprecation warning, and auto-label entries as `custom-secret-N`
 */
function extraSecretDetectors(): SecretDetector[] {
  const raw = process.env.QOOPIA_SECRET_PATTERNS_EXTRA || "";
  if (!raw.trim()) return [];

  if (raw.trimStart().startsWith("[")) {
    try {
      const parsed = JSON.parse(raw) as Array<{ label?: unknown; pattern?: unknown }>;
      if (!Array.isArray(parsed)) {
        warnSecretPattern(
          "QOOPIA_SECRET_PATTERNS_EXTRA JSON payload is not an array; skipping custom detectors",
        );
        return [];
      }
      return parsed
        .map((entry, idx) => {
          const label = String(entry?.label || `custom-secret-${idx + 1}`);
          const pattern = typeof entry?.pattern === "string" ? entry.pattern : "";
          if (!pattern) {
            warnSecretPattern(
              `Skipping QOOPIA_SECRET_PATTERNS_EXTRA detector '${label}': missing pattern`,
            );
            return null;
          }
          return compileSecretDetector(label, pattern);
        })
        .filter((detector): detector is SecretDetector => detector !== null);
    } catch (err) {
      warnSecretPattern(
        `Invalid JSON in QOOPIA_SECRET_PATTERNS_EXTRA; skipping custom detectors: ${err}`,
      );
      return [];
    }
  }

  warnSecretPattern(
    "QOOPIA_SECRET_PATTERNS_EXTRA legacy CSV format is deprecated; switch to JSON array syntax",
  );
  return raw
    .split(",")
    .map((pattern) => pattern.trim())
    .filter(Boolean)
    .map((pattern, idx) =>
      compileSecretDetector(`custom-secret-${idx + 1}`, pattern)
    )
    .filter((detector): detector is SecretDetector => detector !== null);
}

function allSecretDetectors(): SecretDetector[] {
  return [...BASE_SECRET_DETECTORS, ...extraSecretDetectors()];
}

export function detectSecretLabels(text: string): string[] {
  if (!text) return [];
  const labels: string[] = [];
  for (const detector of allSecretDetectors()) {
    if (detector.re.test(text)) {
      labels.push(detector.label);
    }
  }
  return labels;
}

export function assertNoSecrets(text: string, context: string): void {
  const labels = detectSecretLabels(text);
  if (labels.length > 0) {
    throw new QoopiaError(
      "INVALID_INPUT",
      `Secret pattern (${labels[0]}) detected in ${context} — refused. Never store plaintext keys/tokens in Qoopia.`,
    );
  }
}
