export type QoopiaErrorCode =
  | "INVALID_INPUT"
  | "NOT_FOUND"
  | "FORBIDDEN"
  | "CONFLICT"
  | "SIZE_LIMIT"
  | "UNAUTHORIZED"
  | "INTERNAL"
  /**
   * Raised by src/services/embeddings.ts when the local Ollama call
   * fails (network, timeout, model not loaded, wrong dim). Callers
   * upstream (notes.fireAndForgetEmbed, recall.vectorNoteCandidates)
   * catch and degrade — the note still saves, recall returns FTS5
   * results only and marks mode='fts5-fallback'.
   */
  | "EMBEDDING_FAILED";

export class QoopiaError extends Error {
  constructor(public code: QoopiaErrorCode, message: string) {
    super(message);
    this.name = "QoopiaError";
  }

  toString() {
    return `${this.code}: ${this.message}`;
  }
}

export function nowIso(): string {
  return new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
}

export function safeJsonParse<T>(s: string | null | undefined, fallback: T): T {
  if (!s) return fallback;
  try {
    return JSON.parse(s) as T;
  } catch {
    return fallback;
  }
}
