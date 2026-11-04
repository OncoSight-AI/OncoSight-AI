export function normalizeStudy(item = {}) {
  return {
    id: String(item.id ?? item.study_id ?? "unknown"),

    name: item.name ?? item.filename ?? "Untitled study",

    modality: item.modality ?? "Unknown",

    model: item.model ?? "Not specified",

    status: item.status ?? "Review",

    date: item.date ?? item.created_at ?? "—",

    confidence: Number.isFinite(Number(item.confidence))
      ? Number(item.confidence)
      : null,

    finding: item.finding ?? item.summary ?? "No summary available",

    notes: item.notes ?? "",
  };
}
