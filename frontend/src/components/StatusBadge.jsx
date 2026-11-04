export default function StatusBadge({ status = "Review" }) {
  const cls = status.toLowerCase().replace(/\s+/g, "-");

  return <span className={`status-badge ${cls}`}>{status}</span>;
}
