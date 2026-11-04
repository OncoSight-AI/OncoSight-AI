export default function MetricCard({ label, value, description }) {
  return (
    <div className="metric-card">
      <span>{label}</span>

      <strong>{value}</strong>

      <small>{description}</small>
    </div>
  );
}
