export default function LoadingSpinner({ label = "Loading..." }) {
  return (
    <div className="loading-state">
      <span className="spinner"></span>

      <span>{label}</span>
    </div>
  );
}
