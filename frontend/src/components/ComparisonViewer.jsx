export default function ComparisonViewer({ before, after }) {
  return (
    <div className="comparison-grid">
      <div>
        <h4>Original</h4>

        {before ? (
          <img src={before} alt="Original image" />
        ) : (
          <div className="empty-image">No original image</div>
        )}
      </div>

      <div>
        <h4>Model overlay</h4>

        {after ? (
          <img src={after} alt="Model overlay" />
        ) : (
          <div className="empty-image">No overlay available</div>
        )}
      </div>
    </div>
  );
}
