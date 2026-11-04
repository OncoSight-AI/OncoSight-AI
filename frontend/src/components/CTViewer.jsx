import { useState } from "react";

import { Maximize2, Minus, Plus } from "lucide-react";

export default function CTViewer({ imageUrl, title = "Image preview" }) {
  const [zoom, setZoom] = useState(100);

  return (
    <section className="viewer-panel">
      <div className="panel-heading">
        <div>
          <h3>{title}</h3>
          <p>Preview only · not a diagnostic viewer</p>
        </div>

        <div className="viewer-tools">
          <button
            className="icon-button"
            onClick={() => setZoom((v) => Math.max(50, v - 10))}
            aria-label="Zoom out"
          >
            <Minus size={16} />
          </button>

          <span>{zoom}%</span>

          <button
            className="icon-button"
            onClick={() => setZoom((v) => Math.min(200, v + 10))}
            aria-label="Zoom in"
          >
            <Plus size={16} />
          </button>

          <button
            className="icon-button"
            onClick={() => setZoom(100)}
            aria-label="Reset zoom"
          >
            <Maximize2 size={16} />
          </button>
        </div>
      </div>

      <div className="scan-canvas">
        {imageUrl ? (
          <img
            src={imageUrl}
            alt="Uploaded scan preview"
            style={{
              transform: `scale(${zoom / 100})`,
            }}
          />
        ) : (
          <div className="scan-placeholder">
            <div className="scan-circle">OS</div>
            <span>No scan loaded</span>
            <small>Upload an image to preview it here</small>
          </div>
        )}
      </div>
    </section>
  );
}
