import { useState } from "react";

import CTViewer from "../components/CTViewer";
import ComparisonViewer from "../components/ComparisonViewer";

import { useApp } from "../context/AppContext";

export default function CTViewerPage() {
  const { selectedStudy } = useApp();
  const [url, setUrl] = useState("");

  return (
    <>
      <div className="page-title">
        <div>
          <span className="eyebrow">IMAGE EXPLORER</span>
          <h1>Image viewer</h1>
          <p>Basic browser preview for supported image files.</p>
        </div>
      </div>

      <section className="panel">
        <label className="field-label">
          Preview image URL (optional)
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="Paste a publicly accessible image URL"
          />
        </label>

        <CTViewer
          imageUrl={url}
          title={selectedStudy?.name || "Image preview"}
        />
      </section>

      <section className="panel">
        <h2>Image comparison</h2>

        <p className="muted">
          Connect a backend segmentation or overlay endpoint to display model
          output.
        </p>

        <ComparisonViewer />
      </section>
    </>
  );
}
