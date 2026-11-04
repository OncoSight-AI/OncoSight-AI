import { useState } from "react";
import { useNavigate } from "react-router-dom";

import UploadZone from "../components/UploadZone";
import Workflow from "../components/Workflow";

import { useApp } from "../context/AppContext";
import { validateFile } from "../utils/fileValidation";
import { uploadStudy } from "../services/api";
import { normalizeStudy } from "../services/normalizers";
import useDemoMode from "../hooks/useDemoMode";

export default function Upload() {
  const [file, setFile] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const { setStudies, studies, setSelectedStudy, setNotice } = useApp();

  const { demoMode } = useDemoMode();
  const navigate = useNavigate();

  async function submit(e) {
    e.preventDefault();

    const issue = validateFile(file);

    if (issue) {
      setError(issue);
      return;
    }

    setBusy(true);
    setError("");

    try {
      let study;

      if (!demoMode) {
        study = normalizeStudy(await uploadStudy(file));
      } else {
        study = normalizeStudy({
          id: `DEMO-${Date.now().toString().slice(-5)}`,
          name: file.name,
          modality: "Image",
          model: "Demo pipeline",
          status: "Review",
          date: new Date().toISOString().slice(0, 10),
          finding:
            "File received; connect a validated model backend for analysis.",
          confidence: null,
        });
      }

      setStudies([study, ...studies]);
      setSelectedStudy(study);
      setNotice("Study added to workspace.");

      navigate("/results");
    } catch (err) {
      setError(err.message || "Upload failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="page-title">
        <div>
          <span className="eyebrow">NEW ANALYSIS</span>
          <h1>Upload study</h1>
          <p>Add an image file to the research workspace.</p>
        </div>
      </div>

      <section className="panel upload-panel">
        <Workflow current={1} />

        <form onSubmit={submit}>
          <UploadZone
            file={file}
            onFile={(f) => {
              setFile(f);
              setError("");
            }}
          />

          {error && <p className="error-text">{error}</p>}

          <div className="notice">
            Use synthetic, de-identified or otherwise authorized research data
            only. This demo does not perform clinical diagnosis.
          </div>

          <button className="primary-button" disabled={busy}>
            {busy ? "Adding study..." : "Add to workspace"}
          </button>
        </form>
      </section>
    </>
  );
}
