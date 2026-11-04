import StatusBadge from "../components/StatusBadge";
import MetricCard from "../components/MetricCard";

import { useApp } from "../context/AppContext";
import { printReport } from "../utils/pdfGenerator";

export default function Results() {
  const { selectedStudy } = useApp();
  const s = selectedStudy;

  return (
    <>
      <div className="page-title">
        <div>
          <span className="eyebrow">ANALYSIS OUTPUT</span>
          <h1>Results</h1>
          <p>Review the selected study output.</p>
        </div>

        {s && (
          <button className="secondary" onClick={() => printReport(s)}>
            Export report
          </button>
        )}
      </div>

      {!s ? (
        <div className="panel">
          No study selected. Upload or select a study first.
        </div>
      ) : (
        <>
          <section className="panel">
            <div className="section-heading">
              <div>
                <h2>{s.name}</h2>
                <p>
                  {s.id} · {s.modality} · {s.date}
                </p>
              </div>

              <StatusBadge status={s.status} />
            </div>

            <div className="metric-grid">
              <MetricCard
                label="Illustrative score"
                value={
                  s.confidence == null
                    ? "—"
                    : `${Math.round(s.confidence * 100)}%`
                }
                description="Demo value, not a clinical probability"
              />

              <MetricCard
                label="Model"
                value={s.model}
                description="Configured research pipeline"
              />

              <MetricCard
                label="Output status"
                value={s.status}
                description="Requires appropriate review"
              />
            </div>

            <div className="result-summary">
              <h3>Model summary</h3>
              <p>{s.finding}</p>
              <small>{s.notes}</small>
            </div>
          </section>

          <div className="notice">
            AI-generated outputs can be wrong or biased. Do not use this page to
            make diagnosis or treatment decisions.
          </div>
        </>
      )}
    </>
  );
}
