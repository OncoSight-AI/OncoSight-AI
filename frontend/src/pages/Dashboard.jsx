import { Activity, FileCheck2, ScanLine, Users } from "lucide-react";

import { Link } from "react-router-dom";

import StatCard from "../components/StatCard";
import StatusBadge from "../components/StatusBadge";
import Workflow from "../components/Workflow";

import { useApp } from "../context/AppContext";

export default function Dashboard() {
  const { studies, setSelectedStudy } = useApp();

  const completed = studies.filter((s) => s.status === "Completed").length;

  return (
    <>
      <div className="page-title">
        <div>
          <span className="eyebrow">OVERVIEW</span>
          <h1>Dashboard</h1>
          <p>Monitor research studies and model activity.</p>
        </div>

        <Link className="primary-link" to="/upload">
          + New study
        </Link>
      </div>

      <div className="stat-grid">
        <StatCard
          label="Total studies"
          value={studies.length}
          detail="Available in workspace"
          icon={ScanLine}
        />

        <StatCard
          label="Completed"
          value={completed}
          detail="Demo records"
          icon={FileCheck2}
          tone="green"
        />

        <StatCard
          label="Needs review"
          value={studies.filter((s) => s.status === "Review").length}
          detail="Human review required"
          icon={Users}
          tone="orange"
        />

        <StatCard
          label="Model pipeline"
          value="Ready"
          detail="Demo workflow"
          icon={Activity}
          tone="purple"
        />
      </div>

      <section className="panel">
        <div className="section-heading">
          <div>
            <h2>Recent studies</h2>
            <p>Example records for interface testing</p>
          </div>

          <Link to="/history">View history →</Link>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Study</th>
                <th>Modality</th>
                <th>Date</th>
                <th>Status</th>
                <th>Model output</th>
              </tr>
            </thead>

            <tbody>
              {studies.slice(0, 5).map((s) => (
                <tr key={s.id} onClick={() => setSelectedStudy(s)}>
                  <td>
                    <b>{s.name}</b>
                    <small>{s.id}</small>
                  </td>

                  <td>{s.modality}</td>
                  <td>{s.date}</td>

                  <td>
                    <StatusBadge status={s.status} />
                  </td>

                  <td>
                    {s.confidence == null
                      ? "—"
                      : `${Math.round(s.confidence * 100)}% illustrative score`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel">
        <div className="section-heading">
          <div>
            <h2>Analysis workflow</h2>
            <p>Typical research pipeline</p>
          </div>
        </div>

        <Workflow current={4} />
      </section>
    </>
  );
}
