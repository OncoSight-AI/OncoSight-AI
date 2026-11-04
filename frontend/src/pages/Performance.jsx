import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import MetricCard from "../components/MetricCard";
import { performanceData } from "../data/demoData";

export default function Performance() {
  return (
    <>
      <div className="page-title">
        <div>
          <span className="eyebrow">MODEL EVALUATION</span>
          <h1>Performance</h1>
          <p>Illustrative metrics for the demo interface.</p>
        </div>
      </div>

      <div className="metric-grid">
        <MetricCard
          label="Accuracy"
          value="92%"
          description="Illustrative only"
        />
        <MetricCard
          label="Precision"
          value="89%"
          description="Illustrative only"
        />
        <MetricCard
          label="Recall"
          value="87%"
          description="Illustrative only"
        />
        <MetricCard
          label="F1 score"
          value="88%"
          description="Illustrative only"
        />
      </div>

      <section className="panel">
        <h2>Example evaluation metrics</h2>

        <div className="chart-area">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 100]} />
              <Tooltip />

              <Bar dataKey="value" fill="#159b9b" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <p className="muted">
          Replace these placeholder values with metrics measured on a separate,
          representative test set. Include confidence intervals and subgroup
          evaluation in a real study.
        </p>
      </section>
    </>
  );
}
