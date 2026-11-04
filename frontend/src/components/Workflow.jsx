const steps = [
  "Upload image",
  "Pre-processing",
  "Model analysis",
  "Review results",
];

export default function Workflow({ current = 3 }) {
  return (
    <div className="workflow">
      {steps.map((step, i) => (
        <div
          className={`workflow-step ${i < current ? "done" : ""}`}
          key={step}
        >
          <span>{i + 1}</span>
          <small>{step}</small>
        </div>
      ))}
    </div>
  );
}
