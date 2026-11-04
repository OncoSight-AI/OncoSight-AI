import { useState } from "react";

export default function Settings() {
  const [name, setName] = useState("Researcher");
  const [compact, setCompact] = useState(false);

  return (
    <>
      <div className="page-title">
        <div>
          <span className="eyebrow">PREFERENCES</span>
          <h1>Settings</h1>
          <p>Local interface preferences.</p>
        </div>
      </div>

      <section className="panel settings-panel">
        <label className="field-label">
          Workspace display name
          <input value={name} onChange={(e) => setName(e.target.value)} />
        </label>

        <label className="toggle-row">
          <span>
            <b>Compact table spacing</b>
            <small>Preference preview only in this starter template.</small>
          </span>

          <input
            type="checkbox"
            checked={compact}
            onChange={(e) => setCompact(e.target.checked)}
          />
        </label>

        <div className="notice">
          Settings are currently stored only in page state and reset when the
          app reloads.
        </div>
      </section>
    </>
  );
}
