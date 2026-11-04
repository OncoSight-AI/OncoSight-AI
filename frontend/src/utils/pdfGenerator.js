export function printReport(study) {
  const win = window.open("", "_blank", "noopener,noreferrer");

  if (!win) {
    window.alert("Allow pop-ups to print the report.");
    return;
  }

  const safe = (value) =>
    String(value ?? "—").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );

  win.document.write(`
    <!doctype html>
    <html>
      <head>
        <title>Research report ${safe(study.id)}</title>

        <style>
          body {
            font: 15px Arial;
            margin: 40px;
            color: #172033;
          }

          h1 {
            color: #155e75;
          }

          .note {
            background: #eef6f8;
            padding: 14px;
            border-radius: 8px;
          }
        </style>
      </head>

      <body>
        <h1>OncoSight AI — Research Summary</h1>

        <p><b>Study ID:</b> ${safe(study.id)}</p>
        <p><b>Study:</b> ${safe(study.name)}</p>
        <p><b>Modality:</b> ${safe(study.modality)}</p>
        <p><b>Model:</b> ${safe(study.model)}</p>
        <p><b>Status:</b> ${safe(study.status)}</p>
        <p><b>Date:</b> ${safe(study.date)}</p>

        <p>
          <b>Illustrative confidence:</b>
          ${
            study.confidence == null
              ? "—"
              : `${Math.round(study.confidence * 100)}%`
          }
        </p>

        <p><b>Output:</b> ${safe(study.finding)}</p>

        <div class="note">
          Educational/research demonstration only.
          Not for clinical diagnosis or treatment decisions.
        </div>

        <script>
          window.onload = () => window.print();
        <\/script>
      </body>
    </html>
  `);

  win.document.close();
}
