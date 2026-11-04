import { useApp } from "../context/AppContext";
import { printReport } from "../utils/pdfGenerator";

export default function Reports() {
  const { studies } = useApp();

  return (
    <>
      <div className="page-title">
        <div>
          <span className="eyebrow">DOCUMENTATION</span>
          <h1>Reports</h1>
          <p>Generate print-ready research summaries.</p>
        </div>
      </div>

      <section className="panel">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Study</th>
                <th>ID</th>
                <th>Status</th>
                <th>Action</th>
              </tr>
            </thead>

            <tbody>
              {studies.map((s) => (
                <tr key={s.id}>
                  <td>{s.name}</td>
                  <td>{s.id}</td>
                  <td>{s.status}</td>

                  <td>
                    <button
                      className="text-button"
                      onClick={() => printReport(s)}
                    >
                      Print / Save PDF
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
