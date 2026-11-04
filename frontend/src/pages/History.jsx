import StatusBadge from "../components/StatusBadge";
import { useApp } from "../context/AppContext";

export default function History() {
  const { studies, setSelectedStudy } = useApp();

  return (
    <>
      <div className="page-title">
        <div>
          <span className="eyebrow">WORKSPACE RECORDS</span>
          <h1>Study history</h1>
          <p>Previously added demo and research records.</p>
        </div>
      </div>

      <section className="panel">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Modality</th>
                <th>Date</th>
                <th>Status</th>
                <th></th>
              </tr>
            </thead>

            <tbody>
              {studies.map((s) => (
                <tr key={s.id}>
                  <td>{s.id}</td>
                  <td>
                    <b>{s.name}</b>
                  </td>
                  <td>{s.modality}</td>
                  <td>{s.date}</td>

                  <td>
                    <StatusBadge status={s.status} />
                  </td>

                  <td>
                    <button
                      className="text-button"
                      onClick={() => setSelectedStudy(s)}
                    >
                      Select
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
