import { Trash2, UploadCloud } from 'lucide-react';
import { formatBytes } from '../store.js';
import { formatConfidence } from '../api.js';

export default function History({ history, onOpen, onDelete, onClear, navigate }) {
  return (
    <div className="page">
      <div className="page-head row">
        <div>
          <div className="mono eyebrow">Scan history</div>
          <h1>Completed analyses</h1>
          <p className="lede">Stored in this browser only. The backend does not provide a history endpoint.</p>
        </div>
        {history.length > 0 && (
          <button
            type="button"
            className="btn btn-ghost"
            onClick={() => {
              if (window.confirm('Remove all stored scan records from this device?')) onClear();
            }}
          >
            <Trash2 size={15} /> Clear all
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <div className="panel empty-card">
          <p>No completed scans yet.</p>
          <button type="button" className="btn btn-primary" onClick={() => navigate('upload')}><UploadCloud size={16} /> New analysis</button>
        </div>
      ) : (
        <div className="panel table-panel">
          <div className="hist-wrap">
            <table className="hist">
              <thead>
                <tr>
                  <th>File</th>
                  <th>Date</th>
                  <th>Size</th>
                  <th>Candidates</th>
                  <th>Confidence</th>
                  <th>Status</th>
                  <th aria-label="Actions" />
                </tr>
              </thead>
              <tbody>
                {history.map((e) => (
                  <tr key={e.id}>
                    <td>
                      <button type="button" className="row-link" onClick={() => onOpen(e.id)}>{e.filename}</button>
                      {e.record.scanId != null && <div className="mono sub">{String(e.record.scanId)}</div>}
                    </td>
                    <td className="mono">{new Date(e.savedAt).toLocaleString()}</td>
                    <td className="mono">{formatBytes(e.size)}</td>
                    <td className="mono">{e.record.count ?? '—'}</td>
                    <td className="mono">{formatConfidence(e.record.confidence) ?? '—'}</td>
                    <td><span className="pill ok"><span className="dot" />{e.record.status}</span></td>
                    <td className="t-actions">
                      <button type="button" className="btn btn-ghost sm" onClick={() => onOpen(e.id)}>Open</button>
                      <button type="button" className="icon-btn sm" onClick={() => onDelete(e.id)} aria-label={`Delete ${e.filename}`}>
                        <Trash2 size={15} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
