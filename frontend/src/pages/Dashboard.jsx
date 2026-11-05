import { ArrowRight, UploadCloud } from 'lucide-react';
import ScanViewer from '../components/ScanViewer.jsx';
import AnalysisSummary from '../components/AnalysisSummary.jsx';
import PatientAnalysis from '../components/PatientAnalysis.jsx';
import { relativeTime } from '../store.js';

export default function Dashboard({ entry, vols, health, history, navigate, openEntry }) {
  const model = typeof health.data?.model === 'string' ? health.data.model : 'YOLOv8n-seg';
  const recent = history.slice(0, 5);

  return (
    <div className="page">
      <div className="page-head">
        <div className="mono eyebrow">Research workstation · Lung CT</div>
        <h1>Lung Intelligence</h1>
        <p className="lede">Advanced pulmonary imaging and AI-assisted analysis.</p>
      </div>

      <div className="metrics mono" aria-label="Technical status">
        <span><i>Backend</i><b className="cap">{health.state}</b>{health.latency != null ? ` · ${health.latency} ms` : ''}</span>
        <span><i>Model</i>{model}</span>
        <span><i>Input</i>NIfTI volume</span>
        <span><i>Stored scans</i>{history.length}</span>
        <span><i>Last analysis</i>{history[0] ? relativeTime(history[0].savedAt) : '—'}</span>
      </div>

      <div className="dash-grid">
        <ScanViewer
          volume={vols?.volume}
          mask={vols?.mask}
          label={entry?.filename}
          empty={{
            title: entry ? 'Volume not retained in this session' : 'No volume loaded',
            text: entry
              ? 'Scan records are kept on this device, but the volume itself is not stored. Upload the file again to view its slices.'
              : 'Upload a .nii or .nii.gz lung CT volume. Slices are decoded in your browser from your own file, so what you see is the real scan.',
            action: (
              <button type="button" className="btn btn-primary" onClick={() => navigate('upload')}>
                <UploadCloud size={16} /> New analysis
              </button>
            ),
          }}
        />
        <AnalysisSummary
          record={entry?.record}
          status={entry ? 'complete' : 'idle'}
          elapsedMs={entry?.elapsedMs}
          maskState={vols?.maskState}
        />
      </div>

      <div className="dash-lower">
        <PatientAnalysis entry={entry} volume={vols?.volume} readOnly />

        <aside className="panel recent" aria-label="Recent scans">
          <div className="panel-head">
            <div>
              <div className="mono eyebrow">Activity</div>
              <h2>Recent scans</h2>
            </div>
          </div>
          {recent.length === 0 ? (
            <p className="muted pad">No scans yet. Completed analyses are listed here.</p>
          ) : (
            <ul className="recent-list">
              {recent.map((e) => (
                <li key={e.id}>
                  <button type="button" onClick={() => openEntry(e.id)}>
                    <span className="r-main">
                      <span className="r-name">{e.filename}</span>
                      <span className="mono r-meta">
                        {relativeTime(e.savedAt)} · {e.record.count != null ? `${e.record.count} candidate${e.record.count === 1 ? '' : 's'}` : 'count not reported'}
                      </span>
                    </span>
                    <ArrowRight size={15} />
                  </button>
                </li>
              ))}
            </ul>
          )}
          {history.length > 0 && (
            <button type="button" className="link-btn" onClick={() => navigate('history')}>View all scans</button>
          )}
        </aside>
      </div>
    </div>
  );
}
