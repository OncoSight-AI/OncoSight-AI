import { Download, UploadCloud } from 'lucide-react';
import ScanViewer from '../components/ScanViewer.jsx';
import AnalysisSummary from '../components/AnalysisSummary.jsx';
import PatientAnalysis from '../components/PatientAnalysis.jsx';
import { maskUrl } from '../api.js';

export default function Results({ entry, vols, navigate, onNotes }) {
  if (!entry) {
    return (
      <div className="page">
        <div className="page-head">
          <div className="mono eyebrow">Analysis report</div>
          <h1>No report to show</h1>
          <p className="lede">Run an analysis or open a scan from your history to see its report.</p>
        </div>
        <div className="actions">
          <button type="button" className="btn btn-primary" onClick={() => navigate('upload')}><UploadCloud size={16} /> New analysis</button>
          <button type="button" className="btn btn-ghost" onClick={() => navigate('history')}>Scan history</button>
        </div>
      </div>
    );
  }

  const { record } = entry;
  const raw = record.raw === undefined ? null : JSON.stringify(record.raw, null, 2);

  return (
    <div className="page">
      <div className="page-head row">
        <div>
          <div className="mono eyebrow">Analysis report</div>
          <h1>{entry.filename}</h1>
          <p className="lede mono">{new Date(entry.savedAt).toLocaleString()}</p>
        </div>
        {record.scanId != null && (
          <a className="btn btn-ghost" href={maskUrl(record.scanId)} download>
            <Download size={15} /> Download mask
          </a>
        )}
      </div>

      <div className="dash-grid">
        <ScanViewer
          volume={vols?.volume}
          mask={vols?.mask}
          label={entry.filename}
          empty={{
            title: 'Volume not retained',
            text: 'This record was restored from local history. The report data is intact, but the CT volume is not stored, so no slices can be shown. Upload the file again to view it.',
            action: <button type="button" className="btn btn-ghost" onClick={() => navigate('upload')}>Upload again</button>,
          }}
        />
        <AnalysisSummary record={record} status="complete" elapsedMs={entry.elapsedMs} maskState={vols?.maskState} />
      </div>

      <PatientAnalysis entry={entry} volume={vols?.volume} onNotes={onNotes} />

      <details className="panel raw">
        <summary className="mono">Backend response (raw)</summary>
        {raw ? <pre>{raw}</pre> : <p className="muted pad">The raw payload was dropped from this older record to save browser storage.</p>}
      </details>
    </div>
  );
}
