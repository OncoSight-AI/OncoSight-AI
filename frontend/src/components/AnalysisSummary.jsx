import { formatConfidence } from '../api.js';

const MASK_TEXT = {
  ready: 'Mask loaded · overlay available',
  loading: 'Retrieving mask…',
  unavailable: 'Mask could not be displayed',
};

function Row({ label, children, muted }) {
  return (
    <div className="kv">
      <dt className="mono">{label}</dt>
      <dd className={muted ? 'is-muted' : ''}>{children}</dd>
    </div>
  );
}

const NA = <span title="Not returned by the backend">—</span>;

export default function AnalysisSummary({ record, status = 'idle', elapsedMs, maskState = 'none' }) {
  const done = status === 'complete';
  const conf = record ? formatConfidence(record.confidence) : null;

  let segmentation = NA;
  if (MASK_TEXT[maskState]) segmentation = MASK_TEXT[maskState];
  else if (record?.scanId) segmentation = 'Downloadable via /download-mask';
  else if (record?.maskRef) segmentation = 'Reported by backend';

  return (
    <aside className="panel summary" aria-label="Healthcare analysis">
      <div className="panel-head">
        <div>
          <div className="mono eyebrow">Healthcare analysis</div>
          <h2>Model output</h2>
        </div>
        <span className={`pill ${done ? 'ok' : status === 'error' ? 'err' : ''}`}>
          <span className="dot" />
          {done ? record?.status || 'completed' : status}
        </span>
      </div>

      <dl className="kv-list">
        <Row label="Candidate count">
          {record?.count != null ? <span className="big">{record.count}</span> : NA}
          {record?.count != null && <span className="sub"> candidate{record.count === 1 ? '' : 's'}</span>}
        </Row>
        <Row label="Model confidence">
          {conf ? (
            <>
              {conf}
              {record.confidenceNote && <span className="sub"> · {record.confidenceNote}</span>}
            </>
          ) : (
            NA
          )}
        </Row>
        <Row label="Segmentation">{segmentation}</Row>
        <Row label="Model">
          {record?.model ? record.model : <>YOLOv8n-seg<span className="sub"> · configured</span></>}
        </Row>
        <Row label="Scan ID">{record?.scanId != null ? <span className="mono wrap">{String(record.scanId)}</span> : NA}</Row>
        <Row label="Slices processed">{record?.slicesProcessed != null ? record.slicesProcessed : NA}</Row>
        <Row label="Round trip">{elapsedMs != null ? `${(elapsedMs / 1000).toFixed(1)} s` : NA}</Row>
      </dl>

      {record?.message && <p className="note">{record.message}</p>}

      <p className="disclaimer">
        Candidates are model detections for research review. They are not confirmed nodules and not a diagnosis of cancer.
      </p>
    </aside>
  );
}
