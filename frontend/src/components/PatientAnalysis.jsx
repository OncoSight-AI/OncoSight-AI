import { useEffect, useState } from 'react';
import { Check, Info } from 'lucide-react';
import { formatConfidence } from '../api.js';
import { formatBytes } from '../store.js';

const KEY_ORDER = ['slice', 'slice_index', 'z', 'label', 'class', 'class_name', 'confidence', 'conf', 'score', 'bbox', 'box', 'xyxy', 'area', 'size', 'diameter_mm', 'size_mm', 'centroid', 'location'];
const MAX_ROWS = 50;

const isSimple = (v) =>
  v == null || ['string', 'number', 'boolean'].includes(typeof v) || (Array.isArray(v) && v.length <= 4 && v.every((x) => typeof x === 'number'));

function columnsFor(items) {
  const seen = new Set();
  items.slice(0, MAX_ROWS).forEach((d) => Object.keys(d).forEach((k) => isSimple(d[k]) && seen.add(k)));
  const all = [...seen];
  const ordered = [...KEY_ORDER.filter((k) => seen.has(k)), ...all.filter((k) => !KEY_ORDER.includes(k))];
  return ordered.slice(0, 6);
}

function cell(v) {
  if (v == null) return '—';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(3);
  if (Array.isArray(v)) return v.map((x) => (Number.isInteger(x) ? x : x.toFixed(1))).join(', ');
  return String(v);
}

function Field({ label, value, mono }) {
  const empty = value == null || value === '';
  return (
    <div className="sw-field">
      <div className="sw-label">{label}</div>
      <div className={`sw-value ${mono ? 'mono' : ''} ${empty ? 'is-empty' : ''}`}>{empty ? 'Not provided' : value}</div>
    </div>
  );
}

/**
 * Swiss Medical treatment: warm ivory, white surfaces, charcoal type, sage accents.
 * Only fields that exist in the backend response or the uploaded file's header are filled.
 */
export default function PatientAnalysis({ entry, volume, onNotes, readOnly }) {
  const record = entry?.record;
  const [notes, setNotes] = useState(entry?.notes || '');
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setNotes(entry?.notes || '');
    setSaved(false);
  }, [entry?.id]);

  if (!entry) {
    return (
      <section className="swiss" aria-label="Patient analysis">
        <div className="sw-head">
          <div>
            <div className="sw-eyebrow">Patient analysis</div>
            <h2>No analysis yet</h2>
          </div>
        </div>
        <p className="sw-empty">
          Run a new analysis to see the scan record, candidate findings and research notes here. Only values returned by the backend
          or read from the uploaded file are displayed.
        </p>
      </section>
    );
  }

  const dets = record.detections;
  const cols = dets?.length ? columnsFor(dets) : [];
  const dims = volume ? `${volume.nx} × ${volume.ny} × ${volume.nz} voxels · ${volume.spacing.map((s) => s.toFixed(2)).join(' × ')} mm` : null;

  return (
    <section className="swiss" aria-label="Patient analysis">
      <div className="sw-head">
        <div>
          <div className="sw-eyebrow">Patient analysis</div>
          <h2>{entry.filename}</h2>
        </div>
        <span className="sw-badge">{record.status}</span>
      </div>

      <div className="sw-banner" role="note">
        <Info size={16} />
        <p>
          Research prototype output. Candidate findings are model detections that require expert review before any use and must not be
          read as a diagnosis.
        </p>
      </div>

      <div className="sw-grid">
        <Field label="Patient ID" value={record.patientId} />
        <Field label="Scan date" value={record.scanDate} />
        <Field label="Modality" value={record.modality} />
        <Field label="Analysis status" value={record.status} />
        <Field label="Candidate count" value={record.count != null ? String(record.count) : null} />
        <Field label="Model confidence" value={formatConfidence(record.confidence) ? `${formatConfidence(record.confidence)}${record.confidenceNote ? ` (${record.confidenceNote})` : ''}` : null} />
        <Field label="Analysed" value={new Date(entry.savedAt).toLocaleString()} />
        <Field label="Source file" value={`${entry.filename} · ${formatBytes(entry.size)}`} />
        <Field label="Volume (file header)" value={dims} />
        <Field label="Scan ID" value={record.scanId != null ? String(record.scanId) : null} mono />
      </div>

      <div className="sw-card">
        <h3>Segmentation observations</h3>
        <p>
          {record.scanId != null || record.maskRef
            ? 'A segmentation mask is available from the backend for this scan. Overlay is shown in the viewer when the mask matches the volume grid.'
            : 'The backend did not report a segmentation mask for this scan.'}
        </p>
      </div>

      <div className="sw-card sw-table-card">
        <h3>Candidate findings</h3>
        {dets?.length ? (
          <>
            <div className="sw-table-wrap">
              <table className="sw-table">
                <thead>
                  <tr>{cols.map((c) => <th key={c}>{c.replace(/_/g, ' ')}</th>)}</tr>
                </thead>
                <tbody>
                  {dets.slice(0, MAX_ROWS).map((d, i) => (
                    <tr key={i}>{cols.map((c) => <td key={c} className="mono">{cell(d[c])}</td>)}</tr>
                  ))}
                </tbody>
              </table>
            </div>
            {dets.length > MAX_ROWS && <p className="sw-more">Showing {MAX_ROWS} of {dets.length}. The complete list is in the raw backend response.</p>}
          </>
        ) : (
          <p>{record.count === 0 ? 'The model reported no candidates for this volume.' : 'The backend did not return a per-candidate list.'}</p>
        )}
      </div>

      <div className="sw-card">
        <h3>Research notes</h3>
        {readOnly ? (
          <p className={notes ? '' : 'sw-muted'}>{notes || 'No notes recorded.'}</p>
        ) : (
          <>
            <textarea
              value={notes}
              onChange={(e) => {
                setNotes(e.target.value);
                setSaved(false);
              }}
              placeholder="Your own observations about this scan (stored only on this device)"
              rows={4}
            />
            <div className="sw-actions">
              <button
                type="button"
                className="sw-btn"
                onClick={() => {
                  onNotes?.(notes);
                  setSaved(true);
                }}
              >
                Save notes
              </button>
              {saved && <span className="sw-saved"><Check size={14} /> Saved</span>}
            </div>
          </>
        )}
      </div>
    </section>
  );
}
