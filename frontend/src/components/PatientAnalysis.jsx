
import { useEffect, useState } from 'react';
import {
  Check,
  Info,
  AlertTriangle,
  ShieldCheck,
} from 'lucide-react';

import { formatConfidence } from '../api.js';
import { formatBytes } from '../store.js';

const KEY_ORDER = [
  'slice',
  'slice_index',
  'z',
  'label',
  'class',
  'class_name',
  'confidence',
  'conf',
  'score',
  'bbox',
  'box',
  'xyxy',
  'area',
  'size',
  'diameter_mm',
  'size_mm',
  'centroid',
  'location',
];

const MAX_ROWS = 50;

const isSimple = (v) =>
  v == null ||
  ['string', 'number', 'boolean'].includes(typeof v) ||
  (
    Array.isArray(v) &&
    v.length <= 4 &&
    v.every((x) => typeof x === 'number')
  );

function columnsFor(items) {
  const seen = new Set();

  items.slice(0, MAX_ROWS).forEach((d) =>
    Object.keys(d).forEach((k) => {
      if (isSimple(d[k])) seen.add(k);
    })
  );

  const all = [...seen];

  const ordered = [
    ...KEY_ORDER.filter((k) => seen.has(k)),
    ...all.filter((k) => !KEY_ORDER.includes(k)),
  ];

  return ordered.slice(0, 6);
}

function cell(v) {
  if (v == null) return '—';

  if (typeof v === 'number') {
    return Number.isInteger(v) ? String(v) : v.toFixed(3);
  }

  if (Array.isArray(v)) {
    return v
      .map((x) =>
        Number.isInteger(x) ? x : x.toFixed(1)
      )
      .join(', ');
  }

  return String(v);
}

function Field({ label, value, mono }) {
  const empty = value == null || value === '';

  return (
    <div className="sw-field">
      <div className="sw-label">{label}</div>

      <div
        className={`sw-value ${mono ? 'mono' : ''} ${empty ? 'is-empty' : ''
          }`}
      >
        {empty ? 'Not provided' : value}
      </div>
    </div>
  );
}

function Stat({ label, value, description }) {
  return (
    <div className="sw-field">
      <div className="sw-label">{label}</div>

      <div className="sw-value">
        {value ?? 'Not available'}
      </div>

      {description && (
        <div className="sw-muted">{description}</div>
      )}
    </div>
  );
}

export default function PatientAnalysis({
  entry,
  volume,
  onNotes,
  readOnly,
}) {
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
          Run a new analysis to see the scan record,
          candidate findings and research notes here.
        </p>
      </section>
    );
  }

  const count =
    typeof record.count === 'number' &&
      Number.isFinite(record.count)
      ? record.count
      : null;

  const hasCandidates = count !== null && count > 0;
  const noCandidates = count === 0;

  const dets = Array.isArray(record.detections)
    ? record.detections
    : [];

  const cols = dets.length ? columnsFor(dets) : [];

  const dims = volume
    ? `${volume.nx} × ${volume.ny} × ${volume.nz} voxels · ${volume.spacing
      .map((s) => s.toFixed(2))
      .join(' × ')} mm`
    : null;

  const slicesProcessed = record.slicesProcessed;
  const slicesWithSegmentation =
    record.slicesWithSegmentation;
  const segmentedVoxels = record.segmentedVoxels;

  // AI verdict
  const verdict = hasCandidates
    ? 'Attention recommended — possible nodule detected'
    : noCandidates
      ? 'Reassuring result — no nodule candidate detected'
      : 'Result information unavailable';

  // Plain-language explanation
  const explanation = hasCandidates
    ? `The AI model identified ${count} possible lung nodule candidate(s) in this CT scan. The highlighted areas should be reviewed by a qualified radiologist. This finding does not confirm cancer.`
    : noCandidates
      ? 'Good news: the AI model did not identify any lung nodule candidates in the analysed scan. This is a reassuring AI screening result. However, it does not rule out cancer or other lung conditions, and a qualified radiologist should still review the scan.'
      : 'The backend did not provide a usable candidate count. No reliable conclusion can be made from the available information.';

  const simpleConclusion = hasCandidates
    ? 'The AI has marked a possible nodule area that needs a doctor’s attention. It does not mean the person has cancer.'
    : noCandidates
      ? 'Good news! The AI did not find any suspicious nodule candidates in this scan. This is encouraging, but only a doctor can properly assess the scan and determine whether further tests are needed.'
      : 'The AI result is incomplete, so no reliable conclusion can be made.';

  return (
    <section className="swiss" aria-label="Patient analysis">

      {/* REPORT HEADER */}
      <div className="sw-head">
        <div>
          <div className="sw-eyebrow">Patient analysis</div>
          <h2>{entry.filename}</h2>
        </div>

        <span className="sw-badge">
          {record.status || 'Analysis complete'}
        </span>
      </div>

      {/* SAFETY NOTICE */}
      <div className="sw-banner" role="note">
        <Info size={16} />

        <p>
          Research prototype output. Candidate findings are
          AI-generated observations and require expert review.
          This system does not provide a medical diagnosis.
        </p>
      </div>

      {/* AI VERDICT */}
      <div className="sw-card">
        <div className="sw-eyebrow">
          OncoSight AI · Model assessment
        </div>

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            margin: '12px 0',
          }}
        >
          {hasCandidates ? (
            <AlertTriangle
              size={26}
              color="#d97706"
            />
          ) : noCandidates ? (
            <ShieldCheck
              size={26}
              color="#56866c"
            />
          ) : (
            <Info size={26} />
          )}

          <h3 style={{ margin: 0 }}>
            {verdict}
          </h3>
        </div>

        <p>{explanation}</p>

        <div
          style={{
            marginTop: '14px',
            padding: '14px',
            borderRadius: '8px',
            background: noCandidates
              ? 'rgba(86,134,108,0.10)'
              : 'rgba(128,128,128,0.08)',
          }}
        >
          <strong>What does this mean?</strong>

          <p style={{ marginBottom: 0 }}>
            {simpleConclusion}
          </p>
        </div>
      </div>

      {/* PATIENT AND SCAN INFORMATION */}
      <div className="sw-grid">
        <Field
          label="Patient ID"
          value={record.patientId}
        />

        <Field
          label="Scan date"
          value={record.scanDate}
        />

        <Field
          label="Modality"
          value={record.modality}
        />

        <Field
          label="Analysis status"
          value={record.status}
        />

        <Field
          label="Candidate count"
          value={
            count !== null
              ? String(count)
              : null
          }
        />

        <Field
          label="Model confidence"
          value={
            formatConfidence(record.confidence)
              ? `${formatConfidence(record.confidence)}${record.confidenceNote
                ? ` (${record.confidenceNote})`
                : ''
              }`
              : null
          }
        />

        <Field
          label="Analysed"
          value={new Date(entry.savedAt).toLocaleString()}
        />

        <Field
          label="Source file"
          value={`${entry.filename} · ${formatBytes(entry.size)}`}
        />

        <Field
          label="Volume (file header)"
          value={dims}
        />

        <Field
          label="Scan ID"
          value={
            record.scanId != null
              ? String(record.scanId)
              : null
          }
          mono
        />
      </div>

      {/* SEGMENTATION STATISTICS */}
      <div className="sw-card">
        <h3>Segmentation observations</h3>

        <p>
          {record.scanId != null || record.maskRef
            ? 'A segmentation mask is associated with this scan. The viewer can display the mask over the CT image when the volume and mask grids match.'
            : 'The backend did not report a segmentation mask for this scan.'}
        </p>

        <div className="sw-grid">
          <Stat
            label="Slices processed"
            value={slicesProcessed}
          />

          <Stat
            label="Slices with candidates"
            value={slicesWithSegmentation}
          />

          <Stat
            label="Segmented voxels"
            value={
              Number.isFinite(segmentedVoxels)
                ? segmentedVoxels.toLocaleString()
                : null
            }
          />

          <Stat
            label="Detection class"
            value={
              record.detectionClass || 'lung_nodule'
            }
          />
        </div>

        <p className="sw-muted">
          Red regions represent pixels included in the
          model's segmentation mask. They are model-marked
          regions, not confirmed cancer tissue.
        </p>
      </div>

      {/* CANDIDATE FINDINGS */}
      <div className="sw-card sw-table-card">
        <h3>Candidate findings</h3>

        {dets.length ? (
          <>
            <div className="sw-table-wrap">
              <table className="sw-table">
                <thead>
                  <tr>
                    {cols.map((c) => (
                      <th key={c}>
                        {c.replace(/_/g, ' ')}
                      </th>
                    ))}
                  </tr>
                </thead>

                <tbody>
                  {dets.slice(0, MAX_ROWS).map((d, i) => (
                    <tr key={i}>
                      {cols.map((c) => (
                        <td key={c} className="mono">
                          {cell(d[c])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {dets.length > MAX_ROWS && (
              <p className="sw-more">
                Showing {MAX_ROWS} of {dets.length}.
                The complete list is in the raw backend response.
              </p>
            )}
          </>
        ) : (
          <p>
            {noCandidates
              ? 'The model reported no candidates for this volume.'
              : 'The backend did not return a per-candidate findings list.'}
          </p>
        )}
      </div>

      {/* RESEARCH NOTES */}
      <div className="sw-card">
        <h3>Research notes</h3>

        {readOnly ? (
          <p className={notes ? '' : 'sw-muted'}>
            {notes || 'No notes recorded.'}
          </p>
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

              {saved && (
                <span className="sw-saved">
                  <Check size={14} />
                  Saved
                </span>
              )}
            </div>
          </>
        )}
      </div>

      {/* FINAL DISCLAIMER */}
      <div className="sw-banner" role="note">
        <Info size={16} />

        <p>
          <strong>Important medical disclaimer:</strong>{' '}
          OncoSight AI is an experimental research prototype.
          Its model detects possible lung nodule patterns, not
          cancer itself. A positive finding does not confirm
          cancer, and a negative finding does not rule it out.
          Always consult a qualified radiologist or doctor.
        </p>
      </div>

    </section>
  );
}
