import { useEffect, useRef, useState } from 'react';
import { AlertTriangle, FileUp, Loader2, X } from 'lucide-react';
import { predictVolume } from '../api.js';
import { readNifti } from '../nifti.js';
import { formatBytes } from '../store.js';
import ScanViewer from '../components/ScanViewer.jsx';

const isNifti = (name) => /\.nii(\.gz)?$/i.test(name);

export default function Upload({ health, onComplete }) {
  const [file, setFile] = useState(null);
  const [volume, setVolume] = useState(null);
  const [note, setNote] = useState(null);
  const [stage, setStage] = useState('idle'); // idle | uploading | processing | error
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [drag, setDrag] = useState(false);
  const abortRef = useRef(null);
  const inputRef = useRef(null);
  const token = useRef(0);

  useEffect(() => () => abortRef.current?.(), []);

  const accept = (f) => {
    setError(null);
    setNote(null);
    setVolume(null);
    if (!f) return;
    if (!isNifti(f.name)) {
      setFile(null);
      setError('Unsupported file. Choose a .nii or .nii.gz lung CT volume.');
      return;
    }
    setFile(f);
    setStage('idle');
    const t = ++token.current;
    readNifti(f)
      .then((v) => t === token.current && setVolume(v))
      .catch((e) => t === token.current && setNote(`Preview unavailable: ${e.message}`));
  };

  const start = async () => {
    if (!file) return;
    setError(null);
    setProgress(0);
    setStage('uploading');
    const t0 = performance.now();
    const { promise, abort } = predictVolume(file, (p) => {
      setProgress(p);
      if (p >= 1) setStage('processing');
    });
    abortRef.current = abort;
    try {
      const raw = await promise;
      abortRef.current = null;
      onComplete({ file, raw, volume, elapsedMs: performance.now() - t0 });
    } catch (e) {
      abortRef.current = null;
      if (e.name === 'AbortError') {
        setStage('idle');
      } else {
        setError(e.message);
        setStage('error');
      }
    }
  };

  const busy = stage === 'uploading' || stage === 'processing';
  const steps = [
    { id: 'select', label: 'Select volume', done: !!file },
    { id: 'uploading', label: 'Upload', done: stage === 'processing' },
    { id: 'processing', label: 'Model inference', done: false },
  ];

  return (
    <div className="page">
      <div className="page-head">
        <div className="mono eyebrow">New analysis</div>
        <h1>Upload a lung CT volume</h1>
        <p className="lede">Accepted formats: .nii and .nii.gz. The file is sent to the backend as multipart field “file”.</p>
      </div>

      {health.state === 'offline' && (
        <div className="alert warn" role="alert">
          <AlertTriangle size={16} />
          <span>The backend is currently offline. You can select a file, but analysis will fail until the server is reachable.</span>
        </div>
      )}

      <div className="upload-grid">
        <div className="panel upload-panel">
          <div
            className={`dropzone ${drag ? 'is-drag' : ''} ${busy ? 'is-busy' : ''}`}
            onDragOver={(e) => {
              e.preventDefault();
              if (!busy) setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              if (!busy) accept(e.dataTransfer.files?.[0]);
            }}
            onClick={() => !busy && inputRef.current?.click()}
            role="button"
            tabIndex={busy ? -1 : 0}
            onKeyDown={(e) => {
              if (!busy && (e.key === 'Enter' || e.key === ' ')) {
                e.preventDefault();
                inputRef.current?.click();
              }
            }}
            aria-label="Choose a NIfTI file"
          >
            <FileUp size={28} strokeWidth={1.3} />
            <div className="dz-title">{file ? file.name : 'Drop a .nii / .nii.gz file here'}</div>
            <div className="mono dz-sub">{file ? formatBytes(file.size) : 'or click to browse'}</div>
            <input
              ref={inputRef}
              type="file"
              accept=".nii,.nii.gz,.gz"
              hidden
              onChange={(e) => {
                accept(e.target.files?.[0]);
                e.target.value = '';
              }}
            />
          </div>

          <ol className="steps">
            {steps.map((s) => (
              <li key={s.id} className={`${s.done ? 'is-done' : ''} ${stage === s.id || (s.id === 'select' && file && stage === 'idle') ? 'is-now' : ''}`}>
                <span className="mono">{s.label}</span>
              </li>
            ))}
          </ol>

          {(stage === 'uploading' || stage === 'processing') && (
            <div className="progress" aria-live="polite">
              <div className="bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress * 100)}>
                <span style={{ width: `${Math.round(progress * 100)}%` }} className={stage === 'processing' ? 'is-pulse' : ''} />
              </div>
              <div className="mono prog-text">
                {stage === 'uploading' ? `Uploading · ${Math.round(progress * 100)}%` : (
                  <><Loader2 size={13} className="spin" /> Upload complete · waiting for model inference</>
                )}
              </div>
            </div>
          )}

          {error && (
            <div className="alert err" role="alert">
              <AlertTriangle size={16} />
              <span>{error}</span>
            </div>
          )}
          {note && !error && <div className="alert warn"><AlertTriangle size={16} /><span>{note}. Analysis can still run.</span></div>}

          <div className="actions">
            <button type="button" className="btn btn-primary" onClick={start} disabled={!file || busy}>
              {busy ? <><Loader2 size={16} className="spin" /> Analysing…</> : 'Run analysis'}
            </button>
            {busy && (
              <button type="button" className="btn btn-ghost" onClick={() => abortRef.current?.()}>
                <X size={15} /> Cancel
              </button>
            )}
          </div>
        </div>

        <ScanViewer
          volume={volume}
          label={file?.name}
          empty={{
            title: file && !note ? 'Reading volume…' : 'Local preview',
            text: note
              ? 'The volume could not be decoded for preview, so no image is shown.'
              : 'After you choose a file, its slices are decoded here in your browser before anything is sent.',
          }}
        />
      </div>
    </div>
  );
}
