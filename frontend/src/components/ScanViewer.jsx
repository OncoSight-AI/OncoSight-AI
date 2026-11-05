import { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight, Layers, Maximize2, Minimize2, RotateCcw, ScanLine, ZoomIn, ZoomOut } from 'lucide-react';
import { WINDOW_PRESETS, drawMask, drawSlice, sameGrid, windowFor } from '../nifti.js';

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

function Thumbs({ volume, win, z, onPick }) {
  const refs = useRef([]);
  const count = Math.min(12, volume.nz);
  const idx = useMemo(
    () => Array.from({ length: count }, (_, i) => (count === 1 ? 0 : Math.round((i * (volume.nz - 1)) / (count - 1)))),
    [count, volume.nz],
  );

  useEffect(() => {
    const tmp = document.createElement('canvas');
    idx.forEach((zi, i) => {
      const c = refs.current[i];
      if (!c) return;
      drawSlice(tmp, volume, zi, win.lo, win.hi);
      const ctx = c.getContext('2d');
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, c.width, c.height);
      const s = Math.min(c.width / tmp.width, c.height / tmp.height);
      const w = tmp.width * s;
      const h = tmp.height * s;
      ctx.drawImage(tmp, (c.width - w) / 2, (c.height - h) / 2, w, h);
    });
  }, [volume, win.lo, win.hi, idx]);

  const active = idx.reduce((best, zi, i) => (Math.abs(zi - z) < Math.abs(idx[best] - z) ? i : best), 0);

  return (
    <div className="thumbs">
      {idx.map((zi, i) => (
        <button
          key={`${zi}-${i}`}
          type="button"
          className={`thumb ${i === active ? 'is-active' : ''}`}
          onClick={() => onPick(zi)}
          aria-label={`Go to slice ${zi + 1}`}
        >
          <canvas ref={(el) => (refs.current[i] = el)} width={72} height={72} />
          <span className="mono">{zi + 1}</span>
        </button>
      ))}
    </div>
  );
}

/**
 * Axial CT viewer. `volume` / `mask` are parsed NIfTI volumes (see nifti.js).
 * With no volume it renders an honest empty state, never a stand-in image.
 */
export default function ScanViewer({ volume, mask, label, empty }) {
  const wrapRef = useRef(null);
  const stageRef = useRef(null);
  const baseRef = useRef(null);
  const maskRef = useRef(null);
  const drag = useRef(null);

  const [z, setZ] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [preset, setPreset] = useState('auto');
  const [showMask, setShowMask] = useState(true);
  const [full, setFull] = useState(false);

  useEffect(() => {
    setZ(volume ? Math.floor(volume.nz / 2) : 0);
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setPreset(volume?.isHU ? 'lung' : 'auto');
  }, [volume]);

  const win = useMemo(() => (volume ? windowFor(volume, preset) : null), [volume, preset]);
  const maskOk = !!(volume && mask && sameGrid(volume, mask));

  useEffect(() => {
    if (volume && win && baseRef.current) drawSlice(baseRef.current, volume, z, win.lo, win.hi);
  }, [volume, win, z]);

  useEffect(() => {
    const c = maskRef.current;
    if (!c || !volume) return;
    if (maskOk && showMask) drawMask(c, mask, z);
    else {
      c.width = volume.nx;
      c.height = volume.ny;
    }
  }, [volume, mask, maskOk, showMask, z]);

  useEffect(() => {
    const el = stageRef.current;
    if (!el || !volume) return undefined;
    const onWheel = (e) => {
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) setZoom((v) => clamp(v * (e.deltaY < 0 ? 1.1 : 0.9), 0.5, 8));
      else setZ((v) => clamp(v + (e.deltaY > 0 ? 1 : -1), 0, volume.nz - 1));
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [volume]);

  useEffect(() => {
    const h = () => setFull(document.fullscreenElement === wrapRef.current);
    document.addEventListener('fullscreenchange', h);
    return () => document.removeEventListener('fullscreenchange', h);
  }, []);

  const go = (d) => volume && setZ((v) => clamp(v + d, 0, volume.nz - 1));
  const zoomBy = (f) => setZoom((v) => clamp(v * f, 0.5, 8));
  const reset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };
  const toggleFull = () => {
    if (!document.fullscreenElement) wrapRef.current?.requestFullscreen?.();
    else document.exitFullscreen?.();
  };
  const onKey = (e) => {
    if (!volume) return;
    if (e.key === 'ArrowRight' || e.key === 'ArrowUp') go(1);
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') go(-1);
    else if (e.key === '+' || e.key === '=') zoomBy(1.2);
    else if (e.key === '-') zoomBy(1 / 1.2);
    else if (e.key === '0') reset();
    else return;
    e.preventDefault();
  };

  const off = !volume;
  const canFull = typeof document !== 'undefined' && document.fullscreenEnabled;

  return (
    <section className="viewer panel" ref={wrapRef} aria-label="CT volume viewer">
      <div
        ref={stageRef}
        className={`vw-stage ${off ? 'is-empty' : ''}`}
        tabIndex={off ? -1 : 0}
        onKeyDown={onKey}
        onPointerDown={(e) => {
          if (off) return;
          drag.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (drag.current) setPan({ x: e.clientX - drag.current.x, y: e.clientY - drag.current.y });
        }}
        onPointerUp={() => {
          drag.current = null;
        }}
        onDoubleClick={reset}
      >
        {off ? (
          <div className="vw-empty">
            <span className="corner tl" /><span className="corner tr" /><span className="corner bl" /><span className="corner br" />
            <ScanLine size={30} strokeWidth={1.2} />
            <h3>{empty?.title || 'No volume loaded'}</h3>
            <p>
              {empty?.text ||
                'Slices are decoded in your browser from the .nii / .nii.gz file you upload. Nothing is shown until a real volume is available.'}
            </p>
            {empty?.action}
          </div>
        ) : (
          <>
            <div className="vw-layer" style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` }}>
              <canvas ref={baseRef} className="vw-canvas" />
              <canvas ref={maskRef} className="vw-canvas vw-mask" />
            </div>
            <div className="vw-ov tl mono">
              <div>{label || 'Local volume'}</div>
              <div>{volume.nx} × {volume.ny} × {volume.nz} vox</div>
              <div>{volume.spacing.map((s) => s.toFixed(2)).join(' × ')} mm</div>
            </div>
            <div className="vw-ov tr mono">
              <div>Slice {z + 1} / {volume.nz}</div>
              <div>Axial</div>
            </div>
            <div className="vw-ov bl mono">
              <div>W {win.w} · L {win.l}</div>
              <div>Zoom {Math.round(zoom * 100)}%</div>
            </div>
            <div className="vw-ov br mono">
              {maskOk && showMask ? <div className="gold">Segmentation mask</div> : null}
              <div>Scroll: slices · Ctrl+scroll: zoom</div>
            </div>
          </>
        )}
      </div>

      {!off && <Thumbs volume={volume} win={win} z={z} onPick={setZ} />}

      <div className="vw-bar">
        <div className="vw-group vw-nav">
          <button type="button" className="icon-btn sm" onClick={() => go(-1)} disabled={off || z === 0} aria-label="Previous slice">
            <ChevronLeft size={16} />
          </button>
          <input
            type="range"
            min={0}
            max={volume ? volume.nz - 1 : 0}
            value={z}
            disabled={off}
            onChange={(e) => setZ(Number(e.target.value))}
            aria-label="Slice position"
          />
          <button type="button" className="icon-btn sm" onClick={() => go(1)} disabled={off || z === volume.nz - 1} aria-label="Next slice">
            <ChevronRight size={16} />
          </button>
          <span className="mono slice-ind">{off ? '— / —' : `${z + 1} / ${volume.nz}`}</span>
        </div>

        <div className="vw-group">
          {Object.entries(WINDOW_PRESETS).map(([k, p]) => (
            <button key={k} type="button" className={`chip ${preset === k ? 'is-on' : ''}`} onClick={() => setPreset(k)} disabled={off}>
              {p.label}
            </button>
          ))}
        </div>

        <div className="vw-group">
          <button type="button" className="icon-btn sm" onClick={() => zoomBy(1 / 1.2)} disabled={off} aria-label="Zoom out"><ZoomOut size={16} /></button>
          <button type="button" className="icon-btn sm" onClick={() => zoomBy(1.2)} disabled={off} aria-label="Zoom in"><ZoomIn size={16} /></button>
          <button type="button" className="icon-btn sm" onClick={reset} disabled={off} aria-label="Reset zoom and position"><RotateCcw size={15} /></button>
          <button
            type="button"
            className={`icon-btn sm ${maskOk && showMask ? 'is-on' : ''}`}
            onClick={() => setShowMask((v) => !v)}
            disabled={!maskOk}
            aria-pressed={maskOk && showMask}
            aria-label="Toggle segmentation overlay"
            title={maskOk ? 'Toggle segmentation overlay' : 'No segmentation mask matching this volume is available'}
          >
            <Layers size={16} />
          </button>
          {canFull && (
            <button type="button" className="icon-btn sm" onClick={toggleFull} aria-label={full ? 'Exit fullscreen' : 'Fullscreen'}>
              {full ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
