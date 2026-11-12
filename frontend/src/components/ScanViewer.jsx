
import { useEffect, useMemo, useRef, useState } from 'react';

import {
  ChevronLeft,
  ChevronRight,
  Layers,
  Maximize2,
  Minimize2,
  RotateCcw,
  ScanLine,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';

import {
  WINDOW_PRESETS,
  drawMask,
  drawSlice,
  sameGrid,
  windowFor,
  sliceCountForAxis,
} from '../nifti.js';

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

const AXIS_NAMES = {
  0: 'Sagittal',
  1: 'Coronal',
  2: 'Axial',
};

function findBestDetectionSlice(detections, axis) {
  if (!Array.isArray(detections) || !detections.length) return null;

  const valid = detections.filter(
    (d) =>
      Number.isInteger(Number(d.slice_index ?? d.slice)) &&
      (d.slice_axis === undefined || Number(d.slice_axis) === axis)
  );

  if (!valid.length) return null;

  valid.sort((a, b) => {
    const ca = Number(a.confidence);
    const cb = Number(b.confidence);

    return (
      (Number.isFinite(cb) ? cb : -1) -
      (Number.isFinite(ca) ? ca : -1)
    );
  });

  return Number(valid[0].slice_index ?? valid[0].slice);
}

function findBestMaskSlice(mask, axis) {
  if (!mask?.data) return null;

  const count = sliceCountForAxis(mask, axis);
  const counts = new Uint32Array(count);

  const { nx, ny } = mask;
  const sliceSize = nx * ny;

  for (let i = 0; i < mask.data.length; i++) {
    if (mask.data[i] <= 0) continue;

    const x = i % nx;
    const y = Math.floor(i / nx) % ny;
    const z = Math.floor(i / sliceSize);

    const slice = axis === 0 ? x : axis === 1 ? y : z;
    counts[slice]++;
  }

  let bestSlice = null;
  let bestCount = 0;

  for (let i = 0; i < counts.length; i++) {
    if (counts[i] > bestCount) {
      bestCount = counts[i];
      bestSlice = i;
    }
  }

  return bestCount > 0 ? bestSlice : null;
}

function Thumbs({ volume, win, slice, axis, onPick }) {
  const refs = useRef([]);

  const count = Math.min(12, sliceCountForAxis(volume, axis));

  const idx = useMemo(() => {
    const total = sliceCountForAxis(volume, axis);

    return Array.from({ length: count }, (_, i) =>
      count === 1 ? 0 : Math.round((i * (total - 1)) / (count - 1))
    );
  }, [count, volume, axis]);

  useEffect(() => {
    const tmp = document.createElement('canvas');

    idx.forEach((sliceIndex, i) => {
      const canvas = refs.current[i];
      if (!canvas) return;

      drawSlice(tmp, volume, sliceIndex, win.lo, win.hi, axis);

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const scale = Math.min(
        canvas.width / tmp.width,
        canvas.height / tmp.height
      );

      const width = tmp.width * scale;
      const height = tmp.height * scale;

      ctx.drawImage(
        tmp,
        (canvas.width - width) / 2,
        (canvas.height - height) / 2,
        width,
        height
      );
    });
  }, [volume, win.lo, win.hi, idx, axis]);

  const active = idx.reduce(
    (best, value, i) =>
      Math.abs(value - slice) < Math.abs(idx[best] - slice) ? i : best,
    0
  );

  return (
    <div className="thumbs">
      {idx.map((sliceIndex, i) => (
        <button
          key={`${sliceIndex}-${i}`}
          type="button"
          className={`thumb ${i === active ? 'is-active' : ''}`}
          onClick={() => onPick(sliceIndex)}
          aria-label={`Go to ${AXIS_NAMES[axis]} slice ${sliceIndex + 1}`}
        >
          <canvas
            ref={(el) => {
              refs.current[i] = el;
            }}
            width={72}
            height={72}
          />
          <span className="mono">{sliceIndex + 1}</span>
        </button>
      ))}
    </div>
  );
}

export default function ScanViewer({
  volume,
  mask,
  label,
  empty,
  sliceAxis = 2,
  detectionSlices = [],
  detections = [],
}) {
  const wrapRef = useRef(null);
  const stageRef = useRef(null);
  const baseRef = useRef(null);
  const maskRef = useRef(null);
  const drag = useRef(null);

  const axis =
    Number.isInteger(sliceAxis) && sliceAxis >= 0 && sliceAxis <= 2
      ? sliceAxis
      : 2;

  const [slice, setSlice] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [preset, setPreset] = useState('auto');
  const [showMask, setShowMask] = useState(true);
  const [full, setFull] = useState(false);

  const off = !volume;

  const sliceCount = volume ? sliceCountForAxis(volume, axis) : 0;

  const maskOk = !!(volume && mask && sameGrid(volume, mask));

  const win = useMemo(
    () => (volume ? windowFor(volume, preset) : null),
    [volume, preset]
  );

  const bestDetectionSlice = useMemo(
    () => findBestDetectionSlice(detections, axis),
    [detections, axis]
  );

  const fallbackDetectionSlice = useMemo(() => {
    if (!Array.isArray(detectionSlices)) return null;

    const valid = detectionSlices
      .map(Number)
      .filter(
        (value) =>
          Number.isInteger(value) && value >= 0 && value < sliceCount
      );

    return valid.length ? valid[0] : null;
  }, [detectionSlices, sliceCount]);

  // Reset viewer when a new volume is loaded.
  useEffect(() => {
    if (!volume) {
      setSlice(0);
      return;
    }

    setSlice(Math.floor(sliceCountForAxis(volume, axis) / 2));
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setPreset(volume.isHU ? 'lung' : 'auto');
  }, [volume, axis]);

  // Automatically open the most relevant detection slice.
  useEffect(() => {
    if (!volume) return;

    let target = bestDetectionSlice;

    if (target === null) target = fallbackDetectionSlice;

    if (target === null && maskOk) {
      target = findBestMaskSlice(mask, axis);
    }

    if (target !== null && target >= 0 && target < sliceCount) {
      setSlice(target);
    }
  }, [
    volume,
    mask,
    maskOk,
    axis,
    sliceCount,
    bestDetectionSlice,
    fallbackDetectionSlice,
  ]);

  // Draw the original CT image.
  useEffect(() => {
    if (!volume || !win || !baseRef.current) return;

    drawSlice(baseRef.current, volume, slice, win.lo, win.hi, axis);
  }, [volume, win, slice, axis]);

  // Draw mask using the same intrinsic dimensions as the CT canvas.
  useEffect(() => {
    const canvas = maskRef.current;

    if (!canvas || !volume) return;

    if (maskOk && showMask) {
      drawMask(canvas, mask, slice, axis);
    } else {
      const width = baseRef.current?.width || volume.nx;
      const height = baseRef.current?.height || volume.ny;

      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext('2d');
      ctx?.clearRect(0, 0, width, height);
    }
  }, [volume, mask, maskOk, showMask, slice, axis]);

  // Wheel navigation and zoom.
  useEffect(() => {
    const el = stageRef.current;
    if (!el || !volume) return undefined;

    const onWheel = (e) => {
      e.preventDefault();

      if (e.ctrlKey || e.metaKey) {
        setZoom((value) =>
          clamp(value * (e.deltaY < 0 ? 1.1 : 0.9), 0.5, 8)
        );
      } else {
        setSlice((value) =>
          clamp(value + (e.deltaY > 0 ? 1 : -1), 0, sliceCount - 1)
        );
      }
    };

    el.addEventListener('wheel', onWheel, { passive: false });

    return () => el.removeEventListener('wheel', onWheel);
  }, [volume, sliceCount]);

  // Fullscreen state.
  useEffect(() => {
    const handler = () => {
      setFull(document.fullscreenElement === wrapRef.current);
    };

    document.addEventListener('fullscreenchange', handler);

    return () => {
      document.removeEventListener('fullscreenchange', handler);
    };
  }, []);

  const go = (direction) => {
    if (!volume) return;

    setSlice((value) =>
      clamp(value + direction, 0, sliceCount - 1)
    );
  };

  const zoomBy = (factor) => {
    setZoom((value) => clamp(value * factor, 0.5, 8));
  };

  const reset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const toggleFull = () => {
    if (!document.fullscreenElement) {
      wrapRef.current?.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
  };

  const onKey = (e) => {
    if (!volume) return;

    if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
      go(1);
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
      go(-1);
    } else if (e.key === '+' || e.key === '=') {
      zoomBy(1.2);
    } else if (e.key === '-') {
      zoomBy(1 / 1.2);
    } else if (e.key === '0') {
      reset();
    } else {
      return;
    }

    e.preventDefault();
  };

  const canFull =
    typeof document !== 'undefined' && document.fullscreenEnabled;

  return (
    <section
      className="viewer panel"
      ref={wrapRef}
      aria-label="CT volume viewer"
    >
      <div
        ref={stageRef}
        className={`vw-stage ${off ? 'is-empty' : ''}`}
        tabIndex={off ? -1 : 0}
        onKeyDown={onKey}
        onPointerDown={(e) => {
          if (off) return;

          drag.current = {
            x: e.clientX - pan.x,
            y: e.clientY - pan.y,
          };

          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (drag.current) {
            setPan({
              x: e.clientX - drag.current.x,
              y: e.clientY - drag.current.y,
            });
          }
        }}
        onPointerUp={() => {
          drag.current = null;
        }}
        onPointerCancel={() => {
          drag.current = null;
        }}
        onDoubleClick={reset}
      >
        {off ? (
          <div className="vw-empty">
            <span className="corner tl" />
            <span className="corner tr" />
            <span className="corner bl" />
            <span className="corner br" />

            <ScanLine size={30} strokeWidth={1.2} />

            <h3>{empty?.title || 'No volume loaded'}</h3>

            <p>
              {empty?.text ||
                'Upload a NIfTI CT volume to display its slices.'}
            </p>

            {empty?.action}
          </div>
        ) : (
          <>
            {/* Both canvases occupy precisely the same fitted area. */}
            <div
              className="vw-layer"
              style={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                overflow: 'hidden',
                padding: '16px',
                boxSizing: 'border-box',
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                transformOrigin: 'center center',
              }}
            >
              <div
                className="vw-image-stack"
                style={{
                  position: 'relative',
                  width: '100%',
                  height: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  minWidth: 0,
                  minHeight: 0,
                }}
              >
                <canvas
                  ref={baseRef}
                  className="vw-canvas"
                  style={{
                    display: 'block',
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain',
                    position: 'absolute',
                    inset: 0,
                  }}
                />

                <canvas
                  ref={maskRef}
                  className="vw-canvas vw-mask"
                  style={{
                    display: 'block',
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain',
                    position: 'absolute',
                    inset: 0,
                    pointerEvents: 'none',
                  }}
                />
              </div>
            </div>

            <div className="vw-ov tl mono">
              <div>{label || 'Local volume'}</div>
              <div>{volume.nx} × {volume.ny} × {volume.nz} vox</div>
              <div>
                {volume.spacing.map((s) => s.toFixed(2)).join(' × ')} mm
              </div>
            </div>

            <div className="vw-ov tr mono">
              <div>{AXIS_NAMES[axis]} plane</div>
              <div>Slice {slice + 1} / {sliceCount}</div>
            </div>

            <div className="vw-ov bl mono">
              <div>W {win.w} · L {win.l}</div>
              <div>Zoom {Math.round(zoom * 100)}%</div>
            </div>

            <div className="vw-ov br mono">
              {maskOk && showMask ? (
                <div className="gold">Segmentation mask enabled</div>
              ) : !maskOk ? (
                <div>Matching segmentation mask unavailable</div>
              ) : (
                <div>Segmentation mask hidden</div>
              )}

              <div>Scroll: slices · Ctrl+scroll: zoom</div>
            </div>
          </>
        )}
      </div>

      {!off && (
        <Thumbs
          volume={volume}
          win={win}
          slice={slice}
          axis={axis}
          onPick={setSlice}
        />
      )}

      <div className="vw-bar">
        <div className="vw-group vw-nav">
          <button
            type="button"
            className="icon-btn sm"
            onClick={() => go(-1)}
            disabled={off || slice === 0}
            aria-label="Previous slice"
          >
            <ChevronLeft size={16} />
          </button>

          <input
            type="range"
            min={0}
            max={Math.max(0, sliceCount - 1)}
            value={slice}
            disabled={off}
            onChange={(e) => setSlice(Number(e.target.value))}
            aria-label="Slice position"
          />

          <button
            type="button"
            className="icon-btn sm"
            onClick={() => go(1)}
            disabled={off || slice === sliceCount - 1}
            aria-label="Next slice"
          >
            <ChevronRight size={16} />
          </button>

          <span className="mono slice-ind">
            {off ? '— / —' : `${slice + 1} / ${sliceCount}`}
          </span>
        </div>

        <div className="vw-group">
          {Object.entries(WINDOW_PRESETS).map(([key, presetItem]) => (
            <button
              key={key}
              type="button"
              className={`chip ${preset === key ? 'is-on' : ''}`}
              onClick={() => setPreset(key)}
              disabled={off}
            >
              {presetItem.label}
            </button>
          ))}
        </div>

        <div className="vw-group">
          <button
            type="button"
            className="icon-btn sm"
            onClick={() => zoomBy(1 / 1.2)}
            disabled={off}
            aria-label="Zoom out"
          >
            <ZoomOut size={16} />
          </button>

          <button
            type="button"
            className="icon-btn sm"
            onClick={() => zoomBy(1.2)}
            disabled={off}
            aria-label="Zoom in"
          >
            <ZoomIn size={16} />
          </button>

          <button
            type="button"
            className="icon-btn sm"
            onClick={reset}
            disabled={off}
            aria-label="Reset zoom and position"
          >
            <RotateCcw size={15} />
          </button>

          <button
            type="button"
            className={`icon-btn sm ${maskOk && showMask ? 'is-on' : ''}`}
            onClick={() => setShowMask((value) => !value)}
            disabled={!maskOk}
            aria-pressed={maskOk && showMask}
            aria-label="Toggle segmentation overlay"
            title={
              maskOk
                ? 'Toggle segmentation overlay'
                : 'No segmentation mask matching this volume is available'
            }
          >
            <Layers size={16} />
          </button>

          {canFull && (
            <button
              type="button"
              className="icon-btn sm"
              onClick={toggleFull}
              aria-label={full ? 'Exit fullscreen' : 'Fullscreen'}
            >
              {full ? (
                <Minimize2 size={16} />
              ) : (
                <Maximize2 size={16} />
              )}
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
