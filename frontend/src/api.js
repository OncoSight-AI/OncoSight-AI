// Backend: FastAPI  ->  GET /, GET /health, POST /predict-volume (multipart field "file"),
// GET /download-mask/{scan_id}
export const API_BASE = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

export class ApiError extends Error {
  constructor(message, status, body) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
  }
}

function extractDetail(body) {
  if (!body) return null;
  const d = body.detail ?? body.error ?? body.message;
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) return d.map((x) => x?.msg || JSON.stringify(x)).join('; ');
  if (d) return JSON.stringify(d);
  return null;
}

/** GET /health with a timeout. Never throws. */
export async function checkHealth(timeoutMs = 4000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  const t0 = performance.now();
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: ctrl.signal });
    const latency = Math.round(performance.now() - t0);
    let data = null;
    try {
      data = await res.json();
    } catch {
      /* health may return plain text */
    }
    return res.ok ? { state: 'online', latency, data } : { state: 'offline', latency, data, error: `HTTP ${res.status}` };
  } catch (e) {
    return { state: 'offline', error: e.name === 'AbortError' ? 'Timed out' : 'Unreachable' };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * POST /predict-volume. Uses XMLHttpRequest so upload progress is available.
 * Returns { promise, abort }.
 */
export function predictVolume(file, onProgress) {
  const xhr = new XMLHttpRequest();
  const promise = new Promise((resolve, reject) => {
    xhr.open('POST', `${API_BASE}/predict-volume`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress?.(e.loaded / e.total);
    };
    xhr.upload.onload = () => onProgress?.(1);
    xhr.onload = () => {
      let body = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* non-JSON */
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        if (body && typeof body === 'object') resolve(body);
        else reject(new ApiError('Backend returned a response that is not JSON.', xhr.status));
      } else {
        reject(new ApiError(extractDetail(body) || `Backend returned HTTP ${xhr.status}.`, xhr.status, body));
      }
    };
    xhr.onerror = () =>
      reject(new ApiError(`Cannot reach the backend at ${API_BASE}. Check that FastAPI is running and CORS allows this origin.`));
    xhr.onabort = () => {
      const e = new Error('Cancelled');
      e.name = 'AbortError';
      reject(e);
    };
    const fd = new FormData();
    fd.append('file', file);
    xhr.send(fd);
  });
  return { promise, abort: () => xhr.abort() };
}

export const maskUrl = (scanId) => `${API_BASE}/download-mask/${encodeURIComponent(scanId)}`;

/** GET /download-mask/{scan_id} as an ArrayBuffer. */
export async function fetchMask(scanId) {
  const res = await fetch(maskUrl(scanId));
  if (!res.ok) throw new ApiError(`Mask download failed (HTTP ${res.status}).`, res.status);
  return res.arrayBuffer();
}

// ---------------------------------------------------------------------------
// Response normalisation. The raw response is always kept on the record, so
// nothing the backend sends is discarded; these helpers only surface the
// fields the UI knows how to label.
// ---------------------------------------------------------------------------
const COUNT_KEYS = ['nodule_count', 'num_nodules', 'candidate_count', 'num_candidates', 'num_detections', 'detections_count', 'total_detections', 'count'];
const LIST_KEYS = ['detections', 'nodules', 'candidates', 'predictions', 'findings', 'results', 'boxes'];
const CONF_KEYS = ['confidence', 'model_confidence', 'max_confidence', 'mean_confidence', 'avg_confidence', 'score'];
const NESTED_KEYS = ['detections', 'boxes', 'nodules', 'candidates'];

const pick = (o, keys) => {
  for (const k of keys) if (o && o[k] !== undefined && o[k] !== null) return o[k];
  return undefined;
};
const isObj = (v) => v && typeof v === 'object' && !Array.isArray(v);

function flattenList(list) {
  const out = [];
  list.forEach((it, i) => {
    if (isObj(it)) {
      const nested = NESTED_KEYS.map((k) => it[k]).find(Array.isArray);
      if (nested) {
        const slice = it.slice_index ?? it.slice ?? it.z ?? i;
        nested.forEach((n) => out.push(isObj(n) ? { slice, ...n } : { slice, value: n }));
        return;
      }
      out.push(it);
    } else {
      out.push({ value: it });
    }
  });
  return out;
}

export function normalizeResult(raw) {
  const r = isObj(raw) ? raw : {};
  const listRaw = LIST_KEYS.map((k) => r[k]).find(Array.isArray);
  const detections = listRaw ? flattenList(listRaw) : null;

  let count = pick(r, COUNT_KEYS);
  if (typeof count !== 'number') count = detections ? detections.length : null;

  let confidence = pick(r, CONF_KEYS);
  let confidenceNote = null;
  if (typeof confidence !== 'number') {
    confidence = null;
    if (detections) {
      const vals = detections.map((d) => pick(d, ['confidence', 'conf', 'score'])).filter((v) => typeof v === 'number');
      if (vals.length) {
        confidence = Math.max(...vals);
        confidenceNote = 'highest candidate';
      }
    }
  }

  const modelRaw = pick(r, ['model', 'model_name', 'model_info', 'weights']);
  const slicesRaw = pick(r, ['slices_processed', 'num_slices', 'total_slices', 'slice_count', 'n_slices', 'slices']);

  return {
    raw,
    scanId: pick(r, ['scan_id', 'scanId', 'id', 'uuid']) ?? null,
    status: typeof r.status === 'string' ? r.status : 'completed',
    message: typeof r.message === 'string' ? r.message : null,
    count,
    confidence,
    confidenceNote,
    detections,
    model: modelRaw == null ? null : typeof modelRaw === 'string' ? modelRaw : JSON.stringify(modelRaw),
    slicesProcessed: Array.isArray(slicesRaw) ? slicesRaw.length : typeof slicesRaw === 'number' ? slicesRaw : null,
    maskRef: typeof pick(r, ['mask_path', 'mask_file', 'mask_url', 'mask', 'download_url']) === 'string'
      ? pick(r, ['mask_path', 'mask_file', 'mask_url', 'mask', 'download_url'])
      : null,
    patientId: pick(r, ['patient_id', 'patientId']) ?? null,
    modality: pick(r, ['modality']) ?? null,
    scanDate: pick(r, ['scan_date', 'study_date', 'acquisition_date']) ?? null,
  };
}

export function formatConfidence(v) {
  if (typeof v !== 'number') return null;
  return v <= 1 ? `${(v * 100).toFixed(1)}%` : v.toFixed(2);
}
