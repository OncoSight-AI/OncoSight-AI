
// Backend: FastAPI
// GET /, GET /health
// POST /predict-volume (multipart field "file")
// GET /download-mask/{scan_id}

export const API_BASE = (
  import.meta.env.VITE_API_URL || 'http://127.0.0.1:8001'
).replace(/\/$/, '');

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

  if (Array.isArray(d)) {
    return d.map((x) => x?.msg || JSON.stringify(x)).join('; ');
  }

  if (d) return JSON.stringify(d);

  return null;
}

export async function checkHealth(timeoutMs = 4000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  const t0 = performance.now();

  try {
    const res = await fetch(`${API_BASE}/health`, {
      signal: ctrl.signal,
    });

    const latency = Math.round(performance.now() - t0);

    let data = null;

    try {
      data = await res.json();
    } catch {
      // Health endpoint may return plain text.
    }

    return res.ok
      ? { state: 'online', latency, data }
      : {
        state: 'offline',
        latency,
        data,
        error: `HTTP ${res.status}`,
      };
  } catch (e) {
    return {
      state: 'offline',
      error: e.name === 'AbortError' ? 'Timed out' : 'Unreachable',
    };
  } finally {
    clearTimeout(timer);
  }
}

export function predictVolume(file, onProgress) {
  const xhr = new XMLHttpRequest();

  const promise = new Promise((resolve, reject) => {
    xhr.open('POST', `${API_BASE}/predict-volume`);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        onProgress?.(e.loaded / e.total);
      }
    };

    xhr.upload.onload = () => onProgress?.(1);

    xhr.onload = () => {
      let body = null;

      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        // Non-JSON response.
      }

      if (xhr.status >= 200 && xhr.status < 300) {
        if (body && typeof body === 'object') {
          resolve(body);
        } else {
          reject(
            new ApiError(
              'Backend returned a response that is not JSON.',
              xhr.status
            )
          );
        }
      } else {
        reject(
          new ApiError(
            extractDetail(body) ||
            `Backend returned HTTP ${xhr.status}.`,
            xhr.status,
            body
          )
        );
      }
    };

    xhr.onerror = () => {
      reject(
        new ApiError(
          `Cannot reach the backend at ${API_BASE}. Check FastAPI and CORS.`
        )
      );
    };

    xhr.onabort = () => {
      const e = new Error('Cancelled');
      e.name = 'AbortError';
      reject(e);
    };

    const fd = new FormData();
    fd.append('file', file);

    xhr.send(fd);
  });

  return {
    promise,
    abort: () => xhr.abort(),
  };
}

export const maskUrl = (scanId) =>
  `${API_BASE}/download-mask/${encodeURIComponent(scanId)}`;

export async function fetchMask(scanId) {
  const res = await fetch(maskUrl(scanId));

  if (!res.ok) {
    throw new ApiError(
      `Mask download failed (HTTP ${res.status}).`,
      res.status
    );
  }

  return res.arrayBuffer();
}

// ------------------------------------------------------
// Response normalization
// ------------------------------------------------------

const COUNT_KEYS = [
  'nodule_count',
  'num_nodules',
  'candidate_count',
  'num_candidates',
  'num_detections',
  'detections_count',
  'total_detections',
  'count',
];

const LIST_KEYS = [
  'detections',
  'nodules',
  'candidates',
  'predictions',
  'findings',
  'results',
  'boxes',
];

const NESTED_KEYS = [
  'detections',
  'boxes',
  'nodules',
  'candidates',
];

const pick = (obj, keys) => {
  for (const key of keys) {
    if (
      obj &&
      obj[key] !== undefined &&
      obj[key] !== null
    ) {
      return obj[key];
    }
  }

  return undefined;
};

const isObj = (value) =>
  value !== null &&
  typeof value === 'object' &&
  !Array.isArray(value);

function flattenList(list) {
  const out = [];

  list.forEach((item, index) => {
    if (isObj(item)) {
      const nested = NESTED_KEYS
        .map((key) => item[key])
        .find(Array.isArray);

      if (nested) {
        const slice =
          item.slice_index ??
          item.slice ??
          item.z ??
          index;

        nested.forEach((candidate) => {
          out.push(
            isObj(candidate)
              ? { slice, ...candidate }
              : { slice, value: candidate }
          );
        });

        return;
      }

      out.push(item);
    } else {
      out.push({ value: item });
    }
  });

  return out;
}

export function normalizeResult(raw) {
  const r = isObj(raw) ? raw : {};

  const payload = isObj(r.prediction)
    ? r.prediction
    : isObj(r.result)
      ? r.result
      : r;

  const listRaw = LIST_KEYS
    .map((key) => payload[key])
    .find(Array.isArray);

  const detections = listRaw
    ? flattenList(listRaw)
    : null;

  let count = pick(payload, COUNT_KEYS);

  if (typeof count !== 'number') {
    count = detections ? detections.length : null;
  }

  // Read actual average confidence from the backend.
  const averagePercent = pick(payload, [
    'average_detection_confidence_percent',
  ]);

  const averageRaw = pick(payload, [
    'average_detection_confidence',
    'average_confidence',
    'mean_confidence',
  ]);

  const highestPercent = pick(payload, [
    'highest_detection_confidence_percent',
  ]);

  const highestRaw = pick(payload, [
    'highest_detection_confidence',
    'highest_confidence',
    'max_confidence',
  ]);

  let confidence = null;
  let confidenceNote = null;

  if (typeof averagePercent === 'number') {
    confidence = averagePercent / 100;
    confidenceNote = 'average candidate confidence';
  } else if (typeof averageRaw === 'number') {
    confidence = averageRaw;
    confidenceNote = 'average candidate confidence';
  } else if (detections) {
    const values = detections
      .map((d) => pick(d, ['confidence', 'conf', 'score']))
      .filter((v) => typeof v === 'number');

    if (values.length) {
      confidence =
        values.reduce((sum, value) => sum + value, 0) /
        values.length;

      confidenceNote = 'average candidate confidence';
    }
  }

  let highestConfidence = null;

  if (typeof highestPercent === 'number') {
    highestConfidence = highestPercent / 100;
  } else if (typeof highestRaw === 'number') {
    highestConfidence = highestRaw;
  }

  const modelRaw = pick(payload, [
    'model',
    'model_name',
    'model_info',
    'weights',
  ]);

  const slicesRaw = pick(payload, [
    'slices_processed',
    'num_slices',
    'total_slices',
    'slice_count',
    'n_slices',
    'slices',
  ]);

  const scanId =
    pick(payload, ['scan_id', 'scanId', 'uuid']) ??
    pick(r, ['scan_id', 'scanId', 'uuid']) ??
    null;

  const maskRaw = pick(payload, [
    'mask_path',
    'mask_file',
    'mask_url',
    'mask',
    'download_url',
  ]);

  return {
    raw,
    scanId,

    status:
      typeof payload.status === 'string'
        ? payload.status
        : 'completed',

    message:
      typeof payload.message === 'string'
        ? payload.message
        : typeof r.message === 'string'
          ? r.message
          : null,

    count,
    confidence,
    confidenceNote,
    highestConfidence,

    confidenceAvailable:
      payload.confidence_available ??
      (confidence !== null),

    detections,

    slicesProcessed:
      Array.isArray(slicesRaw)
        ? slicesRaw.length
        : typeof slicesRaw === 'number'
          ? slicesRaw
          : null,

    slicesWithSegmentation:
      pick(payload, ['slices_with_segmentation']) ?? null,

    segmentedVoxels:
      pick(payload, ['segmented_voxels']) ?? null,

    detectionClass:
      pick(payload, ['detection_class']) ?? 'lung_nodule',

    model:
      modelRaw == null
        ? null
        : typeof modelRaw === 'string'
          ? modelRaw
          : JSON.stringify(modelRaw),

    maskRef:
      typeof maskRaw === 'string'
        ? maskRaw
        : null,

    patientId:
      pick(payload, ['patient_id', 'patientId']) ?? null,

    modality:
      pick(payload, ['modality']) ?? null,

    scanDate:
      pick(payload, [
        'scan_date',
        'study_date',
        'acquisition_date',
      ]) ?? null,
  };
}

export function formatConfidence(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return null;
  }

  return `${(value * 100).toFixed(1)}%`;
}
