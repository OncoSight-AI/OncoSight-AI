// Minimal in-browser NIfTI-1 reader (.nii / .nii.gz). Slices are decoded from the
// file the user selected, so what is displayed is the real volume, never a stand-in.

const TYPES = {
  2: [Uint8Array, 1],
  4: [Int16Array, 2],
  8: [Int32Array, 4],
  16: [Float32Array, 4],
  64: [Float64Array, 8],
  256: [Int8Array, 1],
  512: [Uint16Array, 2],
  768: [Uint32Array, 4],
};

export const WINDOW_PRESETS = {
  lung: { label: 'Lung', w: 1500, l: -600 },
  soft: { label: 'Soft tissue', w: 400, l: 40 },
  auto: { label: 'Auto', w: 0, l: 0 },
};

async function gunzip(buf) {
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('This browser cannot decompress .nii.gz files. Use a current Chrome, Edge, Firefox or Safari.');
  }
  const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}

export async function readNifti(file) {
  return readNiftiBuffer(await file.arrayBuffer());
}

export async function readNiftiBuffer(input) {
  let buf = input;
  const head = new Uint8Array(buf, 0, Math.min(2, buf.byteLength));
  if (head[0] === 0x1f && head[1] === 0x8b) buf = await gunzip(buf);
  if (buf.byteLength < 352) throw new Error('File is too small to be a NIfTI volume.');

  const dv = new DataView(buf);
  let le = true;
  let hdr = dv.getInt32(0, true);
  if (hdr !== 348) {
    hdr = dv.getInt32(0, false);
    if (hdr === 348) le = false;
    else if (hdr === 540 || dv.getInt32(0, true) === 540) throw new Error('NIfTI-2 files cannot be previewed in the browser.');
    else throw new Error('Not a valid NIfTI-1 file (unexpected header).');
  }

  const ndim = dv.getInt16(40, le);
  if (ndim < 3) throw new Error('Volume has fewer than 3 dimensions.');
  const nx = dv.getInt16(42, le);
  const ny = dv.getInt16(44, le);
  const nz = dv.getInt16(46, le);
  const datatype = dv.getInt16(70, le);
  const spacing = [1, 2, 3].map((i) => Math.abs(dv.getFloat32(76 + 4 * i, le)) || 1);
  let voxOffset = dv.getFloat32(108, le);
  if (!(voxOffset >= 352)) voxOffset = 352;
  let slope = dv.getFloat32(112, le);
  let inter = dv.getFloat32(116, le);
  if (!slope || !Number.isFinite(slope)) slope = 1;
  if (!Number.isFinite(inter)) inter = 0;

  const spec = TYPES[datatype];
  if (!spec) throw new Error(`Unsupported NIfTI data type (${datatype}).`);
  const [Arr, size] = spec;
  const n = nx * ny * nz;
  const end = voxOffset + n * size;
  if (!(n > 0) || end > buf.byteLength) throw new Error('Volume data is truncated or has invalid dimensions.');

  const raw = buf.slice(voxOffset, end);
  if (!le && size > 1) {
    const u = new Uint8Array(raw);
    for (let i = 0; i < n; i++) {
      const o = i * size;
      for (let a = 0, b = size - 1; a < b; a++, b--) {
        const t = u[o + a];
        u[o + a] = u[o + b];
        u[o + b] = t;
      }
    }
  }
  const typed = new Arr(raw);
  const data = new Float32Array(n);
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = typed[i] * slope + inter;
    data[i] = v;
    if (v < min) min = v;
    if (v > max) max = v;
  }

  // 1st / 99th percentile from a strided sample, for the "Auto" window.
  const step = Math.max(1, Math.floor(n / 100000));
  const sample = [];
  for (let i = 0; i < n; i += step) sample.push(data[i]);
  sample.sort((a, b) => a - b);
  const p1 = sample[Math.floor(sample.length * 0.01)] ?? min;
  const p99 = sample[Math.floor(sample.length * 0.99)] ?? max;

  return { nx, ny, nz, spacing, data, min, max, p1, p99, datatype, isHU: min < -200 };
}

export function sameGrid(a, b) {
  return !!a && !!b && a.nx === b.nx && a.ny === b.ny && a.nz === b.nz;
}

export function windowFor(vol, preset) {
  if (preset === 'auto' || !WINDOW_PRESETS[preset]) {
    let lo = vol.p1;
    let hi = vol.p99;
    if (!(hi > lo)) {
      lo = vol.min;
      hi = vol.max > vol.min ? vol.max : vol.min + 1;
    }
    return { lo, hi, w: Math.round(hi - lo), l: Math.round((hi + lo) / 2) };
  }
  const { w, l } = WINDOW_PRESETS[preset];
  return { lo: l - w / 2, hi: l + w / 2, w, l };
}

/** Draw axial slice z into a canvas (resizes the canvas to nx x ny). */
export function drawSlice(canvas, vol, z, lo, hi) {
  const { nx, ny, data } = vol;
  canvas.width = nx;
  canvas.height = ny;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(nx, ny);
  const px = img.data;
  const base = z * nx * ny;
  const scale = 255 / (hi - lo || 1);
  for (let j = 0; j < ny; j++) {
    const dstRow = (ny - 1 - j) * nx; // NIfTI y runs posterior -> anterior; flip for display
    const srcRow = base + j * nx;
    for (let i = 0; i < nx; i++) {
      let g = (data[srcRow + i] - lo) * scale;
      g = g < 0 ? 0 : g > 255 ? 255 : g;
      const o = (dstRow + i) * 4;
      px[o] = px[o + 1] = px[o + 2] = g;
      px[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

/** Draw non-zero mask voxels of slice z in champagne gold. */
export function drawMask(canvas, mask, z) {
  const { nx, ny, data } = mask;
  canvas.width = nx;
  canvas.height = ny;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(nx, ny);
  const px = img.data;
  const base = z * nx * ny;
  for (let j = 0; j < ny; j++) {
    const dstRow = (ny - 1 - j) * nx;
    const srcRow = base + j * nx;
    for (let i = 0; i < nx; i++) {
      if (data[srcRow + i] > 0) {
        const o = (dstRow + i) * 4;
        px[o] = 196;
        px[o + 1] = 166;
        px[o + 2] = 106;
        px[o + 3] = 120;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}
