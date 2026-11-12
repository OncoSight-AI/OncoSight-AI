
// Minimal in-browser NIfTI-1 reader (.nii / .nii.gz).
// Supports displaying matching CT and segmentation planes
// along voxel axes 0, 1 and 2.

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
    throw new Error(
      'This browser cannot decompress .nii.gz files. Use a current browser.'
    );
  }

  const stream = new Blob([buf])
    .stream()
    .pipeThrough(new DecompressionStream('gzip'));

  return new Response(stream).arrayBuffer();
}

export async function readNifti(file) {
  return readNiftiBuffer(await file.arrayBuffer());
}

export async function readNiftiBuffer(input) {
  let buf = input;

  const head = new Uint8Array(
    buf,
    0,
    Math.min(2, buf.byteLength)
  );

  if (head[0] === 0x1f && head[1] === 0x8b) {
    buf = await gunzip(buf);
  }

  if (buf.byteLength < 352) {
    throw new Error('File is too small to be a NIfTI volume.');
  }

  const dv = new DataView(buf);

  let le = true;
  let hdr = dv.getInt32(0, true);

  if (hdr !== 348) {
    hdr = dv.getInt32(0, false);

    if (hdr === 348) {
      le = false;
    } else if (
      hdr === 540 ||
      dv.getInt32(0, true) === 540
    ) {
      throw new Error(
        'NIfTI-2 files cannot be previewed in the browser.'
      );
    } else {
      throw new Error(
        'Not a valid NIfTI-1 file (unexpected header).'
      );
    }
  }

  const ndim = dv.getInt16(40, le);

  if (ndim < 3) {
    throw new Error('Volume has fewer than 3 dimensions.');
  }

  const nx = dv.getInt16(42, le);
  const ny = dv.getInt16(44, le);
  const nz = dv.getInt16(46, le);

  const datatype = dv.getInt16(70, le);

  const spacing = [1, 2, 3].map(
    (i) => Math.abs(dv.getFloat32(76 + 4 * i, le)) || 1
  );

  let voxOffset = dv.getFloat32(108, le);

  if (!(voxOffset >= 352)) {
    voxOffset = 352;
  }

  let slope = dv.getFloat32(112, le);
  let inter = dv.getFloat32(116, le);

  if (!slope || !Number.isFinite(slope)) {
    slope = 1;
  }

  if (!Number.isFinite(inter)) {
    inter = 0;
  }

  const spec = TYPES[datatype];

  if (!spec) {
    throw new Error(
      `Unsupported NIfTI data type (${datatype}).`
    );
  }

  const [Arr, size] = spec;

  const n = nx * ny * nz;
  const end = voxOffset + n * size;

  if (!(n > 0) || end > buf.byteLength) {
    throw new Error(
      'Volume data is truncated or has invalid dimensions.'
    );
  }

  const raw = buf.slice(voxOffset, end);

  if (!le && size > 1) {
    const u = new Uint8Array(raw);

    for (let i = 0; i < n; i++) {
      const o = i * size;

      for (
        let a = 0, b = size - 1;
        a < b;
        a++, b--
      ) {
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

  const step = Math.max(1, Math.floor(n / 100000));
  const sample = [];

  for (let i = 0; i < n; i += step) {
    sample.push(data[i]);
  }

  sample.sort((a, b) => a - b);

  const p1 =
    sample[Math.floor(sample.length * 0.01)] ?? min;

  const p99 =
    sample[Math.floor(sample.length * 0.99)] ?? max;

  return {
    nx,
    ny,
    nz,
    spacing,
    data,
    min,
    max,
    p1,
    p99,
    datatype,
    isHU: min < -200,
  };
}

export function sameGrid(a, b) {
  return !!a &&
    !!b &&
    a.nx === b.nx &&
    a.ny === b.ny &&
    a.nz === b.nz;
}

export function sliceCountForAxis(vol, axis = 2) {
  if (!vol || axis < 0 || axis > 2) return 0;

  return [vol.nx, vol.ny, vol.nz][axis];
}

function planeDimensions(vol, axis) {
  if (axis === 0) {
    return {
      width: vol.ny,
      height: vol.nz,
    };
  }

  if (axis === 1) {
    return {
      width: vol.nx,
      height: vol.nz,
    };
  }

  return {
    width: vol.nx,
    height: vol.ny,
  };
}

// Convert displayed plane coordinates into the original
// NIfTI linear voxel index.
function voxelIndex(vol, axis, slice, u, v) {
  const { nx, ny, nz } = vol;

  let x;
  let y;
  let z;

  if (axis === 0) {
    x = slice;
    y = u;
    z = nz - 1 - v;
  } else if (axis === 1) {
    x = u;
    y = slice;
    z = nz - 1 - v;
  } else {
    x = u;
    y = ny - 1 - v;
    z = slice;
  }

  return x + nx * (y + ny * z);
}

export function windowFor(vol, preset) {
  if (preset === 'auto' || !WINDOW_PRESETS[preset]) {
    let lo = vol.p1;
    let hi = vol.p99;

    if (!(hi > lo)) {
      lo = vol.min;
      hi = vol.max > vol.min
        ? vol.max
        : vol.min + 1;
    }

    return {
      lo,
      hi,
      w: Math.round(hi - lo),
      l: Math.round((hi + lo) / 2),
    };
  }

  const { w, l } = WINDOW_PRESETS[preset];

  return {
    lo: l - w / 2,
    hi: l + w / 2,
    w,
    l,
  };
}

/**
 * Draw a CT plane using the requested voxel axis.
 * The canvas intrinsic dimensions remain the actual
 * plane dimensions, not CSS display dimensions.
 */
export function drawSlice(
  canvas,
  vol,
  slice,
  lo,
  hi,
  axis = 2
) {
  const { width, height } = planeDimensions(vol, axis);
  const count = sliceCountForAxis(vol, axis);

  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const img = ctx.createImageData(width, height);
  const px = img.data;
  const scale = 255 / (hi - lo || 1);

  if (slice < 0 || slice >= count) {
    ctx.putImageData(img, 0, 0);
    return;
  }

  for (let v = 0; v < height; v++) {
    for (let u = 0; u < width; u++) {
      const index = voxelIndex(vol, axis, slice, u, v);

      let g = (vol.data[index] - lo) * scale;
      g = Math.max(0, Math.min(255, g));

      const offset = (v * width + u) * 4;

      px[offset] = g;
      px[offset + 1] = g;
      px[offset + 2] = g;
      px[offset + 3] = 255;
    }
  }

  ctx.putImageData(img, 0, 0);
}

/**
 * Draw the actual matching segmentation plane.
 * A red square encloses all non-zero mask pixels.
 */
export function drawMask(
  canvas,
  mask,
  slice,
  axis = 2
) {
  const { width, height } = planeDimensions(mask, axis);
  const count = sliceCountForAxis(mask, axis);

  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const img = ctx.createImageData(width, height);
  const px = img.data;

  if (slice < 0 || slice >= count) {
    ctx.putImageData(img, 0, 0);
    return;
  }

  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let v = 0; v < height; v++) {
    for (let u = 0; u < width; u++) {
      const index = voxelIndex(mask, axis, slice, u, v);

      if (mask.data[index] > 0) {
        const offset = (v * width + u) * 4;

        px[offset] = 255;
        px[offset + 1] = 35;
        px[offset + 2] = 35;
        px[offset + 3] = 170;

        minX = Math.min(minX, u);
        minY = Math.min(minY, v);
        maxX = Math.max(maxX, u);
        maxY = Math.max(maxY, v);
      }
    }
  }

  ctx.putImageData(img, 0, 0);

  if (maxX < 0) return;

  const boxWidth = maxX - minX + 1;
  const boxHeight = maxY - minY + 1;
  const side = Math.max(boxWidth, boxHeight) + 12;

  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;

  const boxX = centerX - side / 2;
  const boxY = centerY - side / 2;

  ctx.save();

  ctx.strokeStyle = '#ff2020';
  ctx.lineWidth = Math.max(2, width / 250);
  ctx.shadowColor = '#ff0000';
  ctx.shadowBlur = 5;

  ctx.strokeRect(boxX, boxY, side, side);

  ctx.restore();

  const fontSize = Math.max(9, Math.min(13, width / 45));
  const label = 'POSSIBLE NODULE DETECTED';

  ctx.save();

  ctx.font = `bold ${fontSize}px Arial`;

  const paddingX = 5;
  const labelWidth =
    ctx.measureText(label).width + paddingX * 2;
  const labelHeight = fontSize + 6;

  const labelX = Math.max(
    0,
    Math.min(width - labelWidth, boxX)
  );

  let labelY = boxY - labelHeight - 4;

  if (labelY < 0) {
    labelY = Math.min(
      height - labelHeight,
      boxY + side + 4
    );
  }

  labelY = Math.max(0, labelY);

  ctx.fillStyle = 'rgba(35, 5, 5, 0.94)';
  ctx.fillRect(
    labelX,
    labelY,
    labelWidth,
    labelHeight
  );

  ctx.strokeStyle = '#ff2020';
  ctx.lineWidth = 1;

  ctx.strokeRect(
    labelX,
    labelY,
    labelWidth,
    labelHeight
  );

  ctx.fillStyle = '#ff5555';
  ctx.textBaseline = 'middle';

  ctx.fillText(
    label,
    labelX + paddingX,
    labelY + labelHeight / 2
  );

  ctx.restore();
}
