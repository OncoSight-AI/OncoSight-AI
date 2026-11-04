const ALLOWED = [".dcm", ".nii", ".nii.gz", ".png", ".jpg", ".jpeg"];

export function validateFile(file) {
  if (!file) {
    return "Choose a file first.";
  }

  const name = file.name.toLowerCase();

  if (!ALLOWED.some((ext) => name.endsWith(ext))) {
    return "Unsupported file. Use DICOM, NIfTI, PNG or JPEG.";
  }

  if (file.size > 100 * 1024 * 1024) {
    return "File is larger than the 100 MB demo limit.";
  }

  return "";
}
