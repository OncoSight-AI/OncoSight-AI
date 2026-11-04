import { useRef } from "react";

import { FileUp, UploadCloud } from "lucide-react";

export default function UploadZone({ file, onFile }) {
  const input = useRef(null);

  return (
    <div
      className="upload-zone"
      onClick={() => input.current?.click()}
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        onFile(e.dataTransfer.files?.[0] || null);
      }}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && input.current?.click()}
    >
      <div className="upload-icon">
        <UploadCloud size={28} />
      </div>

      <h3>{file ? file.name : "Drop an image or scan file here"}</h3>

      <p>or click to browse · DICOM, NIfTI, PNG, JPEG · up to 100 MB</p>

      <button type="button" className="secondary">
        <FileUp size={16} />
        Choose file
      </button>

      <input
        ref={input}
        hidden
        type="file"
        accept=".dcm,.nii,.nii.gz,.png,.jpg,.jpeg"
        onChange={(e) => onFile(e.target.files?.[0] || null)}
      />
    </div>
  );
}
