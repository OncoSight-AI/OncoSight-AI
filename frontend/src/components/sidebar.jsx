import { useEffect, useRef } from 'react';
import { FileText, History, LayoutGrid, UploadCloud, X } from 'lucide-react';
import { API_BASE } from '../api.js';

const ITEMS = [
  { id: 'dashboard', label: 'Overview', hint: 'Lung Intelligence workstation', icon: LayoutGrid },
  { id: 'upload', label: 'New Analysis', hint: 'Upload a .nii / .nii.gz volume', icon: UploadCloud },
  { id: 'results', label: 'Analysis Report', hint: 'Latest backend result', icon: FileText },
  { id: 'history', label: 'Scan History', hint: 'Stored on this device', icon: History },
];

export default function Sidebar({ open, page, health, onClose, onNavigate }) {
  const closeRef = useRef(null);

  useEffect(() => {
    if (open) closeRef.current?.focus();
  }, [open]);

  return (
    <>
      <div className={`backdrop ${open ? 'is-open' : ''}`} onClick={onClose} aria-hidden="true" />
      <aside
        id="app-sidebar"
        className={`sidebar ${open ? 'is-open' : ''}`}
        aria-hidden={!open}
        aria-label="Primary navigation"
      >
        <div className="sidebar-head">
          <div className="brand">
            <span className="brand-mark" aria-hidden="true">O</span>
            <div>
              <div className="brand-name">OncoSight AI</div>
              <div className="mono brand-sub">Lung Intelligence</div>
            </div>
          </div>
          <button ref={closeRef} type="button" className="icon-btn" onClick={onClose} aria-label="Close navigation menu" tabIndex={open ? 0 : -1}>
            <X size={18} />
          </button>
        </div>

        <div className="mono eyebrow">Navigate</div>
        <nav className="sidebar-nav">
          {ITEMS.map(({ id, label, hint, icon: Icon }) => (
            <button
              key={id}
              type="button"
              className={`nav-item ${page === id ? 'is-active' : ''}`}
              onClick={() => onNavigate(id)}
              tabIndex={open ? 0 : -1}
              aria-current={page === id ? 'page' : undefined}
            >
              <Icon size={18} strokeWidth={1.6} />
              <span>
                <span className="nav-label">{label}</span>
                <span className="nav-hint">{hint}</span>
              </span>
            </button>
          ))}
        </nav>

        <div className="sidebar-foot">
          <div className={`status status-${health.state}`}>
            <span className="dot" />
            <span className="status-text">
              {health.state === 'online' ? 'Backend online' : health.state === 'checking' ? 'Checking backend' : 'Backend offline'}
            </span>
          </div>
          <div className="mono foot-line">{API_BASE}</div>
          <p className="foot-note">Research prototype. Outputs are model candidates for review and are not a clinical diagnosis.</p>
        </div>
      </aside>
    </>
  );
}
