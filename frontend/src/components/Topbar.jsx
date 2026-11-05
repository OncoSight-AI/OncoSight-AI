import { MoreHorizontal } from 'lucide-react';
import { API_BASE } from '../api.js';

const PAGE_LABEL = {
  dashboard: 'Overview',
  upload: 'New Analysis',
  results: 'Analysis Report',
  history: 'Scan History',
};

const STATE_LABEL = {
  checking: 'Checking backend',
  online: 'Backend online',
  offline: 'Backend offline',
};

export default function Topbar({ page, health, menuOpen, onMenu }) {
  const host = API_BASE.replace(/^https?:\/\//, '');
  return (
    <header className="topbar">
      <div className="topbar-left">
        <button
          type="button"
          className="icon-btn"
          onClick={onMenu}
          aria-label="Open navigation menu"
          aria-expanded={menuOpen}
          aria-controls="app-sidebar"
        >
          <MoreHorizontal size={20} />
        </button>
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">O</span>
          <div>
            <div className="brand-name">OncoSight AI</div>
            <div className="mono brand-sub">Research prototype</div>
          </div>
        </div>
      </div>

      <div className="topbar-mid mono">{PAGE_LABEL[page]}</div>

      <div className={`status status-${health.state}`} role="status" aria-live="polite">
        <span className="dot" />
        <span className="status-text">{STATE_LABEL[health.state]}</span>
        <span className="mono status-meta">
          {host}
          {health.latency != null ? ` · ${health.latency} ms` : ''}
        </span>
      </div>
    </header>
  );
}
