
import {
  Activity,
  ArrowRight,
  BrainCircuit,
  Clock3,
  FileSearch,
  Plus,
  ShieldCheck,
  UploadCloud
} from 'lucide-react';

import { relativeTime } from '../store.js';

export default function Dashboard({
  entry,
  vols,
  health,
  history,
  navigate,
  openEntry
}) {
  const model =
    typeof health.data?.model === 'string'
      ? health.data.model
      : 'YOLOv8n-seg';

  const recent = history.slice(0, 5);

  const totalCandidates = history.reduce(
    (total, item) =>
      total + (Number(item.record?.count) || 0),
    0
  );

  const backendOnline = health.state === 'online';

  return (
    <div className="page home-dashboard">

      {/* Welcome section */}
      <section className="home-hero">
        <div className="home-hero-content">
          <div className="home-eyebrow">
            <Activity size={15} />
            AI-POWERED MEDICAL IMAGING
          </div>

          <h1>Welcome to OncoSight AI</h1>

          <p>
            Your intelligent workspace for lung CT analysis.
            Upload a scan, explore AI-generated findings and
            review your analysis history.
          </p>

          <button
            type="button"
            className="home-primary-btn"
            onClick={() => navigate('upload')}
          >
            <Plus size={19} />
            Start New Analysis
            <ArrowRight size={17} />
          </button>
        </div>

        <div className="home-hero-icon">
          <div className="home-icon-ring">
            <BrainCircuit size={54} strokeWidth={1.3} />
          </div>
          <span>INTELLIGENT SCAN ANALYSIS</span>
        </div>
      </section>

      {/* Status */}
      <section className="home-section">
        <div className="home-section-heading">
          <div>
            <div className="home-eyebrow">SYSTEM OVERVIEW</div>
            <h2>Your workspace</h2>
          </div>

          <span className={`home-status ${backendOnline ? 'online' : 'offline'}`}>
            <span className="home-status-dot" />
            {health.state === 'checking'
              ? 'Checking connection'
              : backendOnline
                ? 'System connected'
                : 'Backend unavailable'}
          </span>
        </div>

        <div className="home-stats">

          <div className="home-stat-card">
            <div className="home-stat-icon blue">
              <FileSearch size={21} />
            </div>
            <div className="home-stat-label">Total scans</div>
            <div className="home-stat-value">{history.length}</div>
            <div className="home-stat-note">Analyses stored on this device</div>
          </div>

          <div className="home-stat-card">
            <div className="home-stat-icon purple">
              <Activity size={21} />
            </div>
            <div className="home-stat-label">Candidates detected</div>
            <div className="home-stat-value">{totalCandidates}</div>
            <div className="home-stat-note">
              Model-generated candidates across saved scans
            </div>
          </div>

          <div className="home-stat-card">
            <div className="home-stat-icon green">
              <BrainCircuit size={21} />
            </div>
            <div className="home-stat-label">AI model</div>
            <div className="home-stat-model">{model}</div>
            <div className="home-stat-note">
              {backendOnline ? 'Model service connected' : 'Connection not confirmed'}
            </div>
          </div>

          <div className="home-stat-card">
            <div className="home-stat-icon amber">
              <Clock3 size={21} />
            </div>
            <div className="home-stat-label">Last analysis</div>
            <div className="home-stat-model">
              {history[0]
                ? relativeTime(history[0].savedAt)
                : 'Not yet'}
            </div>
            <div className="home-stat-note">
              Most recent saved scan
            </div>
          </div>

        </div>
      </section>

      {/* Main content */}
      <section className="home-bottom-grid">

        <div className="home-panel">
          <div className="home-panel-heading">
            <div>
              <div className="home-eyebrow">GET STARTED</div>
              <h2>What would you like to do?</h2>
            </div>
          </div>

          <button
            type="button"
            className="home-action-card"
            onClick={() => navigate('upload')}
          >
            <span className="home-action-icon">
              <UploadCloud size={23} />
            </span>
            <span className="home-action-text">
              <strong>Upload a CT scan</strong>
              <small>Start a new AI-assisted analysis using a NIfTI volume.</small>
            </span>
            <ArrowRight size={19} />
          </button>

          <button
            type="button"
            className="home-action-card"
            onClick={() => navigate('history')}
          >
            <span className="home-action-icon">
              <Clock3 size={23} />
            </span>
            <span className="home-action-text">
              <strong>Explore scan history</strong>
              <small>Review previously completed analyses.</small>
            </span>
            <ArrowRight size={19} />
          </button>

          <div className="home-privacy-note">
            <ShieldCheck size={19} />
            <span>
              <strong>Privacy-conscious workflow</strong>
              <small>
                Scan history is stored locally in this browser.
                CT volumes are not retained after the session.
              </small>
            </span>
          </div>
        </div>

        <div className="home-panel home-recent-panel">
          <div className="home-panel-heading">
            <div>
              <div className="home-eyebrow">YOUR ACTIVITY</div>
              <h2>Recent analyses</h2>
            </div>

            {history.length > 0 && (
              <button
                type="button"
                className="home-text-btn"
                onClick={() => navigate('history')}
              >
                View all <ArrowRight size={14} />
              </button>
            )}
          </div>

          {recent.length === 0 ? (
            <div className="home-empty">
              <div className="home-empty-icon">
                <FileSearch size={29} />
              </div>
              <strong>No analyses yet</strong>
              <p>
                Your completed scan analyses will appear here.
                Start by uploading your first CT scan.
              </p>
              <button
                type="button"
                className="home-secondary-btn"
                onClick={() => navigate('upload')}
              >
                Upload first scan
              </button>
            </div>
          ) : (
            <div className="home-recent-list">
              {recent.map((item) => (
                <button
                  type="button"
                  className="home-recent-item"
                  key={item.id}
                  onClick={() => openEntry(item.id)}
                >
                  <span className="home-recent-file">
                    <FileSearch size={19} />
                  </span>

                  <span className="home-recent-info">
                    <strong>{item.filename}</strong>
                    <small>
                      {relativeTime(item.savedAt)}
                      {' · '}
                      {item.record?.count ?? '—'} candidates
                    </small>
                  </span>

                  <ArrowRight size={17} />
                </button>
              ))}
            </div>
          )}
        </div>

      </section>

      {/* Safety note */}
      <div className="home-disclaimer">
        <ShieldCheck size={17} />
        <p>
          <strong>Research prototype:</strong> OncoSight AI identifies
          possible lung nodule candidates. It does not diagnose cancer.
          A result with no detected candidate does not rule out disease.
          All findings require review by a qualified medical professional.
        </p>
      </div>

    </div>
  );
}
