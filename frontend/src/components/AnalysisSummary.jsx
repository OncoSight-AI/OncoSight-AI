
import { formatConfidence } from '../api.js';

import {
  ShieldCheck,
  AlertTriangle,
  Info,
  Activity,
  Scan,
  Layers,
} from 'lucide-react';

export default function AnalysisSummary({
  record,
  status = 'idle',
  elapsedMs,
  maskState = 'none',
}) {
  const count = Number.isFinite(record?.count)
    ? record.count
    : null;

  const detected = count !== null && count > 0;
  const reassuring = count === 0;
  const done = status === 'complete';

  const verdict = detected
    ? 'POSSIBLE NODULE DETECTED'
    : reassuring
      ? 'REASSURING RESULT'
      : 'RESULT UNAVAILABLE';

  const conclusion = detected
    ? `The AI identified ${count} possible lung nodule candidate${count === 1 ? '' : 's'}. The highlighted area${count === 1 ? '' : 's'} should be reviewed by a qualified radiologist. This does not mean you have cancer.`
    : reassuring
      ? 'The AI did not identify any possible lung nodule candidates in the analysed scan. This is an encouraging result, but it cannot rule out disease. A radiologist should still review your scan.'
      : 'The AI could not provide a complete result. Please check the scan analysis and try again.';

  const confidence = detected
    ? formatConfidence(record?.confidence)
    : null;

  const highestConfidence = detected
    ? formatConfidence(record?.highestConfidence)
    : null;

  const segmentation = {
    ready: 'Overlay loaded',
    loading: 'Loading scan markings...',
    unavailable: 'Markings unavailable',
  }[maskState] || (
      record?.scanId ? 'Mask available' : 'Not available'
    );

  return (
    <aside className="panel summary" aria-label="AI analysis report">

      <div className="panel-head">
        <div>
          <div className="mono eyebrow">OncoSight AI</div>
          <h2>Your Scan Results</h2>
        </div>

        <span className={`pill ${done ? 'ok' : ''}`}>
          <span className="dot" />
          {done ? 'Complete' : status}
        </span>
      </div>

      {/* MAIN VERDICT */}

      <div
        className={`verdict-card ${detected
            ? 'verdict-danger'
            : reassuring
              ? 'verdict-safe'
              : 'verdict-unknown'
          }`}
      >
        <div className="verdict-icon">
          {detected ? (
            <AlertTriangle size={30} />
          ) : reassuring ? (
            <ShieldCheck size={30} />
          ) : (
            <Info size={30} />
          )}
        </div>

        <div className="mono verdict-label">
          YOUR AI RESULT
        </div>

        <h2>{verdict}</h2>

        <p>{conclusion}</p>

        <div className="verdict-bottom">
          {detected
            ? 'RADIOLOGIST REVIEW RECOMMENDED'
            : reassuring
              ? 'NO CANDIDATE IDENTIFIED BY MODEL'
              : 'FURTHER REVIEW REQUIRED'}
        </div>
      </div>

      {/* SIMPLE CONFIDENCE EXPLANATION */}

      <div className="summary-section">
        <div className="mono eyebrow">
          UNDERSTANDING YOUR RESULT
        </div>

        <h3>How sure is the AI about its findings?</h3>

        {detected && confidence ? (
          <>
            <div className="confidence-display">
              <span className="confidence-value">
                {confidence}
              </span>
              <span className="confidence-caption">
                Average AI detection confidence
              </span>
            </div>

            <p>
              This tells us how strongly the AI's model
              recognised the detected areas as lung-nodule-like
              patterns. It is not the probability of cancer.
            </p>

            {highestConfidence && (
              <div className="kv">
                <dt className="mono">
                  Highest individual detection score
                </dt>
                <dd>{highestConfidence}</dd>
              </div>
            )}
          </>
        ) : detected ? (
          <p>
            The AI identified possible candidates, but a
            confidence score was not available for this scan.
          </p>
        ) : reassuring ? (
          <p>
            No candidate was detected, so there is no
            candidate confidence score to display. This
            result does not guarantee that the lungs are
            free from disease.
          </p>
        ) : (
          <p>
            Confidence information is unavailable because
            the analysis result is incomplete.
          </p>
        )}
      </div>

      {/* SCAN STATISTICS */}

      <div className="summary-stats">

        <div className="summary-stat">
          <Scan size={19} />
          <span className="stat-number">
            {count ?? '—'}
          </span>
          <span className="stat-label">
            Possible candidates
          </span>
        </div>

        <div className="summary-stat">
          <Layers size={19} />
          <span className="stat-number">
            {record?.slicesProcessed ?? '—'}
          </span>
          <span className="stat-label">
            Images analysed
          </span>
        </div>

        <div className="summary-stat">
          <Activity size={19} />
          <span className="stat-number">
            {record?.slicesWithSegmentation ?? '—'}
          </span>
          <span className="stat-label">
            Images with markings
          </span>
        </div>

      </div>

      {/* WHAT TO DO NEXT */}

      <div className="summary-section">
        <div className="mono eyebrow">
          WHAT SHOULD YOU DO NEXT?
        </div>

        {detected ? (
          <p>
            Please arrange a review with a qualified
            radiologist. They can examine the highlighted
            areas and determine whether further tests are
            needed.
          </p>
        ) : reassuring ? (
          <p>
            This is an encouraging AI screening result.
            Continue with appropriate medical follow-up
            and have a radiologist review the scan.
          </p>
        ) : (
          <p>
            Check that your scan was uploaded correctly
            and repeat the analysis if appropriate.
          </p>
        )}
      </div>

      {/* TECHNICAL DETAILS */}

      <details className="summary-section">
        <summary>
          Technical details
        </summary>

        <dl className="kv-list">

          <div className="kv">
            <dt className="mono">Segmentation</dt>
            <dd>{segmentation}</dd>
          </div>

          <div className="kv">
            <dt className="mono">Segmented voxels</dt>
            <dd>
              {record?.segmentedVoxels?.toLocaleString() ?? '—'}
            </dd>
          </div>

          <div className="kv">
            <dt className="mono">Detection class</dt>
            <dd>
              {record?.detectionClass || 'lung_nodule'}
            </dd>
          </div>

          <div className="kv">
            <dt className="mono">Model</dt>
            <dd>{record?.model || 'YOLOv8n-seg'}</dd>
          </div>

          <div className="kv">
            <dt className="mono">Analysis time</dt>
            <dd>
              {elapsedMs != null
                ? `${(elapsedMs / 1000).toFixed(1)} sec`
                : '—'}
            </dd>
          </div>

        </dl>
      </details>

      <div className="disclaimer">
        <strong>Important medical information:</strong>{' '}
        OncoSight AI is an experimental research prototype.
        It identifies possible lung nodule patterns, not
        cancer. A detected candidate does not confirm cancer,
        and no detected candidate does not rule out disease.
        Always consult a qualified radiologist.
      </div>

    </aside>
  );
}
