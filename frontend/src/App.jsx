
import { useCallback, useEffect, useState } from 'react';

import Topbar from './components/Topbar.jsx';
import Sidebar from './components/Sidebar.jsx';
import Dashboard from './pages/Dashboard.jsx';
import Upload from './pages/Upload.jsx';
import Results from './pages/Results.jsx';
import History from './pages/History.jsx';

import {
  checkHealth,
  fetchMask,
  normalizeResult
} from './api.js';

import {
  readNiftiBuffer,
  sameGrid
} from './nifti.js';

import {
  loadHistory,
  saveHistory
} from './store.js';

export default function App() {
  const [page, setPage] = useState('dashboard');
  const [menuOpen, setMenuOpen] = useState(false);

  const [health, setHealth] = useState({
    state: 'checking'
  });

  const [history, setHistory] = useState(loadHistory);

  const [currentId, setCurrentId] = useState(
    () => loadHistory()[0]?.id ?? null
  );

  // Session-only CT volumes and segmentation masks.
  const [volumes, setVolumes] = useState({});

  // Backend health monitoring.
  useEffect(() => {
    let alive = true;

    const run = async () => {
      const h = await checkHealth();

      if (alive) {
        setHealth(h);
      }
    };

    run();

    const timer = setInterval(run, 15000);

    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  // Save report history.
  useEffect(() => {
    saveHistory(history);
  }, [history]);

  // Close menu using Escape.
  useEffect(() => {
    if (!menuOpen) return undefined;

    const onKey = (e) => {
      if (e.key === 'Escape') {
        setMenuOpen(false);
      }
    };

    window.addEventListener('keydown', onKey);

    return () => {
      window.removeEventListener('keydown', onKey);
    };
  }, [menuOpen]);

  const navigate = useCallback((p) => {
    setPage(p);
    setMenuOpen(false);
    window.scrollTo({ top: 0 });
  }, []);

  const patchVolumes = useCallback((id, patch) => {
    setVolumes((previous) => ({
      ...previous,
      [id]: {
        ...previous[id],
        ...patch
      }
    }));
  }, []);

  // Handle completed CT analysis.
  const handleComplete = useCallback(
    ({ file, raw, volume, elapsedMs }) => {
      const record = normalizeResult(raw);

      const id = `${Date.now()}`;

      const entry = {
        id,
        savedAt: new Date().toISOString(),
        filename: file.name,
        size: file.size,
        elapsedMs,
        notes: '',
        record
      };

      setHistory((previous) => [
        entry,
        ...previous
      ]);

      setCurrentId(id);

      setVolumes((previous) => ({
        ...previous,
        [id]: {
          volume,
          mask: null,
          maskState: record.scanId
            ? 'loading'
            : 'unavailable',
          maskError: record.scanId
            ? null
            : 'Backend did not return a scan ID.'
        }
      }));

      navigate('results');

      // Download the actual model-generated segmentation mask.
      if (!record.scanId) {
        console.error(
          'Segmentation mask unavailable: scan ID missing.',
          raw
        );

        return;
      }

      console.log(
        'Downloading segmentation mask for scan:',
        record.scanId
      );

      fetchMask(record.scanId)
        .then(async (buffer) => {
          console.log(
            'Mask downloaded. Bytes:',
            buffer.byteLength
          );

          const mask = await readNiftiBuffer(buffer);

          console.log('Mask dimensions:', mask);

          const gridMatches = volume
            ? sameGrid(volume, mask)
            : false;

          if (!gridMatches) {
            console.error(
              'CT and segmentation mask grids do not match.'
            );

            patchVolumes(id, {
              mask: null,
              maskState: 'unavailable',
              maskError:
                'Segmentation mask dimensions or orientation do not match the CT scan.'
            });

            return;
          }

          console.log(
            'Segmentation mask loaded successfully.'
          );

          patchVolumes(id, {
            mask,
            maskState: 'ready',
            maskError: null
          });
        })
        .catch((error) => {
          console.error(
            'Segmentation mask loading failed:',
            error
          );

          patchVolumes(id, {
            mask: null,
            maskState: 'unavailable',
            maskError: error.message ||
              'Unable to download or read segmentation mask.'
          });
        });
    },
    [navigate, patchVolumes]
  );

  const openEntry = (id) => {
    setCurrentId(id);
    navigate('results');
  };

  const deleteEntry = (id) => {
    setHistory((previous) =>
      previous.filter((entry) => entry.id !== id)
    );

    setVolumes((previous) => {
      const { [id]: removed, ...remaining } = previous;
      return remaining;
    });

    if (currentId === id) {
      setCurrentId(null);
    }
  };

  const clearAll = () => {
    setHistory([]);
    setVolumes({});
    setCurrentId(null);
  };

  const updateNotes = (notes) => {
    setHistory((previous) =>
      previous.map((entry) =>
        entry.id === currentId
          ? { ...entry, notes }
          : entry
      )
    );
  };

  const entry =
    history.find((item) => item.id === currentId) || null;

  const vols = entry
    ? volumes[entry.id]
    : undefined;

  return (
    <div className="app">
      <Topbar
        page={page}
        health={health}
        menuOpen={menuOpen}
        onMenu={() => setMenuOpen(true)}
      />

      <Sidebar
        open={menuOpen}
        page={page}
        health={health}
        onClose={() => setMenuOpen(false)}
        onNavigate={navigate}
      />

      <main>
        {page === 'dashboard' && (
          <Dashboard
            entry={entry}
            vols={vols}
            health={health}
            history={history}
            navigate={navigate}
            openEntry={openEntry}
          />
        )}

        {page === 'upload' && (
          <Upload
            health={health}
            onComplete={handleComplete}
          />
        )}

        {page === 'results' && (
          <Results
            entry={entry}
            vols={vols}
            navigate={navigate}
            onNotes={updateNotes}
          />
        )}

        {page === 'history' && (
          <History
            history={history}
            onOpen={openEntry}
            onDelete={deleteEntry}
            onClear={clearAll}
            navigate={navigate}
          />
        )}
      </main>

      <footer className="app-foot mono">
        OncoSight AI · research prototype · not for clinical use
      </footer>
    </div>
  );
}
