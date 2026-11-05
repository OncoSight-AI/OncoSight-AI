import { useCallback, useEffect, useState } from 'react';
import Topbar from './components/Topbar.jsx';
import Sidebar from './components/Sidebar.jsx';
import Dashboard from './pages/Dashboard.jsx';
import Upload from './pages/Upload.jsx';
import Results from './pages/Results.jsx';
import History from './pages/History.jsx';
import { checkHealth, fetchMask, normalizeResult } from './api.js';
import { readNiftiBuffer, sameGrid } from './nifti.js';
import { loadHistory, saveHistory } from './store.js';

export default function App() {
  const [page, setPage] = useState('dashboard');
  const [menuOpen, setMenuOpen] = useState(false);
  const [health, setHealth] = useState({ state: 'checking' });
  const [history, setHistory] = useState(loadHistory);
  const [currentId, setCurrentId] = useState(() => loadHistory()[0]?.id ?? null);
  // Volumes and masks are large, so they live in memory for this session only: { [id]: { volume, mask, maskState } }
  const [volumes, setVolumes] = useState({});

  // Backend connection indicator
  useEffect(() => {
    let alive = true;
    const run = async () => {
      const h = await checkHealth();
      if (alive) setHealth(h);
    };
    run();
    const t = setInterval(run, 15000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  useEffect(() => {
    saveHistory(history);
  }, [history]);

  useEffect(() => {
    if (!menuOpen) return undefined;
    const onKey = (e) => e.key === 'Escape' && setMenuOpen(false);
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [menuOpen]);

  const navigate = useCallback((p) => {
    setPage(p);
    setMenuOpen(false);
    window.scrollTo({ top: 0 });
  }, []);

  const patchVolumes = (id, patch) => setVolumes((v) => ({ ...v, [id]: { ...v[id], ...patch } }));

  const handleComplete = useCallback(
    ({ file, raw, volume, elapsedMs }) => {
      const record = normalizeResult(raw);
      const id = `${Date.now()}`;
      const entry = { id, savedAt: new Date().toISOString(), filename: file.name, size: file.size, elapsedMs, notes: '', record };
      setHistory((h) => [entry, ...h]);
      setCurrentId(id);
      setVolumes((v) => ({ ...v, [id]: { volume, mask: null, maskState: record.scanId != null ? 'loading' : 'none' } }));
      navigate('results');

      if (record.scanId != null) {
        fetchMask(record.scanId)
          .then(readNiftiBuffer)
          .then((mask) => {
            const ok = volume && sameGrid(volume, mask);
            patchVolumes(id, { mask: ok ? mask : null, maskState: ok ? 'ready' : 'unavailable' });
          })
          .catch(() => patchVolumes(id, { maskState: 'unavailable' }));
      }
    },
    [navigate],
  );

  const openEntry = (id) => {
    setCurrentId(id);
    navigate('results');
  };

  const deleteEntry = (id) => {
    setHistory((h) => h.filter((e) => e.id !== id));
    setVolumes((v) => {
      const { [id]: _drop, ...rest } = v;
      return rest;
    });
    if (currentId === id) setCurrentId(null);
  };

  const clearAll = () => {
    setHistory([]);
    setVolumes({});
    setCurrentId(null);
  };

  const updateNotes = (notes) => setHistory((h) => h.map((e) => (e.id === currentId ? { ...e, notes } : e)));

  const entry = history.find((e) => e.id === currentId) || null;
  const vols = entry ? volumes[entry.id] : undefined;

  return (
    <div className="app">
      <Topbar page={page} health={health} menuOpen={menuOpen} onMenu={() => setMenuOpen(true)} />
      <Sidebar open={menuOpen} page={page} health={health} onClose={() => setMenuOpen(false)} onNavigate={navigate} />

      <main>
        {page === 'dashboard' && (
          <Dashboard entry={entry} vols={vols} health={health} history={history} navigate={navigate} openEntry={openEntry} />
        )}
        {page === 'upload' && <Upload health={health} onComplete={handleComplete} />}
        {page === 'results' && <Results entry={entry} vols={vols} navigate={navigate} onNotes={updateNotes} />}
        {page === 'history' && (
          <History history={history} onOpen={openEntry} onDelete={deleteEntry} onClear={clearAll} navigate={navigate} />
        )}
      </main>

      <footer className="app-foot mono">OncoSight AI · research prototype · not for clinical use</footer>
    </div>
  );
}
