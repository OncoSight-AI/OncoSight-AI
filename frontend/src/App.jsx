import { Navigate, Route, Routes } from "react-router-dom";

import DashboardLayout from "./layouts/DashboardLayout";
import Dashboard from "./pages/Dashboard";
import Upload from "./pages/Upload";
import CTViewerPage from "./pages/CTViewerPage";
import Results from "./pages/Results";
import Performance from "./pages/Performance";
import History from "./pages/History";
import Reports from "./pages/Reports";
import About from "./pages/About";
import Settings from "./pages/Settings";

export default function App() {
  return (
    <Routes>
      <Route element={<DashboardLayout />}>
        <Route index element={<Dashboard />} />
        <Route path="upload" element={<Upload />} />
        <Route path="viewer" element={<CTViewerPage />} />
        <Route path="results" element={<Results />} />
        <Route path="performance" element={<Performance />} />
        <Route path="history" element={<History />} />
        <Route path="reports" element={<Reports />} />
        <Route path="about" element={<About />} />
        <Route path="settings" element={<Settings />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
