import { Outlet } from "react-router-dom";

import Sidebar from "../components/Sidebar";
import Navbar from "../components/Navbar";

export default function DashboardLayout() {
  return (
    <div className="app-shell">
      <Sidebar />

      <main className="main-area">
        <Navbar />

        <div className="page-content">
          <Outlet />

          <footer>
            OncoSight AI · Research and educational prototype · Not for clinical
            use
          </footer>
        </div>
      </main>
    </div>
  );
}
