import { Bell } from "lucide-react";
import useDemoMode from "../hooks/useDemoMode";

export default function Navbar() {
  const { demoMode } = useDemoMode();

  return (
    <header className="navbar">
      <div>
        <span className="eyebrow">RESEARCH CONSOLE</span>

        <h2>Imaging workspace</h2>
      </div>

      <div className="nav-actions">
        <span className={`mode-pill ${demoMode ? "demo" : "live"}`}>
          {demoMode ? "Demo mode" : "Backend connected"}
        </span>

        <button className="icon-button" aria-label="Notifications">
          <Bell size={18} />
        </button>

        <div className="avatar">OS</div>
      </div>
    </header>
  );
}
