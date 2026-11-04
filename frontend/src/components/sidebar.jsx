import { NavLink } from "react-router-dom";

import {
  Activity,
  BarChart3,
  BookOpen,
  Clock3,
  FileText,
  FolderUp,
  Gauge,
  Info,
  LayoutDashboard,
  ScanLine,
  Settings,
} from "lucide-react";

const links = [
  ["/", "Dashboard", LayoutDashboard],
  ["/upload", "Upload study", FolderUp],
  ["/viewer", "Image viewer", ScanLine],
  ["/results", "Results", Activity],
  ["/performance", "Performance", Gauge],
  ["/history", "History", Clock3],
  ["/reports", "Reports", FileText],
  ["/about", "About", Info],
  ["/settings", "Settings", Settings],
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark">
          <Activity size={20} />
        </span>

        <span>
          OncoSight
          <small>RESEARCH WORKSPACE</small>
        </span>
      </div>

      <nav>
        {links.map(([to, label, Icon]) => (
          <NavLink key={to} to={to} end={to === "/"}>
            <Icon size={18} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-foot">
        <BookOpen size={16} />
        Educational demo
      </div>
    </aside>
  );
}
