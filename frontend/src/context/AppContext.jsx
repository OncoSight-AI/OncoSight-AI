import { createContext, useContext, useMemo, useState } from "react";

import { demoStudies } from "../data/demoData";

const AppContext = createContext(null);

export function AppProvider({ children }) {
  const [studies, setStudies] = useState(demoStudies);

  const [selectedStudy, setSelectedStudy] = useState(demoStudies[0]);

  const [notice, setNotice] = useState("");

  const value = useMemo(
    () => ({
      studies,
      setStudies,
      selectedStudy,
      setSelectedStudy,
      notice,
      setNotice,
    }),
    [studies, selectedStudy, notice],
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const value = useContext(AppContext);

  if (!value) {
    throw new Error("useApp must be used inside AppProvider");
  }

  return value;
}
