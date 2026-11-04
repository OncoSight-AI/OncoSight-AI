import { useEffect, useState } from "react";
import { getHealth } from "../services/api";

export default function useDemoMode() {
  const [demoMode, setDemoMode] = useState(true);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    let active = true;

    getHealth()
      .then((result) => {
        if (active) {
          setDemoMode(!result?.ok);
        }
      })
      .catch(() => {
        if (active) setDemoMode(true);
      })
      .finally(() => {
        if (active) setChecking(false);
      });

    return () => {
      active = false;
    };
  }, []);

  return { demoMode, checking };
}
