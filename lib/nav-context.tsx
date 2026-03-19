"use client";

/**
 * NavContext — global navigation state.
 * Any component can call useNav() to:
 *   - Read the active tab + the mounted-tab set
 *   - Switch tabs programmatically (e.g. "Research this stock")
 *   - Pass a symbol to the target panel via jumpSymbol
 */

import {
  createContext,
  useCallback,
  useContext,
  useState,
  type ReactNode,
} from "react";

export type Tab = "screener" | "advisor" | "research" | "backtest" | "workflow";

interface NavCtxValue {
  activeTab:        Tab;
  mounted:          Set<Tab>;
  jumpSymbol:       string | null;
  switchTab:        (tab: Tab) => void;
  /** Navigate to tab and optionally pre-fill a symbol in the target panel */
  jumpTo:           (tab: Tab, symbol?: string) => void;
  clearJumpSymbol:  () => void;
}

const NavContext = createContext<NavCtxValue>({
  activeTab:       "screener",
  mounted:         new Set(["screener"]),
  jumpSymbol:      null,
  switchTab:       () => {},
  jumpTo:          () => {},
  clearJumpSymbol: () => {},
});

export function NavProvider({ children }: { children: ReactNode }) {
  const [activeTab,   setActiveTab]   = useState<Tab>("screener");
  const [mounted,     setMounted]     = useState<Set<Tab>>(new Set(["screener"] as Tab[]));
  const [jumpSymbol,  setJumpSymbol]  = useState<string | null>(null);

  const switchTab = useCallback((tab: Tab) => {
    setActiveTab(tab);
    setMounted((prev) => {
      if (prev.has(tab)) return prev;
      const next = new Set(prev);
      next.add(tab);
      return next;
    });
  }, []);

  const jumpTo = useCallback(
    (tab: Tab, symbol?: string) => {
      if (symbol) setJumpSymbol(symbol.trim().toLowerCase());
      switchTab(tab);
    },
    [switchTab],
  );

  const clearJumpSymbol = useCallback(() => setJumpSymbol(null), []);

  return (
    <NavContext.Provider
      value={{ activeTab, mounted, jumpSymbol, switchTab, jumpTo, clearJumpSymbol }}
    >
      {children}
    </NavContext.Provider>
  );
}

export function useNav() {
  return useContext(NavContext);
}
