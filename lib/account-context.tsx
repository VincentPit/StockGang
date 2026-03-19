"use client";

/**
 * AccountContext — single polling instance for account state.
 * Wrap the app with <AccountProvider> so any component can call
 * useAccountCtx() without duplicating polling logic.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { getAccount, type AccountInfo } from "@/lib/api";

interface AccountCtxValue {
  account: AccountInfo | null;
  loading: boolean;
  refresh: () => void;
}

const AccountContext = createContext<AccountCtxValue>({
  account: null,
  loading: false,
  refresh: () => {},
});

export function AccountProvider({ children }: { children: ReactNode }) {
  const [account, setAccount] = useState<AccountInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getAccount();
      setAccount(data);
    } catch {
      // keep stale data — don't wipe on transient error
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    timerRef.current = setInterval(refresh, 30_000);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [refresh]);

  return (
    <AccountContext.Provider value={{ account, loading, refresh }}>
      {children}
    </AccountContext.Provider>
  );
}

/** Hook to access shared account state from any component inside <AccountProvider>. */
export function useAccountCtx() {
  return useContext(AccountContext);
}
