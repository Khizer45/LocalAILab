import { useQuery, useQueryClient } from "@tanstack/react-query";
import { lazy, Suspense, useEffect, useState } from "react";
import { Route, Routes } from "react-router-dom";
import { api, getToken, setToken } from "./api/client";
import { Layout } from "./components/Layout";
import { ErrorNotice, Spinner } from "./components/ui";
import type { AuthConfig, Me } from "./lib/queries";
import Login from "./pages/Login";

// Pages are code-split so the first load stays small.
const Overview = lazy(() => import("./pages/Overview"));
const Chat = lazy(() => import("./pages/Chat"));
const Documents = lazy(() => import("./pages/Documents"));
const Benchmarks = lazy(() => import("./pages/Benchmarks"));
const RunDetail = lazy(() => import("./pages/RunDetail"));
const MLLab = lazy(() => import("./pages/MLLab"));
const ModelArena = lazy(() => import("./pages/ModelArena"));
const ArenaRunDetail = lazy(() => import("./pages/ArenaRunDetail"));
const AgentArena = lazy(() => import("./pages/AgentArena"));
const AgentRunDetail = lazy(() => import("./pages/AgentRunDetail"));
const SettingsPage = lazy(() => import("./pages/Settings"));
const Setup = lazy(() => import("./pages/Setup"));
const Suites = lazy(() => import("./pages/Suites"));
const Automation = lazy(() => import("./pages/Automation"));
const WorkAgent = lazy(() => import("./pages/WorkAgent"));
const Admin = lazy(() => import("./pages/Admin"));

export default function App() {
  const qc = useQueryClient();
  const [token, setTok] = useState(getToken());
  const cfg = useQuery({ queryKey: ["auth-config"], queryFn: () => api<AuthConfig>("/api/auth/config") });
  const authOn = cfg.data?.auth_enabled ?? false;
  const me = useQuery({ queryKey: ["me", token], queryFn: () => api<Me>("/api/auth/me"), enabled: !!cfg.data && (!authOn || !!token), retry: false });

  useEffect(() => {
    const onAuth = () => {
      setToken(null);
      setTok(null);
    };
    window.addEventListener("auth-required", onAuth);
    return () => window.removeEventListener("auth-required", onAuth);
  }, []);

  if (cfg.isLoading) return <div className="p-8"><Spinner label="Connecting to the local backend…" /></div>;
  if (cfg.isError)
    return (
      <div className="mx-auto max-w-xl p-8">
        <ErrorNotice error={cfg.error} />
      </div>
    );
  if (authOn && (!token || me.isError)) {
    return (
      <Login
        config={cfg.data!}
        onLoggedIn={(t) => {
          setToken(t);
          setTok(t);
          qc.invalidateQueries();
        }}
      />
    );
  }

  const isAdmin = !authOn || !!me.data?.is_admin;
  const logout = async () => {
    try {
      await api("/api/auth/logout", { method: "POST" });
    } catch {
      /* already logged out */
    }
    setToken(null);
    setTok(null);
    qc.clear();
  };

  return (
    <Layout user={authOn ? me.data : null} onLogout={logout}>
      <Suspense fallback={<Spinner />}>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/chat/:id" element={<Chat />} />
        <Route path="/documents" element={<Documents />} />
        <Route path="/benchmarks" element={<Benchmarks />} />
        <Route path="/benchmarks/runs/:id" element={<RunDetail />} />
        <Route path="/ml" element={<MLLab />} />
        <Route path="/arena" element={<ModelArena />} />
        <Route path="/arena/runs/:id" element={<ArenaRunDetail />} />
        <Route path="/agents" element={<AgentArena />} />
        <Route path="/agents/runs/:id" element={<AgentRunDetail />} />
        <Route path="/settings" element={<SettingsPage isAdmin={isAdmin} />} />
        <Route path="/setup" element={<Setup isAdmin={isAdmin} />} />
        <Route path="/benchmarks/suites" element={<Suites />} />
        <Route path="/automation" element={<Automation isAdmin={isAdmin} />} />
        <Route path="/work-agent" element={<WorkAgent isAdmin={isAdmin} />} />
        <Route path="/admin" element={<Admin isAdmin={isAdmin} />} />
        <Route path="*" element={<div className="text-muted">Page not found.</div>} />
      </Routes>
      </Suspense>
    </Layout>
  );
}
