import { useState } from "react";
import { api } from "../api/client";
import { Button, ErrorNotice, Field, Input } from "../components/ui";
import type { AuthConfig } from "../lib/queries";

export default function Login({ config, onLoggedIn }: { config: AuthConfig; onLoggedIn: (token: string) => void }) {
  const [mode, setMode] = useState<"login" | "register">(config.has_users ? "login" : "register");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === "register") await api("/api/auth/register", { method: "POST", json: { username, password } });
      const res = await api<{ token: string }>("/api/auth/login", { method: "POST", json: { username, password } });
      onLoggedIn(res.token);
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex min-h-full items-center justify-center p-6">
      <form onSubmit={submit} className="w-full max-w-sm rounded-lg border border-line bg-panel p-6 space-y-4">
        <div>
          <h1 className="font-cond text-[24px] font-semibold">Local AI Lab</h1>
          <p className="text-muted text-[13px]">
            {mode === "login" ? "Log in to see your conversations, documents and benchmark runs." :
              config.has_users ? "Create an account on this server." : "Create the first account. It becomes the administrator."}
          </p>
        </div>
        <Field label="Username" htmlFor="u"><Input id="u" autoComplete="username" value={username} onChange={(e) => setUsername(e.target.value)} required minLength={3} /></Field>
        <Field label="Password" htmlFor="p" hint={mode === "register" ? "At least 8 characters." : undefined}>
          <Input id="p" type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} value={password} onChange={(e) => setPassword(e.target.value)} required minLength={8} />
        </Field>
        <ErrorNotice error={error} />
        <Button type="submit" variant="primary" busy={busy} className="w-full">{mode === "login" ? "Log in" : "Create account"}</Button>
        {config.registration_open && (
          <button type="button" className="w-full text-[13px] text-muted hover:text-ink" onClick={() => setMode(mode === "login" ? "register" : "login")}>
            {mode === "login" ? "Need an account? Create one" : "Already have an account? Log in"}
          </button>
        )}
      </form>
    </div>
  );
}
