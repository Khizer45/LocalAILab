import { Bot, Briefcase, CalendarClock, FileText, FlaskConical, Gauge, LayoutDashboard, LogOut, MessageSquare, Settings, ShieldAlert, ShieldCheck, Trophy, UserCog, Wrench } from "lucide-react";
import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { useModels } from "../lib/queries";
import { cx } from "./ui";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/setup", label: "Setup", icon: Wrench },
  { to: "/chat", label: "Chat", icon: MessageSquare },
  { to: "/documents", label: "Documents", icon: FileText },
  { to: "/work-agent", label: "Work agent", icon: Briefcase },
  { to: "/benchmarks", label: "Benchmarks", icon: Gauge },
  { to: "/arena", label: "Model arena", icon: Trophy },
  { to: "/agents", label: "Agent arena", icon: Bot },
  { to: "/ml", label: "ML lab", icon: FlaskConical },
  { to: "/automation", label: "Automation", icon: CalendarClock },
  { to: "/admin", label: "Administration", icon: UserCog },
  { to: "/settings", label: "Settings", icon: Settings },
];

/** The data-boundary strip: tells the user at all times where their data may go. */
function BoundaryStrip() {
  const { data, isError } = useModels();
  const offline = data?.offline_only ?? true;
  const local = data?.status.local;
  const chatModels = data?.models.filter((m) => !m.is_embedding && m.allowed).length ?? 0;
  return (
    <div className="border-b border-line bg-panel/60">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1 px-5 py-2.5 text-[13px]">
        <span className={cx("inline-flex items-center gap-1.5 font-medium", offline ? "text-teal" : "text-amber")}>
          {offline ? <ShieldCheck className="size-4" aria-hidden /> : <ShieldAlert className="size-4" aria-hidden />}
          {offline ? "Data stays on this computer" : "Cloud models allowed – data can leave this computer"}
        </span>
        <span className="text-muted">
          {isError ? "Backend not reachable" : local == null ? "Checking runtime…" : local.reachable
            ? <>Ollama {local.version} at <span className="text-ink">{local.base_url}</span></>
            : <span className="text-coral">Ollama not reachable at {local.base_url}</span>}
        </span>
        {local?.reachable && <span className="text-muted"><span className="num text-ink">{chatModels}</span> chat model{chatModels === 1 ? "" : "s"} available</span>}
      </div>
      <div className={cx("h-[3px]", offline ? "bg-teal" : "bg-[repeating-linear-gradient(90deg,var(--color-amber)_0_14px,transparent_14px_22px)]")} />
    </div>
  );
}

export function Layout({ children, user, onLogout }: { children: ReactNode; user?: { username: string } | null; onLogout?: () => void }) {
  return (
    <div className="flex min-h-full flex-col md:flex-row">
      <aside className="md:w-[212px] shrink-0 border-b md:border-b-0 md:border-r border-line bg-panel">
        <div className="flex md:flex-col md:h-screen md:sticky md:top-0">
        <div className="hidden md:block px-5 pt-5 pb-6">
          <div className="font-cond text-[19px] font-semibold leading-none">Local AI Lab</div>
          <div className="mt-1 text-[12px] text-faint">Private models, measured</div>
        </div>
        <nav aria-label="Main" className="flex md:flex-col gap-0.5 px-2 py-2 md:py-0 overflow-x-auto">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                cx("flex items-center gap-2.5 rounded-md px-3 h-9 text-[13.5px] whitespace-nowrap",
                  isActive ? "bg-raised text-ink shadow-[inset_2px_0_0_var(--color-amber)]" : "text-muted hover:text-ink hover:bg-raised/60")
              }
            >
              <Icon className="size-4" aria-hidden />
              {label}
            </NavLink>
          ))}
        </nav>
        {user && (
          <div className="ml-auto md:ml-0 md:mt-auto flex items-center gap-2 px-4 py-3 md:border-t border-line text-[13px] text-muted">
            <span className="truncate">{user.username}</span>
            <button onClick={onLogout} className="ml-auto rounded p-1 hover:text-ink" title="Log out" aria-label="Log out">
              <LogOut className="size-4" />
            </button>
          </div>
        )}
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <BoundaryStrip />
        <main className="flex-1 px-5 py-6 md:px-8">{children}</main>
      </div>
    </div>
  );
}
