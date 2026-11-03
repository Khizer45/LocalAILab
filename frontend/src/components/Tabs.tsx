import { useSearchParams } from "react-router-dom";
import { cx } from "./ui";

/** Tabs whose selection lives in the URL (?tab=…), so links and reloads keep the view. */
export function useTab<T extends string>(tabs: readonly T[], fallback: T): [T, (t: T) => void] {
  const [params, setParams] = useSearchParams();
  const raw = params.get("tab") as T | null;
  const active = raw && tabs.includes(raw) ? raw : fallback;
  const set = (t: T) => {
    const next = new URLSearchParams(params);
    next.set("tab", t);
    setParams(next, { replace: true });
  };
  return [active, set];
}

export function Tabs<T extends string>({ tabs, active, onChange, labels }: { tabs: readonly T[]; active: T; onChange: (t: T) => void; labels?: Partial<Record<T, string>> }) {
  return (
    <div role="tablist" className="mb-5 flex flex-wrap gap-1 border-b border-line">
      {tabs.map((t) => (
        <button
          key={t}
          role="tab"
          aria-selected={t === active}
          onClick={() => onChange(t)}
          className={cx("-mb-px h-9 px-3.5 text-[13.5px] border-b-2 transition-colors", t === active ? "border-amber text-ink" : "border-transparent text-muted hover:text-ink")}
        >
          {labels?.[t] ?? t}
        </button>
      ))}
    </div>
  );
}
