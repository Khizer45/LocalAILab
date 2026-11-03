import { AlertTriangle, Info, Loader2, XCircle } from "lucide-react";
import { cloneElement, isValidElement, useId } from "react";
import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactElement, ReactNode, SelectHTMLAttributes, TextareaHTMLAttributes } from "react";
import { ApiError } from "../api/client";

export function cx(...parts: (string | false | null | undefined)[]) {
  return parts.filter(Boolean).join(" ");
}

type Variant = "primary" | "secondary" | "ghost" | "danger";

export function Button({ variant = "secondary", busy, className, children, ...rest }:
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; busy?: boolean }) {
  const styles: Record<Variant, string> = {
    primary: "bg-amber text-amber-ink hover:bg-[#f6c868] font-medium",
    secondary: "bg-raised text-ink border border-line hover:border-line-strong",
    ghost: "text-muted hover:text-ink hover:bg-raised",
    danger: "text-coral border border-line hover:border-coral/60 hover:bg-coral/10",
  };
  return (
    <button
      {...rest}
      disabled={rest.disabled || busy}
      className={cx(
        "inline-flex items-center justify-center gap-1.5 rounded-md px-3 h-8 text-[13px] transition-colors",
        "disabled:opacity-50 disabled:cursor-not-allowed",
        styles[variant],
        className,
      )}
    >
      {busy && <Loader2 className="size-3.5 animate-spin" aria-hidden />}
      {children}
    </button>
  );
}

export function Panel({ title, action, children, className, bodyClass }:
  { title?: ReactNode; action?: ReactNode; children: ReactNode; className?: string; bodyClass?: string }) {
  return (
    <section className={cx("rounded-lg border border-line bg-panel", className)}>
      {(title || action) && (
        <header className="flex items-center justify-between gap-3 border-b border-line px-4 h-11">
          <h2 className="font-cond text-[15px] font-medium text-ink">{title}</h2>
          {action}
        </header>
      )}
      <div className={cx("p-4", bodyClass)}>{children}</div>
    </section>
  );
}

export function PageHeader({ title, description, actions }: { title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="font-cond text-[26px] leading-tight font-semibold tracking-tight">{title}</h1>
        {description && <p className="mt-1 max-w-[70ch] text-muted">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
    </div>
  );
}

type Tone = "neutral" | "teal" | "amber" | "coral" | "violet" | "sky";
const toneClass: Record<Tone, string> = {
  neutral: "text-muted border-line",
  teal: "text-teal border-teal/40 bg-teal/10",
  amber: "text-amber border-amber/40 bg-amber/10",
  coral: "text-coral border-coral/40 bg-coral/10",
  violet: "text-violet border-violet/40 bg-violet/10",
  sky: "text-sky border-sky/40 bg-sky/10",
};

export function Badge({ tone = "neutral", children, title }: { tone?: Tone; children: ReactNode; title?: string }) {
  return (
    <span title={title} className={cx("inline-flex items-center gap-1 rounded-full border px-2 py-px text-[12px] whitespace-nowrap", toneClass[tone])}>
      {children}
    </span>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const map: Record<string, Tone> = {
    ready: "teal", completed: "teal", ok: "teal", complete: "teal",
    running: "amber", processing: "amber", queued: "sky", pending: "sky", partial: "amber",
    failed: "coral", error: "coral", timeout: "coral", cancelled: "neutral",
  };
  return <Badge tone={map[status] ?? "neutral"}>{status}</Badge>;
}

export function Meter({ label, value, max, unit, detail }: { label: string; value: number | null; max: number; unit?: string; detail?: string }) {
  const ratio = value == null || !max ? 0 : Math.min(1, value / max);
  const color = ratio > 0.9 ? "bg-coral" : ratio > 0.7 ? "bg-amber" : "bg-teal";
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2 text-[13px]">
        <span className="text-muted">{label}</span>
        <span className="num text-ink">{value == null ? "–" : `${value.toLocaleString()}${unit ?? ""}`}<span className="text-faint"> / {max.toLocaleString()}{unit}</span></span>
      </div>
      <div className="mt-1.5 h-1.5 rounded-full bg-raised overflow-hidden" role="meter" aria-valuenow={value ?? 0} aria-valuemax={max} aria-label={label}>
        <div className={cx("h-full rounded-full", color)} style={{ width: `${ratio * 100}%` }} />
      </div>
      {detail && <div className="mt-1 text-[12px] text-faint">{detail}</div>}
    </div>
  );
}

export function Readout({ label, value, sub, tone }: { label: string; value: ReactNode; sub?: ReactNode; tone?: "amber" | "teal" | "coral" }) {
  const color = tone === "amber" ? "text-amber" : tone === "teal" ? "text-teal" : tone === "coral" ? "text-coral" : "text-ink";
  return (
    <div className="min-w-0">
      <div className="text-[12.5px] text-muted">{label}</div>
      <div className={cx("num font-cond text-[28px] leading-tight font-semibold truncate", color)}>{value}</div>
      {sub && <div className="text-[12px] text-faint truncate">{sub}</div>}
    </div>
  );
}

export function Notice({ tone = "info", title, children }: { tone?: "info" | "warn" | "error"; title?: ReactNode; children?: ReactNode }) {
  const Icon = tone === "error" ? XCircle : tone === "warn" ? AlertTriangle : Info;
  const color = tone === "error" ? "border-coral/40 bg-coral/8 text-coral" : tone === "warn" ? "border-amber/40 bg-amber/8 text-amber" : "border-sky/30 bg-sky/8 text-sky";
  return (
    <div className={cx("flex gap-2.5 rounded-md border px-3 py-2.5 text-[13px]", color)} role={tone === "error" ? "alert" : "status"}>
      <Icon className="mt-0.5 size-4 shrink-0" aria-hidden />
      <div className="min-w-0 text-ink">
        {title && <div className="font-medium">{title}</div>}
        {children && <div className="text-muted [&_code]:text-ink">{children}</div>}
      </div>
    </div>
  );
}

export function ErrorNotice({ error }: { error: unknown }) {
  if (!error) return null;
  const e = error as ApiError;
  return (
    <Notice tone="error" title={e.message ?? String(error)}>
      {e.hint}
    </Notice>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-muted text-[13px]">
      <Loader2 className="size-4 animate-spin" aria-hidden /> {label ?? "Loading…"}
    </div>
  );
}

export function Empty({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="rounded-md border border-dashed border-line px-5 py-8 text-center">
      <div className="font-cond text-[16px] font-medium">{title}</div>
      {children && <div className="mx-auto mt-1 max-w-[52ch] text-muted text-[13px]">{children}</div>}
      {action && <div className="mt-4 flex justify-center">{action}</div>}
    </div>
  );
}

/** A labelled form control. The label and hint are tied to the control (for screen readers and tests) automatically. */
export function Field({ label, hint, children, htmlFor }: { label: string; hint?: ReactNode; children: ReactNode; htmlFor?: string }) {
  const auto = useId();
  const single = isValidElement(children) ? (children as ReactElement<{ id?: string; "aria-describedby"?: string }>) : null;
  const id = htmlFor ?? single?.props.id ?? auto;
  const hintId = `${id}-hint`;
  const control = single && !single.props.id && !htmlFor
    ? cloneElement(single, { id, ...(hint ? { "aria-describedby": hintId } : {}) })
    : children;
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={single || htmlFor ? id : undefined} className="text-[12.5px] text-muted">{label}</label>
      {control}
      {hint && <div id={hintId} className="text-[12px] text-faint">{hint}</div>}
    </div>
  );
}

const control = "rounded-md border border-line bg-bg px-2.5 h-8 text-[13px] text-ink placeholder:text-faint focus:border-amber/60 focus:outline-none";

// Controls are full width unless the caller passes its own width class.
const width = (cls?: string) => (cls && /(^|\s)w-/.test(cls) ? "" : "w-full");

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={cx(control, width(props.className), props.className)} />;
}

export function Select(props: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select {...props} className={cx(control, "pr-7", width(props.className), props.className)} />;
}

export function Textarea(props: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea {...props} className={cx(control, "h-auto py-2 leading-relaxed", width(props.className), props.className)} />;
}

export function Toggle({ checked, onChange, label, description, disabled }:
  { checked: boolean; onChange: (v: boolean) => void; label: ReactNode; description?: ReactNode; disabled?: boolean }) {
  return (
    <label className={cx("flex items-start gap-3 cursor-pointer", disabled && "opacity-50 cursor-not-allowed")}>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={cx("mt-0.5 relative h-5 w-9 shrink-0 rounded-full border transition-colors",
          checked ? "bg-teal/80 border-teal" : "bg-raised border-line-strong")}
      >
        <span className={cx("absolute left-0.5 top-0.5 block size-3.5 rounded-full bg-ink transition-transform", checked ? "translate-x-4" : "translate-x-0")} />
      </button>
      <span>
        <span className="text-[13.5px] text-ink">{label}</span>
        {description && <span className="block text-[12.5px] text-muted">{description}</span>}
      </span>
    </label>
  );
}

export function Table({ children }: { children: ReactNode }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[13px] [&_th]:text-left [&_th]:font-normal [&_th]:text-muted [&_th]:px-3 [&_th]:py-2 [&_th]:border-b [&_th]:border-line [&_th]:whitespace-nowrap [&_td]:px-3 [&_td]:py-2 [&_td]:border-b [&_td]:border-line/60 [&_tr:last-child_td]:border-b-0">
        {children}
      </table>
    </div>
  );
}
