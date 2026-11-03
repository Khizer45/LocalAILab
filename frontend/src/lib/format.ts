export const SERIES = ["#f2b84b", "#57c2b0", "#a28beb", "#6fb2f2", "#ee7468", "#c7d36f", "#e58fc4", "#9aa7b8"];

export function bytes(n: number | null | undefined): string {
  if (n == null) return "–";
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}`;
}

export function ms(v: number | null | undefined): string {
  if (v == null) return "–";
  if (v >= 10_000) return `${(v / 1000).toFixed(1)} s`;
  if (v >= 1000) return `${(v / 1000).toFixed(2)} s`;
  if (v < 1) return `${v.toFixed(2)} ms`;
  if (v < 10) return `${v.toFixed(1)} ms`;
  return `${Math.round(v)} ms`;
}

export function num(v: number | null | undefined, digits = 1): string {
  if (v == null || Number.isNaN(v)) return "–";
  return v.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: 0 });
}

export function pct(v: number | null | undefined): string {
  if (v == null) return "–";
  return `${Math.round(v * 100)}%`;
}

export function duration(seconds: number | null | undefined): string {
  if (seconds == null) return "–";
  const s = Math.round(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${r}s`;
  return `${r}s`;
}

export function when(iso: string | null | undefined): string {
  if (!iso) return "–";
  // The API sends UTC timestamps without a zone ("…T10:00:00"); timestamps with a zone are used as they are.
  const d = new Date(/(Z|[+-]\d\d:?\d\d)$/i.test(iso) ? iso : iso + "Z");
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function humanize(id: string): string {
  const s = id.replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}
