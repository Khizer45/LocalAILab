import type { ModelInfo } from "../api/types";
import { useModels } from "../lib/queries";
import { Select } from "./ui";

/** Chat-capable model picker. Cloud models are labelled; blocked ones are disabled with the reason. */
export function ModelSelect({ value, onChange, id, className }: { value: string; onChange: (v: string) => void; id?: string; className?: string }) {
  const { data } = useModels();
  const models = (data?.models ?? []).filter((m) => !m.is_embedding);
  const local = models.filter((m) => m.provider === "local");
  const cloud = models.filter((m) => m.provider === "cloud");
  const label = (m: ModelInfo) =>
    `${m.name}${m.parameter_size ? ` (${m.parameter_size}${m.quantization ? `, ${m.quantization}` : ""})` : ""}${m.allowed ? "" : ` – ${m.blocked_reason}`}`;
  const known = models.some((m) => m.id === value);
  return (
    <Select id={id} value={value} onChange={(e) => onChange(e.target.value)} className={className} aria-label="Model">
      {!value && <option value="">Choose a model…</option>}
      {value && !known && <option value={value}>{value} (not installed)</option>}
      {local.length > 0 && (
        <optgroup label="On this computer">
          {local.map((m) => (
            <option key={m.id} value={m.id} disabled={!m.allowed}>{label(m)}</option>
          ))}
        </optgroup>
      )}
      {cloud.length > 0 && (
        <optgroup label="Cloud (data leaves this computer)">
          {cloud.map((m) => (
            <option key={m.id} value={m.id} disabled={!m.allowed}>{label(m)}</option>
          ))}
        </optgroup>
      )}
    </Select>
  );
}
