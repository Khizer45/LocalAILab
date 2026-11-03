import { useMutation } from "@tanstack/react-query";
import { Package, PackageOpen } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, download } from "../api/client";
import { Button, ErrorNotice, Notice } from "./ui";

type Kind = "benchmark" | "arena" | "agent";
const RUN_PATH: Record<Kind, string> = { benchmark: "/benchmarks/runs/", arena: "/arena/runs/", agent: "/agents/runs/" };

/** Export a run as a verifiable experiment bundle (ZIP with manifest + SHA-256s). */
export function BundleExport({ kind, runId }: { kind: Kind; runId: number }) {
  const [open, setOpen] = useState(false);
  return (
    <span className="relative">
      <Button onClick={() => setOpen((o) => !o)} aria-expanded={open}><Package className="size-4" />Bundle</Button>
      {open && (
        <div className="absolute right-0 z-20 mt-1 w-72 rounded-md border border-line bg-panel p-2 text-[13px] shadow-lg" role="menu">
          <button role="menuitem" className="block w-full rounded px-2 py-1.5 text-left hover:bg-raised" onClick={() => { setOpen(false); download(`/api/bundles/${kind}/${runId}?scope=full`); }}>
            Full bundle<span className="block text-[12px] text-faint">config, results, responses and traces – to reproduce or archive</span>
          </button>
          <button role="menuitem" className="block w-full rounded px-2 py-1.5 text-left hover:bg-raised" onClick={() => { setOpen(false); download(`/api/bundles/${kind}/${runId}?scope=metadata`); }}>
            Metadata only<span className="block text-[12px] text-faint">scores and settings without prompts, answers or data – safer to share</span>
          </button>
        </div>
      )}
    </span>
  );
}

/** Verify / import / re-run an experiment bundle someone exported. */
export function BundleImport() {
  const nav = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const body = () => { const fd = new FormData(); fd.append("file", file!); return fd; };
  const verify = useMutation({ mutationFn: () => api<{ ok: boolean; kind: Kind; scope: string; problems?: string[] }>("/api/bundles/verify", { method: "POST", body: body() }) });
  const imp = useMutation({ mutationFn: () => api<{ kind: Kind; run_id: number }>("/api/bundles/import", { method: "POST", body: body() }), onSuccess: (r) => nav(RUN_PATH[r.kind] + r.run_id) });
  const rerun = useMutation({ mutationFn: () => api<{ kind: Kind; run_id: number }>("/api/bundles/rerun", { method: "POST", body: body() }), onSuccess: (r) => nav(RUN_PATH[r.kind] + r.run_id) });
  const err = verify.error ?? imp.error ?? rerun.error;
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <label className="inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-md border border-line bg-raised px-3 text-[13px] hover:border-line-strong">
          <PackageOpen className="size-4" />{file ? file.name : "Choose bundle (.zip)"}
          <input type="file" accept=".zip" className="sr-only" onChange={(e) => { setFile(e.target.files?.[0] ?? null); verify.reset(); imp.reset(); rerun.reset(); }} />
        </label>
        <Button disabled={!file} onClick={() => verify.mutate()} busy={verify.isPending}>Verify</Button>
        <Button disabled={!file} onClick={() => imp.mutate()} busy={imp.isPending} title="Add the run and its results to this installation (read-only copy)">Import</Button>
        <Button disabled={!file} onClick={() => rerun.mutate()} busy={rerun.isPending} title="Run the same cases and settings again on this computer">Re-run here</Button>
      </div>
      {verify.data && <Notice tone={verify.data.ok ? "info" : "error"} title={verify.data.ok ? `Valid ${verify.data.kind} bundle (${verify.data.scope}) – every file matches its SHA-256.` : "The bundle failed verification"}>{verify.data.problems?.join("; ")}</Notice>}
      {err && <ErrorNotice error={err} />}
    </div>
  );
}
