import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Brain, Eraser, Plus, Send, SlidersHorizontal, Square, Trash2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, ApiError, streamNdjson, type Metrics, type Source } from "../api/client";
import type { Conversation, Message } from "../api/types";
import { Markdown } from "../components/Markdown";
import { ModelSelect } from "../components/ModelSelect";
import { MetricsLine, SourceList } from "../components/Sources";
import { Button, cx, Empty, ErrorNotice, Field, Input, Textarea, Toggle } from "../components/ui";
import { when } from "../lib/format";
import { useSettings } from "../lib/queries";

interface LiveReply {
  content: string;
  thinking: string;
  sources: Source[];
  warnings: string[];
  metrics: Metrics | null;
  model: string;
  error: ApiError | null;
}

interface Options {
  temperature?: number;
  top_p?: number;
  num_ctx?: number;
  num_predict?: number;
  seed?: number;
}

function Bubble({ m }: { m: Message }) {
  if (m.role === "user")
    return (
      <div className="flex justify-end">
        <div className="max-w-[78%] rounded-lg bg-raised border border-line px-3.5 py-2 whitespace-pre-wrap">{m.content}</div>
      </div>
    );
  return (
    <div className="max-w-[88%]">
      {m.thinking && (
        <details className="mb-2 text-[12.5px] text-muted">
          <summary className="cursor-pointer select-none">Model reasoning ({m.thinking.length} characters)</summary>
          <div className="mt-1 whitespace-pre-wrap border-l-2 border-line pl-3">{m.thinking}</div>
        </details>
      )}
      <Markdown text={m.content} />
      {m.status !== "complete" && <div className="mt-1 text-[12px] text-amber">{m.status === "partial" ? "Stopped before the end" : "Failed"}</div>}
      <SourceList sources={m.sources ?? []} />
      <MetricsLine metrics={m.metrics} model={m.model} />
    </div>
  );
}

export default function Chat() {
  const { id } = useParams();
  const convId = id ? Number(id) : null;
  const nav = useNavigate();
  const qc = useQueryClient();
  const settings = useSettings();
  const list = useQuery({ queryKey: ["conversations"], queryFn: () => api<{ conversations: Conversation[] }>("/api/chat/conversations") });
  const conv = useQuery({ queryKey: ["conversation", convId], queryFn: () => api<Conversation>(`/api/chat/conversations/${convId}`), enabled: convId != null });

  const [model, setModel] = useState("");
  const [options, setOptions] = useState<Options>({});
  const [useDocs, setUseDocs] = useState(false);
  const [think, setThink] = useState(false);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [showParams, setShowParams] = useState(false);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState<string | null>(null);
  const [live, setLive] = useState<LiveReply | null>(null);
  const [error, setError] = useState<unknown>(null);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (conv.data) {
      setModel(conv.data.model || settings.data?.runtime.default_model || "");
      setOptions(conv.data.options ?? {});
      setUseDocs(conv.data.use_documents);
      setSystemPrompt(conv.data.system_prompt);
    } else if (!convId && settings.data) {
      setModel((m) => m || settings.data!.runtime.default_model);
    }
  }, [conv.data, convId, settings.data]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [conv.data?.messages?.length, live?.content, pending]);

  const create = useMutation({
    mutationFn: () => api<Conversation>("/api/chat/conversations", { method: "POST", json: { model, use_documents: useDocs } }),
    onSuccess: (c) => {
      qc.invalidateQueries({ queryKey: ["conversations"] });
      nav(`/chat/${c.id}`);
    },
  });

  const patch = (body: Partial<Conversation>) =>
    convId && api(`/api/chat/conversations/${convId}`, { method: "PATCH", json: body }).then(() => qc.invalidateQueries({ queryKey: ["conversations"] }));

  const send = async () => {
    const text = input.trim();
    if (!text || live) return;
    setError(null);
    let target = convId;
    if (target == null) {
      try {
        const c = await api<Conversation>("/api/chat/conversations", { method: "POST", json: { model, use_documents: useDocs, system_prompt: systemPrompt } });
        target = c.id;
        nav(`/chat/${c.id}`, { replace: true });
      } catch (e) {
        setError(e);
        return;
      }
    }
    setInput("");
    setPending(text);
    const reply: LiveReply = { content: "", thinking: "", sources: [], warnings: [], metrics: null, model, error: null };
    setLive(reply);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      await streamNdjson(`/api/chat/conversations/${target}/messages`,
        { content: text, model, options, use_documents: useDocs, think: think ? true : undefined },
        (ev) => {
          if (ev.type === "token") reply.content += ev.content;
          else if (ev.type === "thinking") reply.thinking += ev.content;
          else if (ev.type === "sources") { reply.sources = ev.sources; reply.warnings = ev.warnings; }
          else if (ev.type === "done") reply.metrics = ev.metrics;
          else if (ev.type === "error") reply.error = new ApiError(500, ev.code, ev.message, ev.hint);
          setLive({ ...reply });
        }, ctrl.signal);
      if (reply.error) setError(reply.error);
    } catch (e) {
      setError(e);
      setInput(text);
    } finally {
      abortRef.current = null;
      await qc.invalidateQueries({ queryKey: ["conversation", target] });
      qc.invalidateQueries({ queryKey: ["conversations"] });
      setLive(null);
      setPending(null);
    }
  };

  const stop = () => abortRef.current?.abort();

  const remove = async (cid: number) => {
    await api(`/api/chat/conversations/${cid}`, { method: "DELETE" });
    qc.invalidateQueries({ queryKey: ["conversations"] });
    if (cid === convId) nav("/chat");
  };

  const clear = async () => {
    if (!convId) return;
    await api(`/api/chat/conversations/${convId}/messages`, { method: "DELETE" });
    qc.invalidateQueries({ queryKey: ["conversation", convId] });
  };

  const messages = conv.data?.messages ?? [];
  const setOpt = (k: keyof Options, v: string) => setOptions((o) => ({ ...o, [k]: v === "" ? undefined : Number(v) }));
  const maxOut = settings.data?.runtime.max_output_tokens;
  const maxCtx = settings.data?.runtime.max_context_tokens;

  return (
    <div className="-mx-5 -my-6 md:-mx-8 flex h-[calc(100vh-50px)] min-h-[520px]">
      <h1 className="sr-only">Chat</h1>
      {/* conversation list */}
      <aside className="hidden lg:flex w-[250px] shrink-0 flex-col border-r border-line">
        <div className="p-3"><Button variant="primary" className="w-full" onClick={() => create.mutate()} busy={create.isPending}><Plus className="size-4" />New chat</Button></div>
        <nav className="flex-1 overflow-y-auto px-2 pb-3" aria-label="Conversations">
          {list.data?.conversations.length === 0 && <p className="px-3 text-[13px] text-faint">Your conversations are stored only in the local database.</p>}
          {list.data?.conversations.map((c) => (
            <div key={c.id} className={cx("group flex items-center rounded-md", c.id === convId ? "bg-raised" : "hover:bg-raised/60")}>
              <button onClick={() => nav(`/chat/${c.id}`)} className="min-w-0 flex-1 px-3 py-2 text-left">
                <div className="truncate text-[13.5px]">{c.title}</div>
                <div className="truncate text-[12px] text-faint">{c.model || "no model"}, {when(c.updated_at)}</div>
              </button>
              <button onClick={() => remove(c.id)} className="mr-1 rounded p-1.5 text-faint opacity-0 group-hover:opacity-100 focus:opacity-100 hover:text-coral" aria-label={`Delete ${c.title}`}>
                <Trash2 className="size-3.5" />
              </button>
            </div>
          ))}
        </nav>
      </aside>

      {/* thread */}
      <section className="flex min-w-0 flex-1 flex-col">
        <header className="flex flex-wrap items-center gap-2 border-b border-line px-4 py-2.5">
          <div className="min-w-[220px] flex-1 max-w-[380px]"><ModelSelect value={model} onChange={(v) => { setModel(v); patch({ model: v }); }} /></div>
          <Toggle checked={useDocs} onChange={(v) => { setUseDocs(v); patch({ use_documents: v }); }} label="Answer from my documents" />
          <div className="ml-auto flex gap-1">
            <Button variant="ghost" onClick={() => setShowParams((s) => !s)} aria-expanded={showParams}><SlidersHorizontal className="size-4" />Parameters</Button>
            {convId && <Button variant="ghost" onClick={clear} title="Delete all messages in this chat"><Eraser className="size-4" />Clear</Button>}
          </div>
        </header>

        <div className="flex-1 overflow-y-auto px-4 md:px-8 py-6">
          <div className="mx-auto max-w-[820px] space-y-6">
            {!convId && !pending && (
              <Empty title="Start a private conversation">
                Messages are sent only to the model runtime on this computer{settings.data?.runtime.offline_only === false ? " (or to the cloud model you pick)" : ""}. Turn on “Answer from my documents” to get cited answers from files you indexed on the Documents page.
              </Empty>
            )}
            {conv.isError && <ErrorNotice error={conv.error} />}
            {messages.map((m) => <Bubble key={m.id} m={m} />)}
            {pending && <Bubble m={{ id: -1, role: "user", content: pending, thinking: null, model: null, status: "complete", metrics: null, sources: null, created_at: "" }} />}
            {live && (
              <div className="max-w-[88%]">
                {live.thinking && <div className="mb-2 text-[12.5px] text-muted whitespace-pre-wrap border-l-2 border-line pl-3">{live.thinking}</div>}
                {live.content ? <Markdown text={live.content} /> : <div className="text-muted text-[13px]">{live.thinking ? "Reasoning…" : "Waiting for the first token… (the first message also loads the model into memory)"}</div>}
                <SourceList sources={live.sources} warnings={live.warnings} />
                <MetricsLine metrics={live.metrics} model={live.model} />
              </div>
            )}
            {error != null && <ErrorNotice error={error} />}
            <div ref={bottomRef} />
          </div>
        </div>

        <div className="border-t border-line px-4 md:px-8 py-3">
          <div className="mx-auto flex max-w-[820px] items-end gap-2">
            <Textarea
              rows={2}
              value={input}
              placeholder={model ? `Message ${model}… (Enter to send, Shift+Enter for a new line)` : "Choose a model first"}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
              className="max-h-48 resize-y"
              aria-label="Message"
            />
            {live ? (
              <Button variant="secondary" onClick={stop} className="h-10"><Square className="size-4" />Stop</Button>
            ) : (
              <Button variant="primary" onClick={send} disabled={!input.trim() || !model} className="h-10"><Send className="size-4" />Send</Button>
            )}
          </div>
        </div>
      </section>

      {/* parameters */}
      {showParams && (
        <aside className="w-[290px] shrink-0 overflow-y-auto border-l border-line p-4 space-y-4" aria-label="Generation parameters">
          <h2 className="font-cond text-[16px] font-medium">Generation parameters</h2>
          <Field label="System prompt" hint="Empty uses the default from Settings.">
            <Textarea rows={5} value={systemPrompt} onChange={(e) => setSystemPrompt(e.target.value)} onBlur={() => patch({ system_prompt: systemPrompt })}
              placeholder={settings.data?.runtime.default_system_prompt} />
          </Field>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Temperature"><Input type="number" step="0.1" min={0} max={2} value={options.temperature ?? ""} placeholder="model default" onChange={(e) => setOpt("temperature", e.target.value)} /></Field>
            <Field label="Top p"><Input type="number" step="0.05" min={0} max={1} value={options.top_p ?? ""} placeholder="default" onChange={(e) => setOpt("top_p", e.target.value)} /></Field>
            <Field label="Context tokens" hint={maxCtx ? `limit ${maxCtx}` : undefined}><Input type="number" min={256} max={maxCtx} value={options.num_ctx ?? ""} placeholder="default" onChange={(e) => setOpt("num_ctx", e.target.value)} /></Field>
            <Field label="Max output tokens" hint={maxOut ? `limit ${maxOut}` : undefined}><Input type="number" min={1} max={maxOut} value={options.num_predict ?? ""} placeholder={String(maxOut ?? "")} onChange={(e) => setOpt("num_predict", e.target.value)} /></Field>
            <Field label="Seed" hint="Same seed + temperature 0 gives repeatable answers"><Input type="number" value={options.seed ?? ""} placeholder="random" onChange={(e) => setOpt("seed", e.target.value)} /></Field>
          </div>
          <Toggle checked={think} onChange={setThink} label={<span className="inline-flex items-center gap-1.5"><Brain className="size-4" />Show reasoning</span>}
            description="Only for models with the thinking capability (e.g. qwen3, deepseek-r1). Slower." />
          <p className="text-[12px] text-faint">Values above are sent with each message and saved on this conversation. Limits are set by the administrator in Settings.</p>
        </aside>
      )}
    </div>
  );
}
