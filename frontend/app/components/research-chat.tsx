"use client";

import { useState, useRef, useEffect } from "react";
import { SearchDisplay } from "./search-display";
import { MarkdownRenderer } from "./markdown-renderer";

// SSE streams must bypass the Next.js rewrite proxy (it buffers the
// full response in dev mode).  Non-streaming JSON calls still go
// through the proxy at /api/*.
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export type ResearchEvent =
  | { type: "phase-update"; phase: string; message: string; run_id?: string; provider?: string; model?: string }
  | { type: "thinking"; phase?: string; message: string; plan_length?: number }
  | { type: "subtasks"; count: number; titles: string[]; subtasks?: { id: string; title: string; description: string; objective: string }[] }
  | { type: "scaling"; complexity: string; subagent_count: number; target_sources: number; tool_calls_per_subagent?: number }
  | { type: "subagents-launch"; iteration: number; total_agents: number; agent_details: { id: string; title: string; description: string }[] }
  | { type: "subagent-step"; subtask_id: string; subtask_title: string; step: string; message: string; evidence_count?: number }
  | { type: "subagent-queries"; subtask_id: string; subtask_title: string; queries: string[]; count: number }
  | { type: "subagent-search"; subtask_id: string; query: string; status: string; results_found?: number; error?: string }
  | { type: "subagent-sources-scored"; subtask_id: string; subtask_title: string; total_candidates: number; unique_sources: number; top_urls: string[]; top_scores: { url: string; title: string; score: number }[] }
  | { type: "subagent-extract"; subtask_id: string; url: string; status: string; error?: string }
  | { type: "subagent-complete"; subtask_id: string; subtask_title: string; report_length: number; sources_count: number; evidence_count: number }
  | { type: "subagent-progress"; completed: number; message: string }
  | { type: "sources-found"; count: number; sources: { url: string; title: string; quality_score: number }[] }
  | { type: "llm-call"; status: string; model: string; provider: string; role: string; attempt?: number; option_index?: number; output_length?: number; error?: string }
  | { type: "log"; message: string; event_type: string }
  | { type: "warning"; phase: string; message: string }
  | { type: "progress"; phase: string; percent: number }
  | { type: "report-draft"; content: string; report_length?: number }
  | { type: "citations-added"; cited_report_length: number }
  | { type: "final-result"; content: string }
  | { type: "reflection"; decision?: string; research_complete: boolean; iteration: number; new_subtask_count?: number; new_subtasks?: { id: string; title: string }[]; total_reports?: number; total_sources?: number }
  | { type: "complete"; message: string; run_id: string; total_sources?: number; total_reports?: number; iterations?: number; provider?: string; model?: string }
  | { type: "error"; error: string; phase?: string; hint?: string };

const SUGGESTED_QUERIES = [
  "Compare the economic impacts of AI automation across different industries",
  "What are the latest breakthroughs in quantum computing?",
  "Analyze the global impact of microplastics on marine ecosystems",
  "How does SpaceX's Starship compare to NASA's SLS?",
];

/* ── Clean report content ────────────────────────────────────── */

function cleanReportContent(raw: string): string {
  if (!raw) return raw;
  let content = raw.trim();

  // If the content looks like JSON with a "report" or "content" key, extract it
  if (content.startsWith("{") && content.endsWith("}")) {
    try {
      const parsed = JSON.parse(content);
      if (typeof parsed.report === "string") content = parsed.report;
      else if (typeof parsed.content === "string") content = parsed.content;
      else if (typeof parsed.text === "string") content = parsed.text;
    } catch {
      /* not valid JSON, use as-is */
    }
  }

  // Strip leading ```markdown or ```json fences
  content = content.replace(/^```(?:markdown|md|json)?\s*\n?/i, "");
  content = content.replace(/\n?```\s*$/i, "");

  return content.trim();
}

/* ── Model Selector Component ─────────────────────────────────── */

function ModelSelector({
  availableModels,
  selectedProvider,
  selectedModel,
  onProviderChange,
  onModelChange,
  disabled,
}: {
  availableModels: Record<string, string[]>;
  selectedProvider: string;
  selectedModel: string;
  onProviderChange: (p: string) => void;
  onModelChange: (m: string) => void;
  disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const providerBadges: Record<string, string> = {
    gemini: "G",
    openai: "O",
    anthropic: "A",
    huggingface: "HF",
  };

  const providerColors: Record<string, string> = {
    gemini: "text-blue-400",
    openai: "text-emerald-400",
    anthropic: "text-orange-400",
    huggingface: "text-yellow-400",
  };

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => !disabled && setOpen(!open)}
        disabled={disabled}
        className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl glass text-[11px] font-medium text-white/50 hover:text-white/70 hover:border-white/15 transition-all duration-200 disabled:opacity-30"
      >
        <span className={`inline-flex items-center justify-center min-w-4 h-4 rounded-md bg-white/5 text-[9px] font-semibold ${providerColors[selectedProvider] || "text-white/50"}`}>
          {providerBadges[selectedProvider] || "-"}
        </span>
        <span className="truncate max-w-[120px]">{selectedModel}</span>
        <svg className={`w-3 h-3 text-white/20 transition-transform ${open ? "rotate-180" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="absolute bottom-full left-0 mb-2 w-64 glass rounded-2xl border border-white/10 shadow-xl shadow-black/30 overflow-hidden z-50 animate-fade-in">
          {Object.entries(availableModels).map(([provider, models]) => (
            <div key={provider}>
              <div className="px-3 py-1.5 text-[10px] uppercase tracking-wider text-white/25 font-semibold flex items-center gap-1.5 border-b border-white/[0.04]">
                <span className={`inline-flex items-center justify-center min-w-4 h-4 rounded-md bg-white/5 text-[9px] font-semibold ${providerColors[provider] || "text-white/30"}`}>
                  {providerBadges[provider] || "-"}
                </span>
                {provider}
              </div>
              {models.map((model) => (
                <button
                  key={`${provider}-${model}`}
                  type="button"
                  onClick={() => {
                    onProviderChange(provider);
                    onModelChange(model);
                    setOpen(false);
                  }}
                  className={`w-full text-left px-3 py-2 text-xs hover:bg-white/[0.04] transition-colors flex items-center gap-2 ${
                    selectedProvider === provider && selectedModel === model
                      ? "text-violet-400 bg-violet-500/5"
                      : "text-white/50"
                  }`}
                >
                  <span className="font-mono text-[11px] truncate">{model}</span>
                  {selectedProvider === provider && selectedModel === model && (
                    <svg className="w-3 h-3 text-violet-400 ml-auto flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                </button>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Main Chat Component ──────────────────────────────────────── */

export function ResearchChat({
  onModelChange,
}: {
  onModelChange?: (provider: string, model: string) => void;
}) {
  const [messages, setMessages] = useState<
    Array<{
      id: string;
      role: "user" | "assistant";
      content: string;
      events?: ResearchEvent[];
      finalReport?: string;
    }>
  >([]);
  const [input, setInput] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [availableModels, setAvailableModels] = useState<Record<string, string[]>>({});
  const [selectedProvider, setSelectedProvider] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const lastSyncedModelRef = useRef("");
  const scrollRef = useRef<HTMLDivElement>(null);

  // Fetch available models on mount
  useEffect(() => {
    fetch("/api/config")
      .then((r) => r.json())
      .then((data) => {
        if (data.available_models) setAvailableModels(data.available_models);
        if (data.default_provider) setSelectedProvider(data.default_provider);
        if (data.default_model) setSelectedModel(data.default_model);
      })
      .catch(() => { /* fallback: use defaults */ });
  }, []);

  useEffect(() => {
    if (selectedProvider && selectedModel) {
      const nextKey = `${selectedProvider}::${selectedModel}`;
      if (lastSyncedModelRef.current !== nextKey) {
        lastSyncedModelRef.current = nextKey;
        onModelChange?.(selectedProvider, selectedModel);
      }
    }
  }, [selectedProvider, selectedModel, onModelChange]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const performSearch = async (query: string) => {
    if (!query.trim() || isSearching) return;
    setIsSearching(true);

    const userMsgId = Date.now().toString();
    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: "user", content: query },
    ]);

    const assistantMsgId = (Date.now() + 1).toString();
    setMessages((prev) => [
      ...prev,
      { id: assistantMsgId, role: "assistant", content: "", events: [] },
    ]);

    setInput("");

    try {
      const response = await fetch(`${BACKEND_URL}/api/research/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          max_iterations: 2,
          quality_threshold: 0.7,
          ...(selectedModel && { model: selectedModel }),
          ...(selectedProvider && { provider: selectedProvider }),
        }),
      });

      if (!response.ok) throw new Error(`Server error: ${response.status}`);

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No readable stream");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const event: ResearchEvent = JSON.parse(line.slice(6));
              setMessages((prev) =>
                prev.map((msg) => {
                  if (msg.id === assistantMsgId) {
                    const updatedEvents = [...(msg.events || []), event];
                    let finalReport = msg.finalReport;
                    if (event.type === "final-result") {
                      const cleaned = cleanReportContent(event.content);
                      finalReport = !msg.finalReport || cleaned.length >= msg.finalReport.length
                        ? cleaned
                        : msg.finalReport;
                    }
                    else if (event.type === "report-draft" && !finalReport) finalReport = cleanReportContent(event.content);
                    return { ...msg, events: updatedEvents, finalReport };
                  }
                  return msg;
                })
              );
            } catch {
              /* skip malformed */
            }
          }
        }
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : "Unknown error";
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMsgId
            ? { ...msg, events: [...(msg.events || []), { type: "error" as const, error: errorMsg }] }
            : msg
        )
      );
    } finally {
      setIsSearching(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    performSearch(input);
  };

  return (
    <div className="w-full max-w-[96rem] mx-auto">
      {/* Messages area */}
      <div
        ref={scrollRef}
        className="space-y-6 mb-6 max-h-[calc(100vh-260px)] overflow-y-auto scrollbar-hide"
      >
        {/* Suggestion cards when empty */}
        {messages.length === 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-2 opacity-0 animate-fade-up [animation-delay:700ms] [animation-fill-mode:forwards]">
            {SUGGESTED_QUERIES.map((q, i) => (
              <button
                key={i}
                onClick={() => performSearch(q)}
                className="text-left p-4 rounded-2xl glass glass-interactive transition-all duration-300 group hover:shadow-lg hover:shadow-violet-500/5"
              >
                <div className="flex items-start gap-3">
                  <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-violet-500/20 to-indigo-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <svg className="w-3.5 h-3.5 text-violet-400/70 group-hover:text-violet-400 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <p className="text-sm text-white/40 group-hover:text-white/70 transition-colors leading-relaxed">
                    {q}
                  </p>
                </div>
              </button>
            ))}
          </div>
        )}

        {/* Messages */}
        {messages.map((msg) => (
          <div key={msg.id} className="animate-fade-in">
            {msg.role === "user" ? (
              <div className="flex items-start gap-3 mb-4">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-500/80 to-cyan-500/80 flex items-center justify-center flex-shrink-0 shadow-lg shadow-blue-500/10">
                  <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                </div>
                <div className="glass-elevated rounded-2xl px-4 py-3 max-w-[80%]">
                  <p className="text-sm font-medium text-white/90">{msg.content}</p>
                </div>
              </div>
            ) : (
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-violet-500/80 to-indigo-600/80 flex items-center justify-center flex-shrink-0 shadow-lg shadow-violet-500/10">
                  <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <div className="flex-1 min-w-0">
                  {msg.events && msg.events.length > 0 && (
                    <SearchDisplay events={msg.events} />
                  )}
                  {msg.finalReport && (
                    <div className="mt-4 glass rounded-2xl p-6 prose prose-invert prose-sm max-w-none prose-headings:text-white/90 prose-p:text-white/70 prose-a:text-violet-400 prose-strong:text-white/80 prose-code:text-violet-300 prose-pre:bg-white/5 prose-pre:border prose-pre:border-white/10">
                      <MarkdownRenderer content={msg.finalReport} />
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* ── Glass Input Area ────────────────────────────────── */}
      <form
        onSubmit={handleSubmit}
        className="sticky bottom-0 pt-4 pb-2"
      >
        <div className="relative group">
          {/* Glow effect behind input */}
          <div className="absolute -inset-0.5 bg-gradient-to-r from-violet-500/20 via-indigo-500/20 to-purple-500/20 rounded-2xl blur-lg opacity-0 group-focus-within:opacity-100 transition-opacity duration-500" />

          <div className="relative glass rounded-2xl transition-all duration-300 group-focus-within:border-white/15">
            <div className="flex items-center">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={isSearching ? "Research in progress..." : "Ask anything — I'll research it deeply..."}
                disabled={isSearching}
                className="flex-1 px-5 py-3 bg-transparent text-white/90 placeholder:text-white/25 focus:outline-none disabled:opacity-40 text-sm"
              />

              {/* Model selector */}
              {Object.keys(availableModels).length > 0 && (
                <div className="pr-1">
                  <ModelSelector
                    availableModels={availableModels}
                    selectedProvider={selectedProvider}
                    selectedModel={selectedModel}
                    onProviderChange={setSelectedProvider}
                    onModelChange={setSelectedModel}
                    disabled={isSearching}
                  />
                </div>
              )}

              <button
                type="submit"
                disabled={isSearching || !input.trim()}
                className="mr-3 w-8 h-8 rounded-xl bg-gradient-to-r from-violet-500 to-indigo-600 text-white flex items-center justify-center flex-shrink-0 disabled:opacity-20 hover:shadow-lg hover:shadow-violet-500/25 transition-all duration-200"
              >
              {isSearching ? (
                <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
              ) : (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              )}
              </button>
            </div>
          </div>
        </div>
      </form>
    </div>
  );
}
