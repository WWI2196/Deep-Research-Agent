"use client";

import { useState, useEffect, useMemo, useRef } from "react";
import type { ResearchEvent } from "./research-chat";
import { MarkdownRenderer } from "./markdown-renderer";

/* ================================================================== */
/* Types                                                               */
/* ================================================================== */

interface SubagentState {
  id: string;
  title: string;
  description: string;
  status:
    | "pending"
    | "generating-queries"
    | "searching"
    | "evaluating-sources"
    | "refining-queries"
    | "extracting"
    | "writing-report"
    | "complete"
    | "error";
  queries: string[];
  searches: { query: string; status: string; results_found?: number }[];
  sources: { url: string; title: string; score: number }[];
  extractions: { url: string; status: string }[];
  evidenceCount: number;
  reportLength: number;
}

interface LLMCall {
  model: string;
  provider: string;
  role: string;
  status: "started" | "completed" | "error";
  attempt: number;
  error?: string;
  outputLength?: number;
  timestamp: number;
}

interface GatheredSource {
  url: string;
  title: string;
  score: number;
  subtask: string;
  timestamp: number;
}

/* ================================================================== */
/* Phase config                                                        */
/* ================================================================== */

const PHASE_ORDER = [
  "init",
  "plan",
  "split",
  "scale",
  "subagents",
  "reflection",
  "synthesize",
  "cite",
  "complete",
];

const PHASE_LABELS: Record<string, string> = {
  init: "Initializing",
  plan: "Planning Research",
  split: "Creating Subtasks",
  scale: "Estimating Complexity",
  subagents: "Running Subagents",
  reflection: "Reflecting & Gap Analysis",
  synthesize: "Synthesizing Report",
  cite: "Adding Citations",
  complete: "Research Complete",
};

const PHASE_ICONS: Record<string, string> = {
  init: "",
  plan: "",
  split: "",
  scale: "",
  subagents: "",
  reflection: "",
  synthesize: "",
  cite: "",
  complete: "",
};

/* ================================================================== */
/* Utility components                                                  */
/* ================================================================== */

function StatusDot({ status }: { status: string }) {
  const colors: Record<string, string> = {
    started: "bg-blue-400 shadow-blue-400/50 shadow-[0_0_6px]",
    completed: "bg-emerald-400 shadow-emerald-400/50 shadow-[0_0_4px]",
    error: "bg-red-400 shadow-red-400/50 shadow-[0_0_6px]",
    failed: "bg-red-400",
    "no-data": "bg-amber-400",
    pending: "bg-white/20",
  };
  return (
    <span
      className={`inline-block w-1.5 h-1.5 rounded-full flex-shrink-0 ${
        colors[status] || "bg-white/20"
      } ${status === "started" ? "animate-pulse" : ""}`}
    />
  );
}

function QualityBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color =
    pct >= 70
      ? "from-emerald-500 to-emerald-400"
      : pct >= 40
        ? "from-amber-500 to-amber-400"
        : "from-red-500 to-red-400";
  return (
    <div className="flex items-center gap-2 min-w-[80px]">
      <div className="flex-1 h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${color} transition-all duration-700 ease-out`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-[10px] font-mono text-white/40 w-8 text-right">
        {pct}%
      </span>
    </div>
  );
}

function PhaseIcon({
  status,
}: {
  status: "pending" | "active" | "completed";
}) {
  if (status === "completed") {
    return (
      <div className="w-7 h-7 rounded-full bg-emerald-400/15 flex items-center justify-center border border-emerald-400/20 shadow-[0_0_12px_rgba(52,211,153,0.15)]">
        <svg
          className="w-3.5 h-3.5 text-emerald-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2.5}
            d="M5 13l4 4L19 7"
          />
        </svg>
      </div>
    );
  }
  if (status === "active") {
    return (
      <div className="w-7 h-7 rounded-full bg-violet-500/15 flex items-center justify-center border border-violet-400/30 shadow-[0_0_16px_rgba(139,92,246,0.2)]">
        <div className="w-2.5 h-2.5 rounded-full bg-violet-400 animate-pulse" />
      </div>
    );
  }
  return (
    <div className="w-7 h-7 rounded-full bg-white/5 flex items-center justify-center border border-white/10">
      <div className="w-2 h-2 rounded-full bg-white/15" />
    </div>
  );
}

/* ================================================================== */
/* Progress Bar                                                        */
/* ================================================================== */

function ProgressBar({
  percent,
  elapsed,
  isComplete,
  currentPhase,
}: {
  percent: number;
  elapsed: number;
  isComplete: boolean;
  currentPhase: string;
}) {
  const formatTime = (ms: number) => {
    const s = Math.floor(ms / 1000);
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
  };

  return (
    <div className="glass rounded-2xl px-5 py-4 mb-3 animate-fade-in">
      <div className="flex items-center justify-between mb-2.5">
        <div className="flex items-center gap-2.5">
          {!isComplete ? (
            <div className="relative w-5 h-5">
              <div className="absolute inset-0 rounded-full border-2 border-violet-500/20" />
              <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-violet-400 animate-spin" />
            </div>
          ) : (
            <div className="w-5 h-5 rounded-full bg-emerald-400/20 flex items-center justify-center">
              <svg
                className="w-3 h-3 text-emerald-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2.5}
                  d="M5 13l4 4L19 7"
                />
              </svg>
            </div>
          )}
          <div>
            <span className="text-sm font-medium text-white/70">
              {isComplete
                ? "Research Complete"
                : PHASE_LABELS[currentPhase] || "Researching..."}
            </span>
            {!isComplete && currentPhase && (
              <span className="text-[10px] text-white/30 ml-2">
                {PHASE_ICONS[currentPhase] || ""}
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-[11px] text-white/25 font-mono tabular-nums">
            {formatTime(elapsed)}
          </span>
          <span className="text-sm font-bold text-white/70 tabular-nums">
            {Math.min(percent, 100)}%
          </span>
        </div>
      </div>
      <div className="h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-1000 ease-out ${
            isComplete
              ? "bg-gradient-to-r from-emerald-500 to-emerald-400"
              : "bg-gradient-to-r from-violet-500 via-indigo-500 to-violet-500 animate-gradient"
          }`}
          style={{ width: `${Math.min(percent, 100)}%` }}
        />
      </div>
    </div>
  );
}

/* ================================================================== */
/* Subagent Card                                                       */
/* ================================================================== */

function SubagentCard({
  agent,
  index,
}: {
  agent: SubagentState;
  index: number;
}) {
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (!["pending", "complete"].includes(agent.status)) {
      setExpanded(true);
    }
  }, [agent.status]);

  const statusLabels: Record<string, string> = {
    pending: "Queued",
    "generating-queries": "Generating queries",
    searching: "Searching web",
    "evaluating-sources": "Evaluating sources",
    "refining-queries": "Refining queries",
    extracting: "Extracting content",
    "writing-report": "Writing report",
    complete: "Complete",
    error: "Failed",
  };

  const statusColors: Record<string, string> = {
    pending: "text-white/25 bg-white/5",
    "generating-queries": "text-blue-400 bg-blue-400/10",
    searching: "text-cyan-400 bg-cyan-400/10",
    "evaluating-sources": "text-violet-400 bg-violet-400/10",
    "refining-queries": "text-amber-400 bg-amber-400/10",
    extracting: "text-orange-400 bg-orange-400/10",
    "writing-report": "text-indigo-400 bg-indigo-400/10",
    complete: "text-emerald-400 bg-emerald-400/10",
    error: "text-red-400 bg-red-400/10",
  };

  const isActive = !["pending", "complete", "error"].includes(agent.status);

  return (
    <div
      className={`rounded-xl transition-all duration-500 animate-fade-in ${
        isActive
          ? "glass-elevated border-violet-500/20 shadow-lg shadow-violet-500/[0.07]"
          : agent.status === "complete"
            ? "glass border-emerald-500/10"
            : "glass"
      }`}
      style={{ animationDelay: `${index * 80}ms` }}
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2.5 px-3.5 py-2.5 text-left group"
      >
        <div
          className={`w-6 h-6 rounded-lg flex items-center justify-center text-[10px] font-bold flex-shrink-0 ${
            agent.status === "complete"
              ? "bg-emerald-400/15 text-emerald-400"
              : isActive
                ? "bg-violet-500/20 text-violet-400"
                : "bg-white/5 text-white/25"
          }`}
        >
          {agent.status === "complete" ? "✓" : index + 1}
        </div>

        <div className="flex-1 min-w-0">
          <span
            className={`text-xs font-medium truncate block ${
              isActive ? "text-white/80" : "text-white/60"
            }`}
          >
            {agent.title}
          </span>
          {isActive && (
            <span className="text-[10px] text-white/25 block mt-0.5">
              {agent.searches.length > 0 &&
                `${agent.searches.length} searches`}
              {agent.sources.length > 0 &&
                ` · ${agent.sources.length} sources`}
              {agent.evidenceCount > 0 &&
                ` · ${agent.evidenceCount} evidence`}
            </span>
          )}
        </div>

        <span
          className={`text-[10px] font-medium px-2 py-0.5 rounded-full flex-shrink-0 ${
            statusColors[agent.status] || "text-white/30 bg-white/5"
          }`}
        >
          {statusLabels[agent.status] || agent.status}
        </span>

        <svg
          className={`w-3 h-3 text-white/20 transition-transform flex-shrink-0 ${
            expanded ? "rotate-180" : ""
          }`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </button>

      {expanded && (
        <div className="px-3.5 pb-3 space-y-3 border-t border-white/[0.06] pt-3 animate-slide-down">
          {/* Queries */}
          {agent.queries.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-white/25 mb-1.5 font-semibold flex items-center gap-1.5">
                <svg
                  className="w-3 h-3 text-blue-400/50"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                  />
                </svg>
                Queries ({agent.queries.length})
              </div>
              <div className="flex flex-wrap gap-1">
                {agent.queries.slice(0, 6).map((q, i) => (
                  <span
                    key={i}
                    className="text-[10px] px-2 py-0.5 rounded-md bg-white/[0.04] text-white/35 border border-white/[0.06] truncate max-w-[200px]"
                  >
                    {q}
                  </span>
                ))}
                {agent.queries.length > 6 && (
                  <span className="text-[10px] px-2 py-0.5 text-white/20">
                    +{agent.queries.length - 6} more
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Searches */}
          {agent.searches.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-white/25 mb-1.5 font-semibold flex items-center gap-1.5">
                <svg
                  className="w-3 h-3 text-cyan-400/50"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                Web Searches
              </div>
              <div className="space-y-1">
                {agent.searches.map((s, i) => (
                  <div key={i} className="flex items-center gap-2 text-[11px]">
                    <StatusDot status={s.status} />
                    <span className="text-white/35 truncate flex-1">
                      {s.query}
                    </span>
                    {s.results_found !== undefined && s.results_found > 0 && (
                      <span className="text-emerald-400/50 text-[10px] font-mono">
                        {s.results_found} hits
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Sources with quality bars */}
          {agent.sources.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-white/25 mb-1.5 font-semibold flex items-center gap-1.5">
                <svg
                  className="w-3 h-3 text-violet-400/50"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
                Rated Sources ({agent.sources.length})
              </div>
              <div className="space-y-1.5 max-h-40 overflow-y-auto scrollbar-hide">
                {agent.sources.slice(0, 10).map((s, i) => (
                  <div key={i} className="flex items-center gap-2 group/src">
                    <QualityBar score={s.score} />
                    <a
                      href={s.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-[11px] text-white/35 hover:text-violet-400 truncate flex-1 transition-colors"
                    >
                      {s.title ||
                        (() => {
                          try {
                            return new URL(s.url).hostname;
                          } catch {
                            return s.url;
                          }
                        })()}
                    </a>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Extractions */}
          {agent.extractions.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-white/25 mb-1.5 font-semibold flex items-center gap-1.5">
                <svg
                  className="w-3 h-3 text-orange-400/50"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                  />
                </svg>
                Extracting
              </div>
              <div className="grid grid-cols-2 gap-1">
                {agent.extractions.map((e, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-1.5 text-[10px] px-2 py-1 rounded-md bg-white/[0.02]"
                  >
                    <StatusDot status={e.status} />
                    <span className="text-white/30 truncate">
                      {(() => {
                        try {
                          return new URL(e.url).hostname;
                        } catch {
                          return e.url;
                        }
                      })()}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Stats row */}
          {(agent.evidenceCount > 0 || agent.reportLength > 0) && (
            <div className="flex gap-4 pt-2 border-t border-white/[0.04]">
              {agent.evidenceCount > 0 && (
                <div className="flex items-center gap-1.5 text-[10px]">
                  <span className="text-emerald-400/70 font-bold">
                    {agent.evidenceCount}
                  </span>
                  <span className="text-white/25">evidence items</span>
                </div>
              )}
              {agent.reportLength > 0 && (
                <div className="flex items-center gap-1.5 text-[10px]">
                  <span className="text-blue-400/70 font-bold">
                    {(agent.reportLength / 1000).toFixed(1)}k
                  </span>
                  <span className="text-white/25">chars written</span>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function SubagentParallelBoard({ subagents }: { subagents: SubagentState[] }) {
  return (
    <div className="space-y-2">
      <div className="text-[10px] uppercase tracking-wider text-white/25 font-semibold">
        Parallel Agent Lanes ({subagents.length})
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-2">
        {subagents.map((agent, idx) => (
          <div key={agent.id} className="glass rounded-xl p-3 animate-fade-in" style={{ animationDelay: `${idx * 40}ms` }}>
            <div className="flex items-start justify-between gap-2 mb-2">
              <div className="min-w-0">
                <div className="text-xs font-semibold text-white/75 truncate">{agent.title}</div>
                <div className="text-[10px] text-white/25 truncate">{agent.id}</div>
              </div>
              <span className={`text-[10px] px-2 py-0.5 rounded-full ${
                agent.status === "complete"
                  ? "bg-emerald-400/10 text-emerald-400"
                  : ["pending", "error"].includes(agent.status)
                    ? "bg-white/10 text-white/40"
                    : "bg-violet-500/10 text-violet-300"
              }`}>
                {agent.status.replace(/-/g, " ")}
              </span>
            </div>

            {agent.queries.length > 0 && (
              <div className="mb-2">
                <div className="text-[10px] text-white/25 mb-1">Queries</div>
                <div className="space-y-1 max-h-20 overflow-y-auto scrollbar-hide">
                  {agent.queries.slice(0, 8).map((q, i) => (
                    <div key={i} className="text-[11px] text-white/40 truncate">• {q}</div>
                  ))}
                </div>
              </div>
            )}

            {agent.sources.length > 0 && (
              <div className="mb-2">
                <div className="text-[10px] text-white/25 mb-1">Web Links Used</div>
                <div className="space-y-1.5 max-h-28 overflow-y-auto scrollbar-hide">
                  {agent.sources.slice(0, 12).map((s, i) => (
                    <a
                      key={i}
                      href={s.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-[11px] text-white/45 hover:text-violet-300 transition-colors"
                    >
                      <StatusDot status="completed" />
                      <span className="truncate flex-1">{s.title || s.url}</span>
                    </a>
                  ))}
                </div>
              </div>
            )}

            <div className="flex items-center gap-3 pt-2 border-t border-white/[0.06]">
              <span className="text-[10px] text-white/25">Evidence: <span className="text-emerald-300">{agent.evidenceCount}</span></span>
              <span className="text-[10px] text-white/25">Report: <span className="text-blue-300">{(agent.reportLength / 1000).toFixed(1)}k</span></span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ================================================================== */
/* Sources Panel – vertical sidebar display of all gathered sources     */
/* ================================================================== */

function SourcesPanel({ sources }: { sources: GatheredSource[] }) {
  if (sources.length === 0) return null;

  const sortedSources = [...sources].sort((a, b) => b.score - a.score);
  const domainCounts: Record<string, number> = {};
  for (const s of sources) {
    try {
      const domain = new URL(s.url).hostname.replace("www.", "");
      domainCounts[domain] = (domainCounts[domain] || 0) + 1;
    } catch {
      /* skip */
    }
  }

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center gap-2.5 px-1">
        <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 flex items-center justify-center">
          <svg className="w-3 h-3 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
          </svg>
        </div>
        <div>
          <div className="text-xs font-semibold text-white/60">Sources Gathered</div>
          <div className="text-[10px] text-white/25">{sources.length} from {Object.keys(domainCounts).length} domains</div>
        </div>
      </div>

      {/* Domain tags */}
      <div className="flex flex-wrap gap-1">
        {Object.entries(domainCounts)
          .sort(([, a], [, b]) => b - a)
          .slice(0, 8)
          .map(([domain, count]) => (
            <span key={domain} className="text-[10px] px-2 py-0.5 rounded-full bg-white/[0.04] text-white/30 border border-white/[0.06]">
              {domain} <span className="text-white/50 font-medium">{count}</span>
            </span>
          ))}
      </div>

      {/* Source list */}
      <div className="space-y-1 max-h-[calc(100vh-400px)] overflow-y-auto scrollbar-hide">
        {sortedSources.map((source, i) => (
          <div key={i} className="flex items-center gap-2 py-1 animate-fade-in" style={{ animationDelay: `${i * 15}ms` }}>
            <QualityBar score={source.score} />
            <a
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[11px] text-white/40 hover:text-violet-400 truncate flex-1 transition-colors"
            >
              {source.title || (() => { try { return new URL(source.url).hostname; } catch { return source.url; } })()}
            </a>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ================================================================== */
/* Sidebar Activity Feed — real-time web accesses and agent activity    */
/* ================================================================== */

function ActivityFeed({ subagents, llmCalls }: { subagents: SubagentState[]; llmCalls: LLMCall[] }) {
  // Build a timeline of activity items
  const activities: { type: string; label: string; detail: string; status: string; time: number }[] = [];

  for (const agent of subagents) {
    for (const search of agent.searches) {
      activities.push({
        type: "search",
        label: agent.title,
        detail: search.query,
        status: search.status,
        time: Date.now(),
      });
    }
    for (const ext of agent.extractions) {
      let hostname = ext.url;
      try { hostname = new URL(ext.url).hostname.replace("www.", ""); } catch { /* skip */ }
      activities.push({
        type: "extract",
        label: agent.title,
        detail: hostname,
        status: ext.status,
        time: Date.now(),
      });
    }
  }

  const recent = activities.slice(-20).reverse();
  if (recent.length === 0) return null;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 px-1">
        <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-blue-500/20 to-cyan-500/20 flex items-center justify-center">
          <svg className="w-3 h-3 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
        <div className="text-xs font-semibold text-white/60">Live Activity</div>
      </div>

      <div className="space-y-1 max-h-60 overflow-y-auto scrollbar-hide">
        {recent.map((act, i) => (
          <div key={i} className="flex items-center gap-2 text-[11px] animate-fade-in">
            <StatusDot status={act.status} />
            <span className="text-white/20 flex-shrink-0 w-12 truncate">
              {act.type === "search" ? "Search" : "Extract"}
            </span>
            <span className="text-white/35 truncate flex-1">{act.detail}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ================================================================== */
/* Sidebar Progress — mini phase timeline                              */
/* ================================================================== */

function SidebarProgress({ steps, currentPhase }: { steps: { id: string; label: string; status: string }[]; currentPhase: string }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 px-1">
        <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-violet-500/20 to-indigo-500/20 flex items-center justify-center">
          <svg className="w-3 h-3 text-violet-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
          </svg>
        </div>
        <div className="text-xs font-semibold text-white/60">Research Progress</div>
      </div>

      <div className="space-y-0.5">
        {steps.filter(s => s.id !== "complete").map((step) => (
          <div key={step.id} className="flex items-center gap-2.5 px-1 py-1">
            <div className={`w-2 h-2 rounded-full flex-shrink-0 transition-all duration-500 ${
              step.status === "completed"
                ? "bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.3)]"
                : step.status === "active"
                  ? "bg-violet-400 animate-pulse shadow-[0_0_8px_rgba(139,92,246,0.4)]"
                  : "bg-white/10"
            }`} />
            <span className={`text-[11px] flex-1 ${
              step.status === "active" ? "text-violet-300 font-medium" :
              step.status === "completed" ? "text-white/45" : "text-white/20"
            }`}>
              {step.label}
            </span>
            {step.status === "active" && (
              <div className="w-3 h-3">
                <div className="w-3 h-3 rounded-full border border-violet-400/30 border-t-violet-400 animate-spin" />
              </div>
            )}
            {step.status === "completed" && (
              <svg className="w-3 h-3 text-emerald-400/60 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
              </svg>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

/* ================================================================== */
/* Agent Status Cards — compact cards for sidebar                      */
/* ================================================================== */

function SidebarAgents({ subagents }: { subagents: SubagentState[] }) {
  if (subagents.length === 0) return null;

  const statusColors: Record<string, string> = {
    pending: "text-white/25",
    "generating-queries": "text-blue-400",
    searching: "text-cyan-400",
    "evaluating-sources": "text-violet-400",
    "refining-queries": "text-amber-400",
    extracting: "text-orange-400",
    "writing-report": "text-indigo-400",
    complete: "text-emerald-400",
    error: "text-red-400",
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 px-1">
        <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-indigo-500/20 to-violet-500/20 flex items-center justify-center">
          <svg className="w-3 h-3 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </div>
        <div className="text-xs font-semibold text-white/60">
          Agents ({subagents.filter(a => a.status === "complete").length}/{subagents.length})
        </div>
      </div>

      <div className="space-y-1">
        {subagents.map((agent) => (
          <div key={agent.id} className="glass rounded-lg px-2.5 py-2 animate-fade-in">
            <div className="flex items-center gap-2">
              <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                agent.status === "complete" ? "bg-emerald-400" :
                agent.status === "pending" ? "bg-white/15" :
                "bg-violet-400 animate-pulse"
              }`} />
              <span className="text-[11px] text-white/60 truncate flex-1">{agent.title}</span>
              <span className={`text-[9px] font-medium ${statusColors[agent.status] || "text-white/30"}`}>
                {agent.status === "complete" ? "Done" : agent.status.replace(/-/g, " ")}
              </span>
            </div>
            {agent.sources.length > 0 && (
              <div className="mt-1 ml-3.5">
                <div className="text-[10px] text-white/20">{agent.sources.length} sources, {agent.evidenceCount} evidence</div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

/* ================================================================== */
/* LLM Call Ticker                                                     */
/* ================================================================== */

function LLMCallTicker({ calls }: { calls: LLMCall[] }) {
  const recent = calls.slice(-6);
  if (recent.length === 0) return null;

  return (
    <div className="space-y-1">
      {recent.map((c, i) => (
        <div key={i} className="flex items-center gap-2 text-[11px] animate-fade-in">
          <StatusDot status={c.status} />
          <span className="text-white/35 font-medium capitalize">
            {c.role.replace(/_/g, " ")}
          </span>
          <span className="text-white/15">→</span>
          <span className="text-white/25 font-mono text-[10px]">
            {c.model.split("/").pop()}
            <span className="text-white/12">@{c.provider}</span>
          </span>
          {c.status === "error" && c.error && (
            <span className="text-red-400/60 text-[10px] truncate max-w-[120px]">
              {c.error.slice(0, 40)}
            </span>
          )}
          {c.status === "completed" && c.outputLength && (
            <span className="text-emerald-400/40 text-[10px] font-mono">
              {(c.outputLength / 1000).toFixed(1)}k
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

/* ================================================================== */
/* Phase Content – extracted separately (avoids hydration issues)      */
/* ================================================================== */

interface PhaseState {
  planPreview: string;
  subtaskList: {
    id: string;
    title: string;
    description: string;
    objective: string;
  }[];
  scalingInfo: {
    complexity: string;
    subagent_count: number;
    target_sources: number;
    tool_calls_per_subagent?: number;
  } | null;
  subagents: SubagentState[];
  reflectionInfo: {
    decision: string;
    new_subtasks: { id: string; title: string }[];
    iteration: number;
    total_reports?: number;
    total_sources?: number;
  } | null;
  details: Record<string, string[]>;
  completionStats: {
    total_sources?: number;
    total_reports?: number;
    iterations?: number;
    provider?: string;
    model?: string;
  } | null;
  llmCalls: LLMCall[];
  warnings: string[];
  reportDraft: string;
}

function PhaseContent({
  step,
  state,
}: {
  step: { id: string; status: string };
  state: PhaseState;
}) {
  return (
    <div className="px-4 py-3 bg-white/[0.008] border-b border-white/[0.04] animate-slide-down space-y-3">
      {/* Plan */}
      {step.id === "plan" && state.planPreview && (
        <div className="glass rounded-xl p-3">
          <div className="text-[10px] uppercase tracking-wider text-white/25 mb-2 font-semibold flex items-center gap-1.5">
            <span className="text-violet-400/80">Plan</span>
          </div>
          <div className="prose prose-invert prose-sm max-w-none prose-headings:text-white/85 prose-p:text-white/70 prose-strong:text-white/85 prose-li:text-white/70">
            <MarkdownRenderer content={state.planPreview} />
          </div>
        </div>
      )}

      {/* Subtasks */}
      {step.id === "split" && state.subtaskList.length > 0 && (
        <div>
          <div className="text-[10px] uppercase tracking-wider text-white/25 mb-2 font-semibold">
            Research Subtasks ({state.subtaskList.length})
          </div>
          <div className="grid gap-1.5">
            {state.subtaskList.map((task) => (
              <div
                key={task.id}
                className="glass rounded-xl px-3 py-2 flex items-start gap-2"
              >
                <span className="text-[10px] font-mono text-violet-400/40 mt-0.5">
                  {task.id}
                </span>
                <div className="min-w-0">
                  <span className="text-xs font-medium text-white/70 block">
                    {task.title}
                  </span>
                  {task.description && (
                    <div className="text-[11px] text-white/25 mt-0.5 line-clamp-2">
                      {task.description}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Scaling */}
      {step.id === "scale" && state.scalingInfo && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {[
            {
              value: state.scalingInfo.complexity,
              label: "Complexity",
              gradient: true,
            },
            { value: state.scalingInfo.subagent_count, label: "Agents" },
            {
              value: state.scalingInfo.target_sources,
              label: "Target Sources",
            },
            ...(state.scalingInfo.tool_calls_per_subagent
              ? [
                  {
                    value: state.scalingInfo.tool_calls_per_subagent,
                    label: "Calls/Agent",
                  },
                ]
              : []),
          ].map((item, idx) => (
            <div key={idx} className="glass rounded-xl px-3 py-3 text-center">
              <div
                className={`text-lg font-bold capitalize ${
                  item.gradient ? "gradient-text" : "text-white/80"
                }`}
              >
                {item.value}
              </div>
              <div className="text-[10px] text-white/25 mt-0.5">
                {item.label}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Subagents */}
      {step.id === "subagents" && state.subagents.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-3">
            <div className="text-[10px] uppercase tracking-wider text-white/25 font-semibold">
              Parallel Agents
            </div>
            <div className="flex items-center gap-2">
              <div className="flex gap-0.5">
                {state.subagents.map((a) => (
                  <div
                    key={a.id}
                    className={`w-3 h-1.5 rounded-sm transition-all duration-500 ${
                      a.status === "complete"
                        ? "bg-emerald-400"
                        : a.status === "pending"
                          ? "bg-white/10"
                          : "bg-violet-400 animate-pulse"
                    }`}
                  />
                ))}
              </div>
              <span className="text-[10px] text-white/30 font-mono">
                {state.subagents.filter((a) => a.status === "complete").length}/
                {state.subagents.length}
              </span>
            </div>
          </div>
          <SubagentParallelBoard subagents={state.subagents} />
        </div>
      )}

      {/* Reflection */}
      {step.id === "reflection" && state.reflectionInfo && (
        <div className="glass rounded-xl p-3">
          <div className="flex items-center gap-2.5">
            <div
              className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                state.reflectionInfo.decision === "research-complete"
                  ? "bg-emerald-400/10"
                  : state.reflectionInfo.decision === "max-iterations-reached"
                    ? "bg-amber-400/10"
                    : "bg-violet-400/10"
              }`}
            >
              <span className="text-[10px] font-semibold text-white/70">
                {state.reflectionInfo.decision === "research-complete"
                  ? "Done"
                  : state.reflectionInfo.decision === "max-iterations-reached"
                    ? "Limit"
                    : "Review"}
              </span>
            </div>
            <div>
              <span className="text-xs font-medium text-white/70 block">
                {state.reflectionInfo.decision === "research-complete"
                  ? "Research quality sufficient"
                  : state.reflectionInfo.decision === "max-iterations-reached"
                    ? "Maximum iterations reached"
                    : `Identified ${state.reflectionInfo.new_subtasks.length} knowledge gaps`}
              </span>
              <span className="text-[10px] text-white/30">
                Iteration {state.reflectionInfo.iteration + 1}
                {state.reflectionInfo.total_reports &&
                  ` · ${state.reflectionInfo.total_reports} reports`}
                {state.reflectionInfo.total_sources &&
                  ` · ${state.reflectionInfo.total_sources} sources`}
              </span>
            </div>
          </div>
          {state.reflectionInfo.new_subtasks.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mt-2.5 pt-2.5 border-t border-white/[0.06]">
              {state.reflectionInfo.new_subtasks.map((s, i) => (
                <span
                  key={i}
                  className="text-[10px] px-2 py-0.5 rounded-md bg-amber-400/10 text-amber-400/80 border border-amber-400/15"
                >
                  + {s.title}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Synthesize / Cite – using <div> instead of <p> to avoid hydration error with StatusDot */}
      {(step.id === "synthesize" || step.id === "cite") &&
        state.details[step.id] && (
          <div className="glass rounded-xl p-3 space-y-1.5">
            {state.details[step.id].slice(-3).map((msg, i) => (
              <div
                key={i}
                className="text-xs text-white/40 flex items-center gap-2"
              >
                <StatusDot status="completed" />
                <span>{msg}</span>
              </div>
            ))}
          </div>
        )}

      {/* Complete stats */}
      {step.id === "complete" && state.completionStats && (
        <div className="grid grid-cols-3 gap-3">
          {[
            {
              value: state.completionStats.total_sources,
              label: "Sources Found",
              color: "text-emerald-400",
            },
            {
              value: state.completionStats.total_reports,
              label: "Reports Written",
              color: "text-blue-400",
            },
            {
              value: state.completionStats.iterations,
              label: "Iterations",
              color: "text-violet-400",
            },
          ]
            .filter((s) => s.value !== undefined)
            .map((stat, idx) => (
              <div key={idx} className="glass rounded-xl p-3 text-center">
                <div className={`text-2xl font-bold ${stat.color}`}>
                  {stat.value}
                </div>
                <div className="text-[10px] text-white/25 mt-0.5">
                  {stat.label}
                </div>
              </div>
            ))}
        </div>
      )}

      {/* LLM calls for this phase */}
      {(() => {
        const phaseRoles: Record<string, string[]> = {
          plan: ["planner"],
          split: ["splitter"],
          scale: ["scaler"],
          subagents: ["subagent", "evaluator"],
          reflection: ["reflection"],
          synthesize: ["coordinator"],
          cite: ["citation"],
        };
        const filtered = state.llmCalls.filter((c) =>
          (phaseRoles[step.id] || []).some((r) => c.role.includes(r))
        );
        return filtered.length > 0 ? (
          <div>
            <div className="text-[10px] uppercase tracking-wider text-white/20 mb-1.5 font-semibold flex items-center gap-1.5">
              <svg
                className="w-3 h-3 text-indigo-400/40"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                />
              </svg>
              AI Calls
            </div>
            <LLMCallTicker calls={filtered} />
          </div>
        ) : null;
      })()}

      {/* Warnings */}
      {state.warnings
        .filter((w) => w.startsWith(`[${step.id}]`))
        .map((w, i) => (
          <div
            key={i}
            className="text-[11px] text-amber-400/70 bg-amber-400/5 rounded-lg px-3 py-2 border border-amber-400/10"
          >
            Warning: {w}
          </div>
        ))}
    </div>
  );
}

/* ================================================================== */
/* Main SearchDisplay component                                        */
/* ================================================================== */

export function SearchDisplay({ events }: { events: ResearchEvent[] }) {
  const [expandedPhase, setExpandedPhase] = useState<string | null>(null);
  const [startTime] = useState(Date.now());
  const [elapsed, setElapsed] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  // Elapsed time ticker
  useEffect(() => {
    const interval = setInterval(
      () => setElapsed(Date.now() - startTime),
      1000
    );
    return () => clearInterval(interval);
  }, [startTime]);

  /* ── Compute derived state from events ─────────────────────────── */

  const state = useMemo(() => {
    const phaseStartOrder: string[] = [];
    let currentPhase = "";
    const details: Record<string, string[]> = {};
    const subagents: Record<string, SubagentState> = {};
    const llmCalls: LLMCall[] = [];
    const allSources: GatheredSource[] = [];
    let planPreview = "";
    let subtaskList: {
      id: string;
      title: string;
      description: string;
      objective: string;
    }[] = [];
    let scalingInfo: {
      complexity: string;
      subagent_count: number;
      target_sources: number;
      tool_calls_per_subagent?: number;
    } | null = null;
    let reflectionInfo: {
      decision: string;
      new_subtasks: { id: string; title: string }[];
      iteration: number;
      total_reports?: number;
      total_sources?: number;
    } | null = null;
    let errorInfo: { error: string; hint?: string; phase?: string } | null =
      null;
    let isComplete = false;
    let completionStats: {
      total_sources?: number;
      total_reports?: number;
      iterations?: number;
      provider?: string;
      model?: string;
    } | null = null;
    const warnings: string[] = [];
    let progressPercent = 0;
    let reportDraft = "";

    for (const evt of events) {
      /* eslint-disable @typescript-eslint/no-explicit-any */
      const e: any = evt;

      if (e.type === "phase-update") {
        if (!phaseStartOrder.includes(e.phase)) {
          phaseStartOrder.push(e.phase);
        }
        currentPhase = e.phase;
        if (!details[e.phase]) details[e.phase] = [];
        details[e.phase].push(e.message);
      }
      if (e.type === "thinking") {
        planPreview = e.message;
        if (!phaseStartOrder.includes("plan")) phaseStartOrder.push("plan");
      }
      if (e.type === "subtasks") {
        subtaskList =
          e.subtasks ||
          (e.titles || []).map((t: string, i: number) => ({
            id: `task_${i}`,
            title: t,
            description: "",
            objective: "",
          }));
        if (!phaseStartOrder.includes("split")) phaseStartOrder.push("split");
      }
      if (e.type === "scaling") {
        scalingInfo = e;
        if (!phaseStartOrder.includes("scale")) phaseStartOrder.push("scale");
      }
      if (e.type === "progress") {
        progressPercent = e.percent;
      }

      /* ── Subagent events ── */
      if (e.type === "subagents-launch") {
        if (!phaseStartOrder.includes("subagents"))
          phaseStartOrder.push("subagents");
        for (const agent of e.agent_details || []) {
          if (!subagents[agent.id]) {
            subagents[agent.id] = {
              id: agent.id,
              title: agent.title,
              description: agent.description || "",
              status: "pending",
              queries: [],
              searches: [],
              sources: [],
              extractions: [],
              evidenceCount: 0,
              reportLength: 0,
            };
          }
        }
      }
      if (e.type === "subagent-step") {
        const sa = subagents[e.subtask_id];
        if (sa) {
          sa.status = e.step;
          if (e.evidence_count !== undefined)
            sa.evidenceCount = e.evidence_count;
        }
      }
      if (e.type === "subagent-queries") {
        const sa = subagents[e.subtask_id];
        if (sa)
          sa.queries = [
            ...sa.queries,
            ...(e.queries || []).filter(
              (q: string) => !sa.queries.includes(q)
            ),
          ];
      }
      if (e.type === "subagent-search") {
        const sa = subagents[e.subtask_id];
        if (sa) {
          const existing = sa.searches.find((s) => s.query === e.query);
          if (existing) {
            existing.status = e.status;
            if (e.results_found !== undefined)
              existing.results_found = e.results_found;
          } else {
            sa.searches.push({
              query: e.query,
              status: e.status,
              results_found: e.results_found,
            });
          }
        }
      }
      if (e.type === "subagent-sources-scored") {
        const sa = subagents[e.subtask_id];
        if (sa) {
          sa.sources = e.top_scores || [];
          for (const src of e.top_scores || []) {
            if (!allSources.find((s) => s.url === src.url)) {
              allSources.push({
                url: src.url,
                title: src.title,
                score: src.score,
                subtask: e.subtask_title || "",
                timestamp: Date.now(),
              });
            }
          }
        }
      }
      if (e.type === "subagent-extract") {
        const sa = subagents[e.subtask_id];
        if (sa) {
          const existing = sa.extractions.find((x) => x.url === e.url);
          if (existing) existing.status = e.status;
          else sa.extractions.push({ url: e.url, status: e.status });
        }
      }
      if (e.type === "subagent-complete") {
        const sa = subagents[e.subtask_id];
        if (sa) {
          sa.status = "complete";
          sa.reportLength = e.report_length || 0;
          sa.evidenceCount = e.evidence_count || 0;
        }
      }
      if (e.type === "sources-found") {
        for (const src of e.sources || []) {
          if (!allSources.find((s) => s.url === src.url)) {
            allSources.push({
              url: src.url,
              title: src.title,
              score: src.quality_score || 0,
              subtask: "",
              timestamp: Date.now(),
            });
          }
        }
      }

      /* ── LLM calls ── */
      if (e.type === "llm-call") {
        if (e.status === "started") {
          llmCalls.push({
            model: e.model || "",
            provider: e.provider || "",
            role: e.role || "",
            status: "started",
            attempt: e.attempt || 1,
            timestamp: Date.now(),
          });
        } else {
          const existing = [...llmCalls]
            .reverse()
            .find((c) => c.role === e.role && c.status === "started");
          if (existing) {
            existing.status = e.status;
            existing.error = e.error;
            existing.outputLength = e.output_length;
          }
        }
      }

      /* ── Reflection ── */
      if (e.type === "reflection") {
        if (!phaseStartOrder.includes("reflection"))
          phaseStartOrder.push("reflection");
        reflectionInfo = {
          decision:
            e.decision ||
            (e.research_complete ? "research-complete" : "gaps-found"),
          new_subtasks: e.new_subtasks || [],
          iteration: e.iteration ?? 0,
          total_reports: e.total_reports,
          total_sources: e.total_sources,
        };
      }

      if (e.type === "report-draft") {
        if (!phaseStartOrder.includes("synthesize"))
          phaseStartOrder.push("synthesize");
        reportDraft = e.content || "";
      }
      if (e.type === "citations-added") {
        if (!phaseStartOrder.includes("cite")) phaseStartOrder.push("cite");
      }
      if (e.type === "warning")
        warnings.push(`[${e.phase}] ${e.message}`);
      if (e.type === "error")
        errorInfo = { error: e.error, hint: e.hint, phase: e.phase };
      if (e.type === "complete") {
        isComplete = true;
        progressPercent = 100;
        if (!phaseStartOrder.includes("complete"))
          phaseStartOrder.push("complete");
        completionStats = {
          total_sources: e.total_sources,
          total_reports: e.total_reports,
          iterations: e.iterations,
          provider: e.provider,
          model: e.model,
        };
      }
      /* eslint-enable @typescript-eslint/no-explicit-any */
    }

    /* ── Build steps in strict PHASE_ORDER ── */
    const seenSet = new Set(phaseStartOrder);
    const steps = PHASE_ORDER.map((phase) => ({
      id: phase,
      label: PHASE_LABELS[phase] || phase,
      status: (
        isComplete && phase !== "complete"
          ? "completed"
          : phase === "complete" && isComplete
            ? "completed"
            : phase === currentPhase
              ? "active"
              : seenSet.has(phase)
                ? "completed"
                : "pending"
      ) as "pending" | "active" | "completed",
    }));

    return {
      steps,
      details,
      subagents: Object.values(subagents),
      llmCalls,
      planPreview,
      subtaskList,
      scalingInfo,
      reflectionInfo,
      errorInfo,
      isComplete,
      completionStats,
      warnings,
      currentPhase,
      progressPercent,
      allSources,
      reportDraft,
    };
  }, [events]);

  // Auto-expand active phase
  useEffect(() => {
    const active = state.steps.find((s) => s.status === "active");
    if (active) setExpandedPhase(active.id);
  }, [state.steps]);

  // Auto-scroll into view
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    }
  }, [events.length]);

  const activeSteps = state.steps.filter((s) => s.status !== "pending");

  /* =================================================================
   * POST-REPORT LAYOUT — sidebar sources + main report process
   * ================================================================= */

  if (state.isComplete) {
    return (
      <div ref={containerRef} className="animate-fade-in">
        {/* Completion banner */}
        <div className="glass rounded-2xl px-5 py-4 border border-emerald-500/10 bg-emerald-500/[0.02] mb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-xl bg-emerald-400/15 flex items-center justify-center">
                <svg className="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <div>
                <span className="text-sm font-semibold text-white/80">Research Complete</span>
                <span className="text-[11px] text-white/30 ml-2">
                  {Math.floor(elapsed / 1000)}s
                  {state.completionStats?.provider && (
                    <> · <span className="capitalize">{state.completionStats.provider}</span></>
                  )}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-4">
              {state.completionStats && (
                <>
                  {state.completionStats.total_sources !== undefined && (
                    <div className="text-center">
                      <span className="text-lg font-bold text-emerald-400">{state.completionStats.total_sources}</span>
                      <span className="text-[10px] text-white/25 block">sources</span>
                    </div>
                  )}
                  {state.completionStats.total_reports !== undefined && (
                    <div className="text-center">
                      <span className="text-lg font-bold text-blue-400">{state.completionStats.total_reports}</span>
                      <span className="text-[10px] text-white/25 block">reports</span>
                    </div>
                  )}
                  {state.completionStats.iterations !== undefined && (
                    <div className="text-center">
                      <span className="text-lg font-bold text-violet-400">{state.completionStats.iterations}</span>
                      <span className="text-[10px] text-white/25 block">iterations</span>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>

        {/* Two-column layout: main process + sidebar sources */}
        <div className="flex gap-4">
          {/* Main content */}
          <div className="flex-1 min-w-0 space-y-3">
            {/* Collapsible research process */}
            <details className="glass rounded-2xl overflow-hidden group">
              <summary className="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-white/[0.015] transition-colors list-none">
                <svg className="w-4 h-4 text-white/30 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
                <span className="text-sm font-medium text-white/50">Research Process</span>
                <span className="text-[10px] text-white/20 font-mono">{activeSteps.length} steps</span>
                <div className="flex items-center gap-0.5 ml-auto mr-2">
                  {state.steps.filter(s => s.id !== "complete").map((step) => (
                    <div key={step.id} className="w-2 h-2 rounded-full bg-emerald-400/60" />
                  ))}
                </div>
              </summary>
              <div className="border-t border-white/[0.04]">
                {activeSteps.map((step) => (
                  <div key={step.id}>
                    <button
                      onClick={() => setExpandedPhase(expandedPhase === step.id ? null : step.id)}
                      className="w-full flex items-center gap-3 px-4 py-2.5 hover:bg-white/[0.015] transition-colors text-left border-b border-white/[0.04] last:border-b-0"
                    >
                      <span className="text-sm">{PHASE_ICONS[step.id] || "•"}</span>
                      <span className="text-xs text-white/50 flex-1">{step.label}</span>
                      <svg className={`w-3 h-3 text-white/15 transition-transform ${expandedPhase === step.id ? "rotate-180" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </button>
                    {expandedPhase === step.id && <PhaseContent step={step} state={state} />}
                  </div>
                ))}
              </div>
            </details>
          </div>

          {/* Sidebar: Sources */}
          {state.allSources.length > 0 && (
            <div className="hidden lg:block w-72 xl:w-80 flex-shrink-0 self-start">
              <div className="glass rounded-2xl p-4 sticky top-4 max-h-[calc(100vh-180px)] overflow-y-auto scrollbar-hide">
                <SourcesPanel sources={state.allSources} />
              </div>
            </div>
          )}
        </div>

        {/* Sources for mobile (below main) */}
        {state.allSources.length > 0 && (
          <div className="lg:hidden mt-3 glass rounded-2xl p-4">
            <SourcesPanel sources={state.allSources} />
          </div>
        )}
      </div>
    );
  }

  /* =================================================================
   * ACTIVE RESEARCH LAYOUT — sidebar with sources, progress, agents
   * ================================================================= */

  return (
    <div ref={containerRef} className="space-y-3">
      {/* Progress bar */}
      {events.length > 0 && (
        <ProgressBar
          percent={state.progressPercent}
          elapsed={elapsed}
          isComplete={state.isComplete}
          currentPhase={state.currentPhase}
        />
      )}

      {/* Error banner */}
      {state.errorInfo && (
        <div className="glass rounded-2xl px-4 py-3 animate-fade-in border border-red-500/20 bg-red-500/[0.03]">
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-xl bg-red-500/10 flex items-center justify-center flex-shrink-0">
              <svg className="w-4 h-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
            </div>
            <div>
              <div className="text-sm text-red-300 font-medium">{state.errorInfo.error}</div>
              {state.errorInfo.hint && <div className="text-xs text-white/30 mt-1">{state.errorInfo.hint}</div>}
            </div>
          </div>
        </div>
      )}

      {/* Two-column layout: phases + sidebar */}
      <div className="flex gap-4">
        {/* Main: Phase Timeline */}
        <div className="flex-1 min-w-0">
          <div className="glass rounded-2xl overflow-hidden">
            {/* Mini phase dots at top */}
            <div className="flex items-center gap-1 px-4 py-2.5 border-b border-white/[0.04]">
              {state.steps.filter(s => s.id !== "complete").map((step, i) => (
                <div key={step.id} className="flex items-center">
                  <div className={`w-2 h-2 rounded-full transition-all duration-500 ${
                    step.status === "completed" ? "bg-emerald-400" :
                    step.status === "active" ? "bg-violet-400 animate-pulse shadow-[0_0_8px_rgba(139,92,246,0.4)]" :
                    "bg-white/10"
                  }`} />
                  {i < state.steps.filter(s => s.id !== "complete").length - 1 && (
                    <div className={`w-6 h-px mx-0.5 transition-all duration-500 ${
                      step.status === "completed" ? "bg-emerald-400/30" : "bg-white/[0.06]"
                    }`} />
                  )}
                </div>
              ))}
              <span className="text-[10px] text-white/20 ml-auto font-mono tabular-nums">{activeSteps.length}/{state.steps.length - 1}</span>
            </div>

            {/* Expanded phases */}
            {activeSteps.map((step, i) => (
              <div key={step.id} className="animate-fade-in" style={{ animationDelay: `${i * 30}ms` }}>
                <div className="w-full flex items-center gap-3 px-4 py-3 border-b border-white/[0.04] last:border-b-0">
                  <PhaseIcon status={step.status} />
                  <span className={`text-sm font-medium flex-1 ${
                    step.status === "active" ? "text-violet-300" :
                    step.status === "completed" ? "text-white/60" : "text-white/30"
                  }`}>
                    {step.label}
                  </span>
                  {step.status === "active" && (
                    <span className="text-[10px] text-violet-400 bg-violet-500/10 px-2 py-0.5 rounded-full font-medium border border-violet-500/20 animate-pulse">
                      In Progress
                    </span>
                  )}
                </div>
                <PhaseContent step={step} state={state} />
              </div>
            ))}
          </div>
        </div>

        {/* Sidebar: Progress + Agents + Sources + Activity */}
        <div className="hidden lg:flex flex-col gap-3 w-72 xl:w-80 flex-shrink-0 self-start sticky top-4 max-h-[calc(100vh-180px)] overflow-y-auto scrollbar-hide pr-1">
          {/* Progress panel */}
          <div className="glass rounded-2xl p-4">
            <SidebarProgress steps={state.steps} currentPhase={state.currentPhase} />
          </div>

          {/* Agents panel */}
          {state.subagents.length > 0 && (
            <div className="glass rounded-2xl p-4">
              <SidebarAgents subagents={state.subagents} />
            </div>
          )}

          {/* Sources panel */}
          {state.allSources.length > 0 && (
            <div className="glass rounded-2xl p-4">
              <SourcesPanel sources={state.allSources} />
            </div>
          )}

          {/* Activity feed */}
          {state.subagents.some(a => a.searches.length > 0 || a.extractions.length > 0) && (
            <div className="glass rounded-2xl p-4">
              <ActivityFeed subagents={state.subagents} llmCalls={state.llmCalls} />
            </div>
          )}
        </div>
      </div>

      {/* Mobile: sources below */}
      {state.allSources.length > 0 && !state.isComplete && (
        <div className="lg:hidden glass rounded-2xl p-4">
          <SourcesPanel sources={state.allSources} />
        </div>
      )}

      {/* Active indicator */}
      {!state.isComplete && !state.errorInfo && events.length > 0 && (
        <div className="flex items-center justify-center gap-2.5 py-2">
          <div className="flex space-x-1">
            <div className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-bounce [animation-delay:0ms]" />
            <div className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-bounce [animation-delay:150ms]" />
            <div className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-bounce [animation-delay:300ms]" />
          </div>
          <span className="text-xs text-white/25">
            {state.subagents.filter(a => !["pending", "complete"].includes(a.status)).length > 0
              ? `${state.subagents.filter(a => !["pending", "complete"].includes(a.status)).length} agents researching...`
              : PHASE_LABELS[state.currentPhase] || "Processing..."}
          </span>
        </div>
      )}
    </div>
  );
}
