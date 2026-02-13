"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ResearchChat } from "./components/research-chat";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export default function Home() {
  const [providerInfo, setProviderInfo] = useState<{ provider: string; model: string } | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<"checking" | "online" | "offline">("checking");
  const consecutiveFailuresRef = useRef(0);

  const handleModelChange = useCallback((provider: string, model: string) => {
    setProviderInfo((prev) => {
      if (prev && prev.provider === provider && prev.model === model) {
        return prev;
      }
      return { provider, model };
    });
  }, []);

  useEffect(() => {
    const fetchWithTimeout = async (url: string, timeoutMs = 3000) => {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), timeoutMs);
      try {
        return await fetch(url, { signal: controller.signal });
      } finally {
        clearTimeout(timeout);
      }
    };

    const checkStatus = async (): Promise<boolean> => {
      try {
        let cfg: any = null;
        let healthOk = false;

        // Check health directly first (independent of /config success).
        try {
          const healthResDirect = await fetchWithTimeout(`${BACKEND_URL}/api/health`);
          healthOk = healthResDirect.ok;
        } catch {
          /* fallback below */
        }

        if (!healthOk) {
          try {
            const healthRes = await fetchWithTimeout("/api/health");
            healthOk = healthRes.ok;
          } catch {
            /* offline fallback */
          }
        }

        // Load config independently; do not block online detection on config.
        try {
          const cfgResDirect = await fetchWithTimeout(`${BACKEND_URL}/api/config`);
          if (cfgResDirect.ok) cfg = await cfgResDirect.json();
        } catch {
          /* fallback below */
        }
        if (!cfg) {
          try {
            const cfgRes = await fetchWithTimeout("/api/config");
            if (cfgRes.ok) cfg = await cfgRes.json();
          } catch {
            /* ignore config errors */
          }
        }

        if (cfg) {
          setProviderInfo({ provider: cfg.default_provider || "", model: cfg.default_model || "" });
        }

        if (healthOk) {
          consecutiveFailuresRef.current = 0;
          setConnectionStatus("online");
          return true;
        } else {
          consecutiveFailuresRef.current += 1;
          if (consecutiveFailuresRef.current >= 2) {
            setConnectionStatus("offline");
          } else {
            setConnectionStatus("checking");
          }
          return false;
        }
      } catch {
        consecutiveFailuresRef.current += 1;
        if (consecutiveFailuresRef.current >= 2) {
          setConnectionStatus("offline");
        }
        return false;
      }
    };

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const loop = async () => {
      if (cancelled) return;
      const online = await checkStatus();
      const nextDelay = online ? 15000 : 2000;
      timer = setTimeout(loop, nextDelay);
    };

    void loop();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      {/* ── Glass Header ────────────────────────────────────── */}
      <header className="sticky top-0 z-50 glass border-b border-white/[0.06]">
        <div className="max-w-[96rem] mx-auto flex items-center justify-between px-5 py-3.5">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-violet-500/20">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            <div>
              <h1 className="text-base font-semibold tracking-tight text-white/90">Deep Research Agent</h1>
              <p className="text-[11px] text-white/40">Multi-agent orchestration</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-full glass text-[10px] sm:text-[11px] text-white/50">
              <div className={`w-1.5 h-1.5 rounded-full ${
                connectionStatus === "online"
                  ? "bg-emerald-400 animate-pulse"
                  : connectionStatus === "offline"
                    ? "bg-red-400"
                    : "bg-amber-400 animate-pulse"
              }`} />
              <span>
                {connectionStatus === "online"
                  ? "Online"
                  : connectionStatus === "offline"
                    ? "Offline"
                    : "Checking"}
              </span>
            </div>
            <div className="flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-full glass text-[10px] sm:text-[11px] text-white/50 max-w-[220px] sm:max-w-none">
              <span className="text-white/30">Model:</span>
              <span className="capitalize">{providerInfo?.provider || "-"}</span>
              <span className="text-white/20">|</span>
              <span className="text-white/40 max-w-[220px] truncate">{providerInfo?.model || "loading..."}</span>
            </div>
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-white/30 hover:text-white/60 transition-colors"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" className="w-5 h-5">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
              </svg>
            </a>
          </div>
        </div>
      </header>

      {/* ── Hero ────────────────────────────────────────────── */}
      <div className="px-5 pt-8 pb-4">
        <div className="max-w-[96rem] mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full glass text-[11px] text-white/50 mb-6 opacity-0 animate-fade-up [animation-delay:100ms] [animation-fill-mode:forwards]">
            <div className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" />
            Powered by multi-provider AI
          </div>
          <h2 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight opacity-0 animate-fade-up [animation-delay:200ms] [animation-fill-mode:forwards]">
            <span className="gradient-text">Deep Research</span>
          </h2>
          <h2 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight mt-1 opacity-0 animate-fade-up [animation-delay:350ms] [animation-fill-mode:forwards]">
            <span className="text-white/80">Agent</span>
          </h2>
          <p className="mt-5 text-base sm:text-lg text-white/40 max-w-xl mx-auto opacity-0 animate-fade-up [animation-delay:500ms] [animation-fill-mode:forwards]">
            Orchestrates parallel AI agents for comprehensive,<br className="hidden sm:block" />
            citation-backed research in real time.
          </p>
        </div>
      </div>

      {/* ── Main Content ────────────────────────────────────── */}
      <div className="flex-1 px-5 pb-10">
        <ResearchChat
          onModelChange={handleModelChange}
        />
      </div>

      {/* ── Glass Footer ────────────────────────────────────── */}
      <footer className="glass border-t border-white/[0.04]">
        <div className="max-w-[96rem] mx-auto px-5 py-5">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-2">
            <p className="text-[11px] text-white/30">
              Powered by{" "}
              <span className="text-white/50">Gemini</span>
              {" / "}
              <span className="text-white/50">OpenAI</span>
              {" / "}
              <span className="text-white/50">Claude</span>
              {" / "}
              <span className="text-white/50">HuggingFace</span>
              {" + "}
              <span className="text-white/50">Firecrawl</span>
              {" + "}
              <span className="text-white/50">LangGraph</span>
            </p>
            {providerInfo && (
              <p className="text-[11px] text-white/20">
                Active: <span className="capitalize text-white/40">{providerInfo.provider}</span>
              </p>
            )}
          </div>
        </div>
      </footer>
    </div>
  );
}
