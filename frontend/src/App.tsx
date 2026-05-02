import { useState, useCallback } from "react";
import { QueryInput } from "@/components/QueryInput";
import { AnswerPanel } from "@/components/AnswerPanel";
import { queryFilings, type Source } from "@/lib/api";

export default function App() {
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = useCallback(async (question: string) => {
    setAnswer("");
    setSources([]);
    setError(null);
    setIsLoading(true);
    setIsStreaming(false);

    await queryFilings(question, {
      onSources: (s) => {
        setSources(s);
        setIsLoading(false);
        setIsStreaming(true);
      },
      onText: (delta) => {
        setAnswer((prev) => prev + delta);
      },
      onDone: () => {
        setIsStreaming(false);
      },
      onError: (err) => {
        setError(err);
        setIsLoading(false);
        setIsStreaming(false);
      },
    });
  }, []);

  return (
    <div className="min-h-screen bg-white">
      <header className="border-b border-slate-200 px-6 py-4">
        <div className="max-w-3xl mx-auto flex items-center gap-3">
          <div className="w-7 h-7 rounded bg-[#0066CC]" />
          <div>
            <span className="text-base font-semibold text-slate-900 tracking-tight">
              Filing Intelligence
            </span>
            <span className="ml-2 text-xs text-slate-400">
              SEC EDGAR · 246 filings · 54 companies
            </span>
          </div>
        </div>
      </header>

      <main className="max-w-3xl mx-auto px-6 py-10 space-y-8">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900 mb-1">
            Ask a business question
          </h1>
          <p className="text-sm text-slate-500">
            Answers grounded in SEC 10-K and 10-Q filings from 2022–2026.
          </p>
        </div>

        <QueryInput onSubmit={handleSubmit} isLoading={isLoading || isStreaming} />

        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        {isLoading && !isStreaming && (
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <div className="w-4 h-4 border-2 border-slate-300 border-t-[#0066CC] rounded-full animate-spin" />
            Retrieving relevant filing excerpts...
          </div>
        )}

        <AnswerPanel answer={answer} sources={sources} isStreaming={isStreaming} />
      </main>
    </div>
  );
}
