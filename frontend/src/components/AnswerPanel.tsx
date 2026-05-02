import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { Badge } from "@/components/ui/badge";
import { ChevronDown, ChevronUp } from "lucide-react";
import type { Source } from "@/lib/api";

interface AnswerPanelProps {
  answer: string;
  sources: Source[];
  isStreaming: boolean;
}

export function AnswerPanel({ answer, sources, isStreaming }: AnswerPanelProps) {
  const [sourcesOpen, setSourcesOpen] = useState(false);

  if (!answer && !isStreaming) return null;

  return (
    <div className="w-full space-y-4">
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="max-w-none text-sm leading-relaxed">
          <ReactMarkdown>{answer}</ReactMarkdown>
          {isStreaming && (
            <span className="inline-block w-2 h-4 bg-[#0066CC] ml-0.5 animate-pulse" />
          )}
        </div>
      </div>

      {sources.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-slate-50 overflow-hidden">
          <button
            onClick={() => setSourcesOpen((o) => !o)}
            className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition-colors"
          >
            <span>
              Sources ({sources.length} filing
              {sources.length !== 1 ? "s" : ""})
            </span>
            {sourcesOpen ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </button>
          {sourcesOpen && (
            <div className="px-4 pb-4 flex flex-wrap gap-2">
              {sources.map((s, i) => (
                <Badge
                  key={i}
                  variant="secondary"
                  className="font-mono text-xs bg-white border border-slate-200 text-slate-700"
                >
                  {s.ticker} · {s.filing_type} · {s.period}
                </Badge>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
