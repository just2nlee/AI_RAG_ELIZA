import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

const EXAMPLES = [
  "What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?",
  "How has NVIDIA's revenue and growth outlook changed over the last two years?",
  "What regulatory risks do the major pharmaceutical companies face, and how are they addressing them?",
];

interface QueryInputProps {
  onSubmit: (question: string) => void;
  isLoading: boolean;
}

export function QueryInput({ onSubmit, isLoading }: QueryInputProps) {
  const [question, setQuestion] = useState("");

  const handleSubmit = () => {
    const trimmed = question.trim();
    if (!trimmed || isLoading) return;
    onSubmit(trimmed);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      handleSubmit();
    }
  };

  return (
    <div className="w-full space-y-4">
      <div className="relative">
        <Textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a business question about SEC filings..."
          className="min-h-[100px] resize-none text-base pr-4 border-slate-200 focus:border-blue-500 focus:ring-blue-500"
          disabled={isLoading}
        />
      </div>

      <div className="flex items-center justify-between">
        <Button
          onClick={handleSubmit}
          disabled={!question.trim() || isLoading}
          className="bg-[#0066CC] hover:bg-[#0052a3] text-white px-8"
        >
          {isLoading ? "Analyzing..." : "Analyze"}
        </Button>
        <span className="text-xs text-slate-400">⌘ + Enter to submit</span>
      </div>

      {!isLoading && !question && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-slate-500 uppercase tracking-wide">
            Example questions
          </p>
          <div className="flex flex-col gap-2">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                onClick={() => setQuestion(ex)}
                className="text-left text-sm text-slate-600 hover:text-[#0066CC] hover:bg-slate-50 rounded-md px-3 py-2 transition-colors border border-transparent hover:border-slate-200"
              >
                {ex}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
