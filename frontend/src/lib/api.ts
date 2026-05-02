export interface Source {
  ticker: string;
  filing_type: string;
  period: string;
}

export interface QueryCallbacks {
  onSources: (sources: Source[]) => void;
  onText: (delta: string) => void;
  onDone: () => void;
  onError: (error: string) => void;
}

export async function queryFilings(
  question: string,
  callbacks: QueryCallbacks
): Promise<void> {
  const response = await fetch("http://localhost:8000/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    const text = await response.text();
    callbacks.onError(`Request failed: ${response.status} ${text}`);
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    callbacks.onError("No response body");
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const json = line.slice(6).trim();
      if (!json) continue;

      try {
        const event = JSON.parse(json) as
          | { type: "sources"; sources: Source[] }
          | { type: "text"; content: string }
          | { type: "done" };

        if (event.type === "sources") callbacks.onSources(event.sources);
        else if (event.type === "text") callbacks.onText(event.content);
        else if (event.type === "done") callbacks.onDone();
      } catch {
        // malformed SSE line — skip
      }
    }
  }
}
