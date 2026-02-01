"use client";

import { useState, useRef, useEffect } from "react";

const DOCUMENT_RETRIEVER = "Document_retriever";
const CHUNK_RETRIEVER = "Chunk_retriever";
const GENERATOR = "Generator";

const TOP_K = 10;

/** doc_id e.g. "2005.11401v4" → https://arxiv.org/pdf/2005.11401 */
function arxivPdfUrl(docId: string): string {
  const base = docId.replace(/v\d+$/i, "");
  return `https://arxiv.org/pdf/${base}`;
}

type DocWithScore = { doc_id: string; rerank_score: number };
type Message = {
  role: "user" | "assistant";
  content: string;
  answer?: string;
  documentIds?: string[];
};

/** Call Supabase Edge Function via Next.js API proxy (avoids CORS / "Failed to fetch"). */
async function invokeFunction<T>(
  name: string,
  body: Record<string, unknown>
): Promise<{ ok: true; data: T } | { ok: false; error: string }> {
  const res = await fetch("/api/invoke", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ functionName: name, body }),
  });
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const errMsg =
      (data && typeof data === "object" && "error" in data ? (data as { error: string }).error : null) ??
      res.statusText ??
      "Request failed";
    return { ok: false, error: `${errMsg} (${res.status})` };
  }
  return { ok: true, data: data as T };
}

export default function ChatUI() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const query = input.trim();
    if (!query || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setLoading(true);
    setError(null);

    try {
      // 1️⃣ Document retriever → doc_ids + rerank scores
      const docRes = await invokeFunction<DocWithScore[]>(DOCUMENT_RETRIEVER, {
        query,
      });
      if (!docRes.ok) {
        setError(docRes.error);
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `Error: ${docRes.error}` },
        ]);
        return;
      }

      const docs = docRes.data ?? [];
      const docIds = docs.map((d) => d.doc_id).filter(Boolean);

      if (docIds.length === 0) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: "No relevant documents found.",
            answer: "No relevant documents were retrieved for your query.",
            documentIds: [],
          },
        ]);
        return;
      }

      // Transform document retriever output → chunk retriever input: { doc_id, rerank_score } → { doc_id, document_score }
      const docsForChunkRetriever = docs.map((d) => ({
        doc_id: d.doc_id,
        document_score: d.rerank_score,
      }));

      // 2️⃣ Chunk retriever → context (chunks)
      const chunkRes = await invokeFunction<{ context: { content: string }[]; query: string }>(
        CHUNK_RETRIEVER,
        { docs: docsForChunkRetriever, query, top_k: TOP_K }
      );
      console.log("2. Chunk retriever raw response:", chunkRes);

      if (!chunkRes.ok) {
        setError(chunkRes.error);
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `Error: ${chunkRes.error}`, documentIds: docIds },
        ]);
        return;
      }

      const rawData = chunkRes.data as unknown;
      const context = Array.isArray((rawData as { context?: unknown }).context)
        ? (rawData as { context: { content: string }[] }).context
        : (rawData as { context?: { content: string }[] }).context ?? [];
      const generatorPayload = { context, query };
      console.log("Exact payload passed to generator:", JSON.stringify(generatorPayload, null, 2));

      // 3️⃣ Generator → answer
      const genRes = await invokeFunction<{ answer: string }>(GENERATOR, generatorPayload);
      if (!genRes.ok) {
        setError(genRes.error);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Error: ${genRes.error}`,
            documentIds: docIds,
          },
        ]);
        return;
      }

      const answer = genRes.data?.answer ?? "No answer generated.";

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: answer,
          answer,
          documentIds: docIds,
        },
      ]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${msg}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex h-[calc(100vh-8rem)] flex-col rounded-xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <p className="text-center text-zinc-500 dark:text-zinc-400 text-sm">
            Ask a question about your documents.
          </p>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[85%] rounded-2xl px-4 py-2.5 ${
                m.role === "user"
                  ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
                  : "bg-zinc-100 text-zinc-900 dark:bg-zinc-800 dark:text-zinc-100"
              }`}
            >
              <p className="whitespace-pre-wrap text-sm">{m.content}</p>
              {m.role === "assistant" && m.documentIds && m.documentIds.length > 0 && (
                <div className="mt-3 border-t border-zinc-200 dark:border-zinc-700 pt-3">
                  <p className="text-xs font-medium text-zinc-600 dark:text-zinc-400 mb-1.5">
                    Documents used:
                  </p>
                  <ul className="flex flex-wrap gap-1.5">
                    {m.documentIds.map((id, j) => (
                      <li key={j}>
                        <a
                          href={arxivPdfUrl(id)}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="rounded bg-white/80 px-2 py-0.5 text-xs font-mono text-zinc-700 underline hover:text-zinc-900 dark:bg-zinc-900/80 dark:text-zinc-300 dark:hover:text-zinc-100"
                        >
                          {id}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="rounded-2xl bg-zinc-100 px-4 py-2.5 dark:bg-zinc-800">
              <span className="text-sm text-zinc-500">Searching and generating…</span>
            </div>
          </div>
        )}
        {error && (
          <p className="text-center text-sm text-red-600 dark:text-red-400">{error}</p>
        )}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={handleSubmit} className="border-t border-zinc-200 p-4 dark:border-zinc-800">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question..."
            className="flex-1 rounded-xl border border-zinc-300 bg-white px-4 py-2.5 text-sm outline-none focus:border-zinc-500 focus:ring-1 focus:ring-zinc-500 dark:border-zinc-700 dark:bg-zinc-900 dark:focus:border-zinc-500"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="rounded-xl bg-zinc-900 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-zinc-800 disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
