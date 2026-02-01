import ChatUI from "./components/ChatUI";

export default function Home() {
  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 font-sans">
      <main className="mx-auto max-w-3xl px-4 py-8">
        <h1 className="mb-6 text-2xl font-semibold text-zinc-900 dark:text-zinc-50">
          Document Q&A
        </h1>
        <ChatUI />
      </main>
    </div>
  );
}
