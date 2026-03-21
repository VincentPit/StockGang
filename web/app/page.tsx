"use client";

import dynamic from "next/dynamic";

const TrainLoopPanel = dynamic(() => import("@/components/TrainLoopPanel"), {
  ssr: false,
});

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-950 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        <h1 className="text-2xl font-bold text-gray-100">MyQuant Dashboard</h1>
        <TrainLoopPanel />
      </div>
    </main>
  );
}
