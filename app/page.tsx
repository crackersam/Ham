"use client";

import dynamic from "next/dynamic";

const MakeupStudio = dynamic(() => import("./components/MakeupStudio"), {
  ssr: false,
  loading: () => (
    <div className="mx-auto flex min-h-[75vh] w-full max-w-6xl items-center justify-center px-6">
      <div className="rounded-2xl border border-fuchsia-100 bg-white/80 px-6 py-5 shadow-sm backdrop-blur">
        <p className="text-sm font-medium text-fuchsia-700">Preparing virtual studio...</p>
      </div>
    </div>
  ),
});

export default function Home() {
  return <MakeupStudio />;
}
