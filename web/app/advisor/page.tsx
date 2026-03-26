"use client";

import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import dynamic from "next/dynamic";

const AdvisorPanel = dynamic(() => import("@/components/AdvisorPanel"), { ssr: false });

function AdvisorContent() {
  const params = useSearchParams();
  const symbol = params.get("symbol") ?? undefined;
  return <AdvisorPanel initialSymbol={symbol} />;
}

export default function AdvisorPage() {
  return (
    <Suspense fallback={<div className="text-gray-600 text-sm">Loading…</div>}>
      <AdvisorContent />
    </Suspense>
  );
}
