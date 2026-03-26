import dynamic from "next/dynamic";

const TrainLoopPanel = dynamic(() => import("@/components/TrainLoopPanel"), { ssr: false });

export default function TrainLoopPage() {
  return <TrainLoopPanel />;
}
