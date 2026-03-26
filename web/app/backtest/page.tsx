import dynamic from "next/dynamic";

const BacktestPanel = dynamic(() => import("@/components/BacktestPanel"), { ssr: false });

export default function BacktestPage() {
  return <BacktestPanel />;
}
