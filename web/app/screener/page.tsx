import dynamic from "next/dynamic";

const ScreenerPanel = dynamic(() => import("@/components/ScreenerPanel"), { ssr: false });

export default function ScreenerPage() {
  return <ScreenerPanel />;
}
