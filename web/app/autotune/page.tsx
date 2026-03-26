import dynamic from "next/dynamic";

const AutoTunePanel = dynamic(() => import("@/components/AutoTunePanel"), { ssr: false });

export default function AutoTunePage() {
  return <AutoTunePanel />;
}
