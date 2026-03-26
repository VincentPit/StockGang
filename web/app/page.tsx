import dynamic from "next/dynamic";

const WorkflowPanel = dynamic(() => import("@/components/WorkflowPanel"), { ssr: false });

export default function WorkflowPage() {
  return <WorkflowPanel />;
}
