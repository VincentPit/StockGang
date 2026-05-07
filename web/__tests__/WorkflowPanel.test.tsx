import { render, screen } from "@testing-library/react";
import WorkflowPanel from "@/components/WorkflowPanel";

jest.mock("@/lib/api", () => ({
  startWorkflow: jest.fn(),
  getWorkflow:   jest.fn(),
  pollJob:       jest.fn(),
  listJobs:      jest.fn().mockResolvedValue([]),
}));

describe("WorkflowPanel", () => {
  it("renders without crashing", () => {
    const { container } = render(<WorkflowPanel />);
    expect(container.firstChild).toBeInTheDocument();
  });
});
