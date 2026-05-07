import { fireEvent, render, screen } from "@testing-library/react";
import { ModelExplainerCard } from "@/components/ModelExplainerCard";
import type { StoredModelInfo } from "@/lib/api";

const baseModel: StoredModelInfo = {
  model_id:      "sh600519_lgbm_core",
  symbol:        "sh600519",
  strategy_id:   "lgbm_core",
  trained_at:    1_700_000_000,
  bar_count:     504,
  last_bar_date: "2025-01-02",
  oos_accuracy:  0.617,
  feature_cols:  ["rsi_14", "macd_hist", "vol_20"],
};

describe("ModelExplainerCard", () => {
  it("renders model meta", () => {
    render(<ModelExplainerCard model={baseModel} />);
    expect(screen.getByText("sh600519")).toBeInTheDocument();
    expect(screen.getByText("61.7%")).toBeInTheDocument();
    expect(screen.getByText("504")).toBeInTheDocument();
    expect(screen.getByText("2025-01-02")).toBeInTheDocument();
  });

  it("invokes onAnalyse and onDelete callbacks", () => {
    const onAnalyse = jest.fn();
    const onDelete  = jest.fn();
    render(<ModelExplainerCard model={baseModel} onAnalyse={onAnalyse} onDelete={onDelete} />);
    fireEvent.click(screen.getByTitle("Analyse this symbol"));
    fireEvent.click(screen.getByTitle("Delete model"));
    expect(onAnalyse).toHaveBeenCalledWith("sh600519");
    expect(onDelete).toHaveBeenCalledWith("sh600519");
  });

  it("renders the top-10 feature importance list when supplied", () => {
    const features = Array.from({ length: 12 }, (_, i) => ({
      feature: `f_${i}`,
      importance: 12 - i,
    }));
    render(<ModelExplainerCard model={baseModel} features={features} />);
    expect(screen.getByText("Feature Importance")).toBeInTheDocument();
    expect(screen.getByText("f_0")).toBeInTheDocument();   // top
    expect(screen.queryByText("f_11")).toBeNull();         // truncated
  });

  it("handles missing oos_accuracy", () => {
    const m = { ...baseModel, oos_accuracy: undefined };
    render(<ModelExplainerCard model={m} />);
    // Two em-dashes (one for OOS Acc, but Trained is hydrated), assert at least one
    expect(screen.getAllByText("—").length).toBeGreaterThanOrEqual(1);
  });
});
