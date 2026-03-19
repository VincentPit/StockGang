"""
Sentiment Analyzer — multi-layer Chinese financial text scoring.

Architecture:
  ┌───────────────────────────────────────────────────────────────────────┐
  │ Layer 1 — SnowNLP baseline:  [0, 1] general sentiment probability    │
  │ Layer 2 — Keyword dictionary: financial terms with signed weights     │
  │ Layer 3 — Negation detection: "不/未/无/难以" within 5-char window   │
  │                                                                       │
  │ Final score = weighted blend of layers → normalized to [-1, +1]      │
  │   -1 = very bearish   0 = neutral   +1 = very bullish                │
  └───────────────────────────────────────────────────────────────────────┘

The keyword dictionaries cover:
  • Earnings & fundamentals (业绩超预期, 净利润增长, 亏损, …)
  • Market structure      (涨停, 跌停, 北向资金, 资金流入/流出, …)
  • Policy & macro        (降息, 降准, 政策利好/利空, 贸易摩擦, …)
  • Risk events           (退市, 债务违约, 制裁, 调查, …)

Usage:
    analyzer = SentimentAnalyzer()
    score    = analyzer.analyze("三季度净利润增长45%，超预期")
    # SentimentScore(raw_score≈0.72, label='bullish')

    # Batch aggregation (e.g., multiple news headlines)
    combined = analyzer.analyze_batch([item.full_text for item in news_items])
"""
from __future__ import annotations

from dataclasses import dataclass, field

from myquant.config.logging_config import get_logger

logger = get_logger(__name__)


# ── Keyword dictionaries ─────────────────────────────────────────────────────

_POS: dict[str, float] = {
    # Earnings & fundamentals
    "净利润增长":   0.80,
    "业绩超预期":   0.90,
    "收入增长":     0.70,
    "利润大增":     0.80,
    "超预期":       0.75,
    "业绩增长":     0.75,
    "高增长":       0.70,
    "盈利提升":     0.70,
    "创历史新高":   0.90,
    "营收增长":     0.70,
    "分红":         0.50,
    "股息":         0.45,
    "回购":         0.60,
    "增持":         0.65,
    "买入":         0.55,
    "强烈推荐":     0.85,
    "目标价上调":   0.80,
    "上调评级":     0.80,
    # Market structure
    "涨停":         0.85,
    "大涨":         0.70,
    "反弹":         0.50,
    "突破":         0.60,
    "资金流入":     0.65,
    "北向资金买入": 0.80,
    "外资增持":     0.70,
    "订单增加":     0.60,
    "需求旺盛":     0.60,
    "扩产":         0.50,
    "市占率提升":   0.65,
    # Policy & macro
    "政策利好":     0.80,
    "降息":         0.60,
    "降准":         0.70,
    "财政刺激":     0.60,
    "利好":         0.65,
    "提振":         0.55,
    "宽松":         0.50,
    "减税":         0.55,
    # Sector / theme
    "AI需求旺盛":   0.75,
    "芯片国产化":   0.65,
    "新能源利好":   0.60,
    "消费复苏":     0.65,
}

_NEG: dict[str, float] = {
    # Earnings & fundamentals
    "净利润下降":   0.80,
    "业绩低于预期": 0.90,
    "亏损":         0.80,
    "收入下滑":     0.70,
    "盈利下降":     0.80,
    "业绩暴雷":     1.00,
    "商誉减值":     0.75,
    "减值":         0.55,
    "目标价下调":   0.75,
    "下调评级":     0.80,
    "减持":         0.60,
    "卖出":         0.55,
    # Market structure
    "跌停":         0.90,
    "大跌":         0.80,
    "下跌":         0.50,
    "资金流出":     0.70,
    "外资减持":     0.70,
    "北向资金卖出": 0.80,
    # Macro / policy
    "加息":         0.65,
    "收紧":         0.50,
    "政策收紧":     0.70,
    "利空":         0.70,
    "通胀上升":     0.55,
    "流动性收紧":   0.65,
    "汇率贬值":     0.55,
    # Risk events
    "退市":         1.00,
    "诉讼":         0.60,
    "调查":         0.70,
    "违规":         0.80,
    "债务违约":     1.00,
    "流动性危机":   0.90,
    "信用降级":     0.75,
    "贸易摩擦":     0.65,
    "制裁":         0.70,
    "欺诈":         0.90,
    "造假":         0.95,
}

_NEGATIONS = {"不", "没有", "未", "无", "非", "否", "难以", "缺乏", "并非", "绝非"}


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class SentimentScore:
    raw_score:     float          # [-1, +1]
    snownlp_score: float          # [0, 1] baseline (0.5 = neutral)
    keyword_hits:  list[str] = field(default_factory=list)
    confidence:    float = 0.50

    @property
    def label(self) -> str:
        if self.raw_score >  0.25:
            return "bullish"
        if self.raw_score < -0.25:
            return "bearish"
        return "neutral"

    @property
    def is_strong(self) -> bool:
        return abs(self.raw_score) > 0.45 and self.confidence > 0.55


# ── Analyzer ─────────────────────────────────────────────────────────────────

class SentimentAnalyzer:
    """
    Multi-layer Chinese financial sentiment scorer.

    Parameters
    ----------
    snownlp_weight : Weight of SnowNLP baseline in final blend (default: 0.30).
    keyword_weight : Weight of keyword score in final blend (default: 0.70).
    """

    def __init__(
        self,
        snownlp_weight: float = 0.30,
        keyword_weight: float = 0.70,
    ) -> None:
        self._sw = snownlp_weight
        self._kw = keyword_weight
        self._has_snow = self._check_snownlp()

    # ── Public API ────────────────────────────────────────────────────────

    def analyze(self, text: str) -> SentimentScore:
        """Score a single piece of financial text."""
        if not text or not text.strip():
            return SentimentScore(raw_score=0.0, snownlp_score=0.5)

        snow       = self._snow_score(text)
        kw_val, hits = self._kw_score(text)

        # SnowNLP [0,1] → [-1,+1] centered
        snow_centered = (snow - 0.5) * 2.0
        raw = snow_centered * self._sw + kw_val * self._kw
        raw = max(-1.0, min(1.0, raw))

        # Confidence: boosted by keyword evidence
        if hits:
            conf = min(0.95, 0.50 + len(hits) * 0.10)
        elif self._has_snow:
            conf = 0.40 + abs(snow_centered) * 0.30
        else:
            conf = 0.35  # no evidence at all

        return SentimentScore(
            raw_score     = raw,
            snownlp_score = snow,
            keyword_hits  = hits,
            confidence    = conf,
        )

    def analyze_batch(self, texts: list[str]) -> SentimentScore:
        """Average sentiment across a list of texts (e.g. multiple headlines)."""
        if not texts:
            return SentimentScore(raw_score=0.0, snownlp_score=0.5)
        scores = [self.analyze(t) for t in texts if t.strip()]
        if not scores:
            return SentimentScore(raw_score=0.0, snownlp_score=0.5)

        n         = len(scores)
        avg_raw   = sum(s.raw_score     for s in scores) / n
        avg_snow  = sum(s.snownlp_score for s in scores) / n
        avg_conf  = sum(s.confidence    for s in scores) / n
        all_hits  = [h for s in scores for h in s.keyword_hits]
        return SentimentScore(avg_raw, avg_snow, all_hits, avg_conf)

    # ── Internals ─────────────────────────────────────────────────────────

    def _snow_score(self, text: str) -> float:
        if not self._has_snow:
            return 0.5
        try:
            from snownlp import SnowNLP
            return float(SnowNLP(text).sentiments)
        except Exception:
            return 0.5

    def _kw_score(self, text: str) -> tuple[float, list[str]]:
        pos_total = 0.0
        neg_total = 0.0
        hits: list[str] = []

        for kw, w in _POS.items():
            if kw in text:
                idx    = text.find(kw)
                prefix = text[max(0, idx - 6): idx]
                if any(neg in prefix for neg in _NEGATIONS):
                    neg_total += w * 0.5  # negated positive → mild negative
                else:
                    pos_total += w
                    hits.append(f"+{kw}")

        for kw, w in _NEG.items():
            if kw in text:
                idx    = text.find(kw)
                prefix = text[max(0, idx - 6): idx]
                if any(neg in prefix for neg in _NEGATIONS):
                    pos_total += w * 0.3  # negated negative → mild positive
                else:
                    neg_total += w
                    hits.append(f"-{kw}")

        total = pos_total + neg_total
        if total == 0.0:
            return 0.0, hits

        score = (pos_total - neg_total) / total
        return float(score), hits

    @staticmethod
    def _check_snownlp() -> bool:
        try:
            import snownlp  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "snownlp not installed — sentiment will rely on keywords only. "
                "Run: pip install snownlp"
            )
            return False
