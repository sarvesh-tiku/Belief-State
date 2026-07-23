"""Data and temporal alignment (Section 3.1).

For each firm and reporting period we construct a temporally aligned tuple

    (historical market context, earnings-call transcript, realized behavior)

where the historical context precedes the call, the transcript is the
forward-looking disclosure, and the realized behavior (post-call volatility /
price movement) is used *only* for alignment and analysis -- never as a label
or training target (paper, Section 3.1).

Two sources are supported:

    * ``synthetic`` -- a self-contained generator that produces price paths with
      controllable volatility *regimes* (low / high) and earnings transcripts
      whose guidance/uncertainty language is drawn from regime-conditioned
      templates.  This makes the controlled and null contrasts of Section 4
      exact and reproducible without any downloads.

    * ``local`` -- reads user-supplied ``data/raw/<ticker>_<period>.json`` files
      with fields ``prices`` (list of daily closes spanning lookback+horizon),
      ``call_date_index`` (split point), and ``transcript`` (string).

The synthetic generator is intentionally realistic in structure: returns show
volatility clustering, regimes differ in both realized variance and in the
uncertainty tone of the disclosure, and there is a controllable degree of
misalignment between disclosure tone and recent market behavior (the stress
condition of Section 2.4).
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Transcript language banks (regime- and uncertainty-conditioned).
# These stand in for real earnings-call disclosures; each phrase carries the
# forward-looking guidance / uncertainty content that the belief-state should
# be sensitive to, while the surrounding boilerplate is shared across regimes
# so that *surface* form is matched (important for the null contrasts).
# ---------------------------------------------------------------------------

_BOILERPLATE_OPEN = [
    "Thank you all for joining today's call.",
    "Good afternoon, and welcome to our quarterly earnings call.",
    "We appreciate everyone taking the time to join us today.",
]

_BOILERPLATE_CLOSE = [
    "With that, we will open the line for questions.",
    "We look forward to updating you next quarter.",
    "Thank you, and we will now take your questions.",
]

# Guidance / uncertainty phrases indexed by (regime, uncertainty level).
# uncertainty level 0 = confident, 1 = hedged, 2 = highly uncertain.
_GUIDANCE = {
    "low_vol": [
        [
            "Demand trends remain stable and we are reaffirming full-year guidance.",
            "Our pipeline is healthy and margins are holding steady.",
        ],
        [
            "Conditions are broadly stable, though we are monitoring input costs.",
            "We expect performance to remain within our prior range.",
        ],
        [
            "While the backdrop is calm, we note some uncertainty in the coming quarters.",
            "We are maintaining guidance but flag a few watch items.",
        ],
    ],
    "high_vol": [
        [
            "Despite market turbulence, we are confident in our operating plan.",
            "We see volatility as transient and reaffirm our outlook.",
        ],
        [
            "The environment is choppy and we are widening our guidance range.",
            "Near-term visibility is limited given recent market swings.",
        ],
        [
            "Conditions are highly uncertain and we are withdrawing formal guidance.",
            "We caution that downside risks have materially increased.",
        ],
    ],
}


@dataclass
class AlignedTuple:
    """One temporally aligned firm/period record (Section 3.1)."""

    firm_id: int
    ticker: str
    sector: str
    period: int
    regime: str                     # ground-truth volatility regime label
    uncertainty: int                # 0/1/2 disclosure uncertainty level
    prices: np.ndarray              # daily closes, length lookback + horizon
    call_index: int                 # split between historical and realized
    transcript: str
    # Derived market summaries used as the "historical context" the agent sees.
    hist_returns: np.ndarray = field(default_factory=lambda: np.empty(0))
    hist_vol: float = 0.0
    realized_vol: float = 0.0
    realized_move: float = 0.0

    @property
    def historical_prices(self) -> np.ndarray:
        return self.prices[: self.call_index + 1]

    @property
    def realized_prices(self) -> np.ndarray:
        return self.prices[self.call_index + 1 :]

    def market_context_text(self) -> str:
        """A compact textual rendering of the historical market context.

        This is the numeric time-series signal (Section 2.1: returns and
        volatility over a fixed lookback window) rendered so a language agent
        can consume it alongside the transcript.
        """

        ann_vol = self.hist_vol * np.sqrt(252)
        cum = float(self.historical_prices[-1] / self.historical_prices[0] - 1.0)
        trend = "up" if cum > 0.02 else ("down" if cum < -0.02 else "flat")
        return (
            f"Historical market context for {self.ticker} ({self.sector}) over the "
            f"trailing {len(self.historical_prices) - 1} sessions: cumulative return "
            f"{cum * 100:+.1f}%, annualized volatility {ann_vol * 100:.1f}%, "
            f"trend {trend}."
        )


class DataModule:
    """Builds / loads the corpus of aligned tuples."""

    def __init__(self, cfg, rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.seed)

    # -- public API ---------------------------------------------------------

    def build(self) -> List[AlignedTuple]:
        source = self.cfg.data.source
        if source == "synthetic":
            corpus = self._build_synthetic()
        elif source == "local":
            corpus = self._load_local()
        else:  # pragma: no cover - guarded by config
            raise ValueError(f"unknown data.source: {source}")
        for rec in corpus:
            self._annotate(rec)
        return corpus

    # -- synthetic generator ------------------------------------------------

    def _build_synthetic(self) -> List[AlignedTuple]:
        d = self.cfg.data
        sectors = list(d.sectors)
        regimes = list(d.regimes)
        lookback = int(d.lookback_days)
        horizon = int(d.horizon_days)
        corpus: List[AlignedTuple] = []

        for firm in range(int(d.n_firms)):
            sector = sectors[firm % len(sectors)]
            ticker = f"{sector[:3].upper()}{firm:02d}"
            base_price = float(self.rng.uniform(40.0, 320.0))
            for period in range(int(d.periods_per_firm)):
                # Assign a regime; alternate to balance the split across periods.
                regime = regimes[(firm + period) % len(regimes)]
                uncertainty = int(self.rng.integers(0, 3))
                prices, call_index = self._simulate_prices(
                    base_price, lookback, horizon, regime
                )
                base_price = float(prices[call_index])  # carry forward
                transcript = self._synth_transcript(regime, uncertainty)
                corpus.append(
                    AlignedTuple(
                        firm_id=firm,
                        ticker=ticker,
                        sector=sector,
                        period=period,
                        regime=regime,
                        uncertainty=uncertainty,
                        prices=prices,
                        call_index=call_index,
                        transcript=transcript,
                    )
                )
        return corpus

    def _simulate_prices(
        self, start: float, lookback: int, horizon: int, regime: str
    ) -> tuple[np.ndarray, int]:
        """GARCH-flavoured price path with regime-dependent volatility.

        Low-vol regimes have small, weakly-clustered daily variance; high-vol
        regimes have larger variance with stronger clustering (volatility
        clustering, Section 2.4).
        """

        n = lookback + horizon + 1
        if regime == "low_vol":
            base_sigma, cluster, drift = 0.008, 0.15, 0.0004
        else:
            base_sigma, cluster, drift = 0.026, 0.55, -0.0002

        sigma = base_sigma
        rets = np.empty(n - 1)
        for t in range(n - 1):
            shock = self.rng.normal(0.0, sigma)
            rets[t] = drift + shock
            # AR(1)-style variance update -> volatility clustering.
            sigma = np.sqrt(
                (1 - cluster) * base_sigma**2 + cluster * shock**2 + 1e-8
            )
        prices = start * np.exp(np.concatenate([[0.0], np.cumsum(rets)]))
        call_index = lookback  # the call happens right after the lookback window
        return prices, call_index

    def _synth_transcript(self, regime: str, uncertainty: int) -> str:
        opener = _BOILERPLATE_OPEN[self.rng.integers(len(_BOILERPLATE_OPEN))]
        closer = _BOILERPLATE_CLOSE[self.rng.integers(len(_BOILERPLATE_CLOSE))]
        guidance = _GUIDANCE[regime][uncertainty]
        body = " ".join(guidance)
        return f"{opener} {body} {closer}"

    # -- local loader -------------------------------------------------------

    def _load_local(self) -> List[AlignedTuple]:
        raw_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "raw"
        )
        paths = sorted(glob.glob(os.path.join(raw_dir, "*.json")))
        if not paths:
            raise FileNotFoundError(
                f"data.source=local but no *.json files found in {raw_dir}. "
                "Provide files with keys: prices, call_date_index, transcript, "
                "sector, regime (optional)."
            )
        corpus: List[AlignedTuple] = []
        for i, path in enumerate(paths):
            with open(path, "r", encoding="utf-8") as handle:
                rec = json.load(handle)
            prices = np.asarray(rec["prices"], dtype=float)
            call_index = int(rec["call_date_index"])
            name = os.path.splitext(os.path.basename(path))[0]
            ticker, _, period_s = name.partition("_")
            corpus.append(
                AlignedTuple(
                    firm_id=i,
                    ticker=ticker or name,
                    sector=rec.get("sector", "unknown"),
                    period=int(period_s) if period_s.isdigit() else 0,
                    regime=rec.get("regime", self._infer_regime(prices, call_index)),
                    uncertainty=int(rec.get("uncertainty", 1)),
                    prices=prices,
                    call_index=call_index,
                    transcript=rec["transcript"],
                )
            )
        return corpus

    @staticmethod
    def _infer_regime(prices: np.ndarray, call_index: int) -> str:
        hist = prices[: call_index + 1]
        rets = np.diff(np.log(hist))
        vol = float(np.std(rets)) if rets.size else 0.0
        return "high_vol" if vol * np.sqrt(252) > 0.30 else "low_vol"

    # -- annotation ---------------------------------------------------------

    @staticmethod
    def _annotate(rec: AlignedTuple) -> None:
        hist = rec.historical_prices
        real = rec.realized_prices
        hist_rets = np.diff(np.log(hist)) if hist.size > 1 else np.zeros(1)
        rec.hist_returns = hist_rets
        rec.hist_vol = float(np.std(hist_rets))
        if real.size > 1:
            real_rets = np.diff(np.log(real))
            rec.realized_vol = float(np.std(real_rets))
            rec.realized_move = float(real[-1] / hist[-1] - 1.0)
        else:
            rec.realized_vol = 0.0
            rec.realized_move = 0.0


def split_by_regime(corpus: List[AlignedTuple]) -> Dict[str, List[AlignedTuple]]:
    """Group records by ground-truth volatility regime."""

    out: Dict[str, List[AlignedTuple]] = {}
    for rec in corpus:
        out.setdefault(rec.regime, []).append(rec)
    return out
