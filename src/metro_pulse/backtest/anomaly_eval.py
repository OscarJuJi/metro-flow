"""Measure the anomaly detector against what actually happened.

Run as ``python -m metro_pulse.backtest.anomaly_eval``.

Anomaly detection is usually evaluated on synthetic injections because nobody
has labels. Here the labels are real: the service mask derived in
:mod:`metro_pulse.clean.service` records every day each station was out of
service across sixteen years -- the Line 12 collapse, the Line 1 modernization,
the control-centre fire, the pandemic closures -- and every isolated day a
running station carried nobody.

Evaluation walks forward one *month* at a time: the detector is fitted on
everything before a month and scored on that month, so no event is ever detected
by a model that had already seen it. Monthly is not an arbitrary choice -- it is
how often the source publishes, so it is how often the detector could really be
refitted. Refitting yearly instead leaves the expected-demand level up to a year
stale, and since the network is in long-term decline the residuals drift
negative, the CUSUM runs away and the detector flags half the network.

Two numbers are reported and they answer different questions. *Day level* asks
how many station-days were classified correctly. *Event level* asks the question
an operator would ask -- when a station went down, did the detector notice
within three days? Flagging day four hundred of a two-year closure is worth
nothing, and event-level recall refuses to count it.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from metro_pulse.clean.service import CLOSED, OPERATING, find_runs
from metro_pulse.config import ARTIFACTS_DIR, OPERATORS, ensure_dirs
from metro_pulse.models.anomaly import DEFAULT_FALSE_ALARM_RATE, RobustAnomalyDetector
from metro_pulse.panel import load_panel, to_wide

WARMUP_YEARS = 3
HISTORY_YEARS = 3
ONSET_WINDOW_DAYS = 3


@dataclass
class DetectionScore:
    """Day-level confusion of the detector over the evaluated period."""

    n_days_scored: int
    n_positive_days: int
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        flagged = self.true_positives + self.false_positives
        return self.true_positives / flagged if flagged else float("nan")

    @property
    def recall(self) -> float:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else float("nan")

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        if not np.isfinite(precision) or not np.isfinite(recall) or precision + recall == 0:
            return float("nan")
        return 2 * precision * recall / (precision + recall)

    def summary(self) -> str:
        return (
            f"scored {self.n_days_scored:,} station-days, {self.n_positive_days:,} of them "
            f"abnormal | precision {self.precision:.3f} | recall {self.recall:.3f} "
            f"| F1 {self.f1:.3f}"
        )


def ground_truth(panel: pd.DataFrame) -> pd.DataFrame:
    """Wide boolean matrix: was this station-day genuinely abnormal?

    A closure is abnormal. So is a day a station was open and carried nobody.
    Days before a line opened, days nobody counted, and the mislabelled December
    2020 rows are excluded from scoring altogether.
    """
    scoreable = panel["status"].isin([OPERATING, CLOSED])
    abnormal = (panel["status"] == CLOSED) | (
        (panel["status"] == OPERATING) & (panel["ridership"] == 0)
    )
    labelled = panel.assign(
        label=np.where(scoreable, abnormal, np.nan),
        series_id=panel["line_id"] + ":" + panel["station_id"],
    )
    return labelled.pivot_table(
        index="date", columns="series_id", values="label", aggfunc="first", dropna=False
    )


def walk_forward_flags(
    panel: pd.DataFrame,
    *,
    warmup_years: int = WARMUP_YEARS,
    history_years: int = HISTORY_YEARS,
    rates: tuple[float, ...] = (DEFAULT_FALSE_ALARM_RATE,),
) -> dict[float, pd.DataFrame]:
    """Refit before every month and flag that month, once per alarm rate.

    Only the most recent ``history_years`` are used to fit: the level, the
    weekday shape and the residual spread of 2013 say nothing useful about 2026,
    and dragging them along makes every fit slower and staler.

    Every requested rate reuses the same monthly fit, so producing an operating
    curve costs one pass rather than one pass per point.
    """
    observed = to_wide(panel, value="ridership")
    operating = to_wide(panel, value="target")

    months = observed.index.to_period("M").unique()
    collected: dict[float, list[pd.DataFrame]] = {rate: [] for rate in rates}
    n_fits = 0
    for month in months[warmup_years * 12 :]:
        month_start = month.to_timestamp()
        history = operating.loc[
            (operating.index < month_start)
            & (operating.index >= month_start - pd.DateOffset(years=history_years))
        ]
        detector = RobustAnomalyDetector().fit(history)
        n_fits += 1

        current = observed.loc[observed.index.to_period("M") == month]
        for rate in rates:
            collected[rate].append(detector.set_false_alarm_rate(rate).detect(current))

    print(f"  {n_fits} monthly refits, {history_years} years of history each")
    return {rate: pd.concat(parts) for rate, parts in collected.items()}


def score_days(flags: pd.DataFrame, truth: pd.DataFrame) -> DetectionScore:
    """Day-level confusion over the station-days that carry a label."""
    truth = truth.reindex(index=flags.index, columns=flags.columns)
    labelled = truth.notna().to_numpy()
    actual = (truth.to_numpy() == 1) & labelled
    predicted = flags.to_numpy().astype(bool) & labelled

    return DetectionScore(
        n_days_scored=int(labelled.sum()),
        n_positive_days=int(actual.sum()),
        true_positives=int((predicted & actual).sum()),
        false_positives=int((predicted & ~actual & labelled).sum()),
        false_negatives=int((~predicted & actual).sum()),
    )


def score_event_onsets(
    flags: pd.DataFrame, truth: pd.DataFrame, *, window: int = ONSET_WINDOW_DAYS
) -> pd.DataFrame:
    """For each interruption, did the detector fire within ``window`` days of its start?"""
    truth = truth.reindex(index=flags.index, columns=flags.columns)
    records = []
    for series in truth.columns:
        labels = (truth[series].to_numpy() == 1)
        fired = flags[series].to_numpy().astype(bool)
        for start, end in find_runs(labels):
            onset = fired[start : min(start + window, end + 1)]
            records.append(
                {
                    "series_id": series,
                    "start": truth.index[start],
                    "end": truth.index[end],
                    "days": end - start + 1,
                    "detected": bool(onset.any()),
                }
            )
    return pd.DataFrame.from_records(
        records, columns=["series_id", "start", "end", "days", "detected"]
    )


def operating_curve(
    flags_by_rate: dict[float, pd.DataFrame], truth: pd.DataFrame, *, min_event_days: int = 4
) -> pd.DataFrame:
    """One row per alarm rate: what the operator pays and what they get."""
    rows = []
    for rate, flags in sorted(flags_by_rate.items()):
        day = score_days(flags, truth)
        events = score_event_onsets(flags, truth)
        long_events = events[events["days"] >= min_event_days]
        single_day = events[events["days"] == 1]
        n_days = flags.index.nunique()
        rows.append(
            {
                "false_alarm_rate": rate,
                "alerts_per_day": flags.to_numpy().sum() / n_days,
                "precision": day.precision,
                "day_recall": day.recall,
                "f1": day.f1,
                "onset_recall_all": events["detected"].mean(),
                f"onset_recall_{min_event_days}d_plus": long_events["detected"].mean(),
                "onset_recall_1d": single_day["detected"].mean(),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", default="metro", choices=sorted(OPERATORS))
    parser.add_argument(
        "--rates",
        nargs="+",
        type=float,
        default=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
        help="target shares of ordinary days the detector is allowed to flag",
    )
    parser.add_argument("--min-event-days", type=int, default=4)
    args = parser.parse_args(argv)

    ensure_dirs()
    panel = load_panel(args.operator)
    truth = ground_truth(panel)

    print("Walking the detector forward one month at a time:")
    flags_by_rate = walk_forward_flags(panel, rates=tuple(args.rates))

    curve = operating_curve(flags_by_rate, truth, min_event_days=args.min_event_days)
    print()
    print("Operating curve:")
    print(curve.round(4).to_string(index=False))

    chosen = (
        DEFAULT_FALSE_ALARM_RATE
        if DEFAULT_FALSE_ALARM_RATE in flags_by_rate
        else args.rates[0]
    )
    events = score_event_onsets(flags_by_rate[chosen], truth)
    events.to_csv(ARTIFACTS_DIR / f"{args.operator}_anomaly_events.csv", index=False)
    curve.to_csv(ARTIFACTS_DIR / f"{args.operator}_anomaly_curve.csv", index=False)

    day = score_days(flags_by_rate[chosen], truth)
    pd.DataFrame([asdict(day)]).to_csv(
        ARTIFACTS_DIR / f"{args.operator}_anomaly_days.csv", index=False
    )
    print()
    print(f"Operating point {chosen}: {day.summary()}")
    print(f"Written to {ARTIFACTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
