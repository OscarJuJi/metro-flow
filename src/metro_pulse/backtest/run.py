"""Run the rolling-origin backtest over the baseline models.

Run as ``python -m metro_pulse.backtest.run``.

Every model sees exactly the same folds, is fitted only on that fold's training
window, and is scored on the same station-days, so the numbers are comparable by
construction rather than by convention.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

import pandas as pd

from metro_pulse.backtest.metrics import score_forecast, seasonal_naive_insample_mae
from metro_pulse.backtest.splitter import Fold, rolling_origin_folds
from metro_pulse.config import ARTIFACTS_DIR, OPERATORS, ensure_dirs
from metro_pulse.models.baselines import HierarchicalSeasonalModel, SeasonalNaive
from metro_pulse.models.gbm import GradientBoostingForecaster
from metro_pulse.panel import load_panel, to_wide

BASELINE_MODEL = "seasonal_naive"

ModelFactory = Callable[[], object]

BASELINES: dict[str, ModelFactory] = {
    BASELINE_MODEL: SeasonalNaive,
    "hierarchical_seasonal": HierarchicalSeasonalModel,
}

# The baseline is always included: it is the denominator of every relative score.
MODELS: dict[str, ModelFactory] = {**BASELINES, "gbm": GradientBoostingForecaster}


def run_fold(
    wide: pd.DataFrame, fold: Fold, models: dict[str, ModelFactory]
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    """Fit and score every model on one fold. Returns rows plus the predictions."""
    fold.validate()
    history = wide.loc[fold.train_start : fold.train_end]
    truth = wide.loc[fold.test_start : fold.test_end]
    insample_mae = seasonal_naive_insample_mae(history)

    predictions = {
        name: factory().fit(history).predict(fold.horizon) for name, factory in models.items()
    }
    baseline = predictions[BASELINE_MODEL]

    rows = []
    for name, prediction in predictions.items():
        score = score_forecast(
            truth, prediction, baseline=baseline, insample_mae=insample_mae
        )
        rows.append({"fold": fold.index, "model": name, **score.as_dict()})
    return rows, predictions


def score_by_horizon(
    wide: pd.DataFrame, fold: Fold, predictions: dict[str, pd.DataFrame]
) -> list[dict]:
    """Break one fold's accuracy down by how far ahead the day was forecast."""
    truth = wide.loc[fold.test_start : fold.test_end]
    baseline = predictions[BASELINE_MODEL]
    rows = []
    for step, date in enumerate(truth.index, start=1):
        day = truth.loc[[date]]
        for name, prediction in predictions.items():
            score = score_forecast(
                day,
                prediction.loc[[date]],
                baseline=baseline.loc[[date]],
                insample_mae=float("nan"),
            )
            rows.append(
                {
                    "fold": fold.index,
                    "model": name,
                    "horizon": step,
                    "mae": score.mae,
                    "relative_mae": score.relative_mae,
                }
            )
    return rows


def run_backtest(
    operator: str = "metro",
    *,
    n_folds: int = 12,
    horizon: int = 14,
    models: dict[str, ModelFactory] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the models across all folds. Returns (per-fold, per-horizon) tables."""
    models = MODELS if models is None else models
    if BASELINE_MODEL not in models:
        models = {BASELINE_MODEL: MODELS[BASELINE_MODEL], **models}
    wide = to_wide(load_panel(operator))
    folds = rolling_origin_folds(wide.index, n_folds=n_folds, horizon=horizon)

    fold_rows: list[dict] = []
    horizon_rows: list[dict] = []
    for fold in folds:
        print(f"  {fold.describe()}")
        rows, predictions = run_fold(wide, fold, models)
        fold_rows.extend(rows)
        horizon_rows.extend(score_by_horizon(wide, fold, predictions))

    return pd.DataFrame(fold_rows), pd.DataFrame(horizon_rows)


def check_gate(per_fold: pd.DataFrame, *, model: str, limit: float) -> tuple[bool, str]:
    """Has ``model`` stayed better than the baseline by the required margin?

    This is the check the monthly retrain runs before it publishes anything. A
    model that no longer beats "the same weekday last week" is not worth
    shipping, and the point of an automated pipeline is that nobody has to
    remember to look.
    """
    scores = per_fold[per_fold["model"] == model]["relative_mae"]
    if scores.empty:
        return False, f"{model!r} was not scored, so the gate cannot pass"
    mean = float(scores.mean())
    verdict = "PASS" if mean <= limit else "FAIL"
    return mean <= limit, (
        f"{verdict}: {model} averaged relative_mae {mean:.4f} across "
        f"{len(scores)} folds (limit {limit:.4f}); worst fold {scores.max():.4f}"
    )


def summarize(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Average each model's scores across folds."""
    return (
        per_fold.groupby("model")[["mae", "smape", "mase", "relative_mae"]]
        .mean()
        .sort_values("relative_mae")
        .round(4)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", default="metro", choices=sorted(OPERATORS))
    parser.add_argument("--folds", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument(
        "--gate",
        type=float,
        default=None,
        metavar="LIMIT",
        help="exit non-zero unless the gbm averages a relative_mae at or below LIMIT",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=sorted(MODELS),
        choices=sorted(MODELS),
        help="which models to score; the seasonal naive is always added",
    )
    args = parser.parse_args(argv)

    ensure_dirs()
    print(f"Backtesting {args.folds} folds of {args.horizon} days on {args.operator}:")
    per_fold, per_horizon = run_backtest(
        args.operator,
        n_folds=args.folds,
        horizon=args.horizon,
        models={name: MODELS[name] for name in args.models},
    )

    print("\nPer-model averages across folds:")
    print(summarize(per_fold).to_string())

    print("\nMean absolute error by horizon:")
    pivot = per_horizon.pivot_table(
        index="horizon", columns="model", values="mae", aggfunc="mean"
    ).round(1)
    print(pivot.to_string())

    per_fold.to_csv(ARTIFACTS_DIR / f"{args.operator}_backtest_folds.csv", index=False)
    per_horizon.to_csv(ARTIFACTS_DIR / f"{args.operator}_backtest_horizons.csv", index=False)
    print(f"\nWritten to {ARTIFACTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
