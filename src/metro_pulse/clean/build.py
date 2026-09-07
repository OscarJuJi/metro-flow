"""Turn the raw ridership CSV into the modelling panel.

Run as ``python -m metro_pulse.clean.build``.

Output: ``data/interim/<operator>_panel.parquet``, a strictly rectangular
station-day panel (every line-station pair on every date in the span) carrying

``ridership``
    the observed count, ``NaN`` where the source is unusable;
``status``
    why a day is or is not usable (see :mod:`.service`);
``target``
    the modelling target: ``ridership`` on operating days, ``NaN`` otherwise.

Nothing downstream needs to know about mojibake, closures or mislabelled rows.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from metro_pulse.clean.canonical import (
    assert_expected_network,
    build_station_registry,
    canonicalize,
    line_sort_rank,
)
from metro_pulse.clean.quality import find_duplicate_station_days
from metro_pulse.clean.service import (
    AMBIGUOUS_LABEL,
    OPERATING,
    derive_service_status,
    summarize_status,
)
from metro_pulse.config import (
    INTERIM_DIR,
    OPERATORS,
    PROJECT_ROOT,
    RAW_DIR,
    OperatorSource,
    ensure_dirs,
)

PANEL_COLUMNS = [
    "date",
    "line_id",
    "station_id",
    "station_name",
    "ridership",
    "status",
    "target",
]


class PanelError(RuntimeError):
    """The panel did not come out with the shape the pipeline guarantees."""


def to_rectangular_panel(canonical: pd.DataFrame, ambiguous: pd.Index) -> pd.DataFrame:
    """Drop unusable rows, then reindex onto the full pair x date grid.

    Dropping the ambiguous December 2020 rows leaves 62 holes; reindexing puts
    them back as explicit ``NaN`` so the panel stays rectangular and every
    downstream lag lands on the calendar day it claims to.
    """
    usable = canonical.drop(index=ambiguous)
    pairs = pd.MultiIndex.from_frame(
        usable[["line_id", "station_id"]].drop_duplicates().sort_values(["line_id", "station_id"])
    )
    dates = pd.date_range(canonical["date"].min(), canonical["date"].max(), freq="D")
    grid = pd.MultiIndex.from_tuples(
        [(line, station, date) for line, station in pairs for date in dates],
        names=["line_id", "station_id", "date"],
    )

    panel = (
        usable.set_index(["line_id", "station_id", "date"])[["station_name", "ridership"]]
        .reindex(grid)
        .reset_index()
    )
    panel["ridership"] = panel["ridership"].astype("float64")

    names = usable.groupby(["line_id", "station_id"], observed=True)["station_name"].agg(
        lambda s: s.mode().iat[0]
    )
    panel["station_name"] = panel.set_index(["line_id", "station_id"]).index.map(names)

    panel["_order"] = panel["line_id"].map(line_sort_rank)
    panel = panel.sort_values(["_order", "line_id", "station_id", "date"]).drop(columns="_order")
    return panel.reset_index(drop=True)


def build_panel(operator: OperatorSource, *, min_closure_days: int | None = None) -> pd.DataFrame:
    """Read the raw parquet for one operator and return its modelling panel."""
    raw_path = RAW_DIR / f"{operator.key}_ridership.parquet"
    if not raw_path.exists():
        raise PanelError(
            f"{raw_path} not found. Run `python -m metro_pulse.ingest.download "
            f"--operator {operator.key}` first."
        )

    print(f"[1/5] Reading {raw_path} ...")
    raw = pd.read_parquet(raw_path)
    print(f"      {len(raw):,} raw rows")

    print("[2/5] Canonicalizing line and station labels ...")
    canonical = canonicalize(raw)
    counts = assert_expected_network(canonical, operator.network)
    print(
        f"      raw labels: {raw['linea'].nunique()} lines / {raw['estacion'].nunique()} stations"
    )
    print(
        f"      canonical : {counts['n_lines']} lines / {counts['n_stations']} stations "
        f"/ {counts['n_pairs']} line-station pairs"
    )

    print("[3/5] Scanning for duplicate station-days ...")
    ambiguous_rows, duplicate_report = find_duplicate_station_days(canonical)
    print(f"      {duplicate_report.summary()}")

    print("[4/5] Building the rectangular panel and deriving service status ...")
    panel = to_rectangular_panel(canonical, ambiguous_rows)
    kwargs = {} if min_closure_days is None else {"min_closure_days": min_closure_days}
    panel["status"] = derive_service_status(panel, **kwargs)
    panel.loc[panel["ridership"].isna(), "status"] = AMBIGUOUS_LABEL
    panel["target"] = panel["ridership"].where(panel["status"] == OPERATING)

    n_pairs = counts["n_pairs"]
    n_dates = panel["date"].nunique()
    if len(panel) != n_pairs * n_dates:
        raise PanelError(
            f"Panel is not rectangular: {len(panel):,} rows for {n_pairs} pairs x {n_dates} dates"
        )

    print("[5/5] Status breakdown:")
    print(summarize_status(panel["status"]).to_string())
    return panel[PANEL_COLUMNS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", default="metro", choices=sorted(OPERATORS))
    parser.add_argument(
        "--min-closure-days",
        type=int,
        default=None,
        help="zero runs at least this long count as a closure (default: 4)",
    )
    args = parser.parse_args(argv)

    ensure_dirs()
    operator = OPERATORS[args.operator]
    panel = build_panel(operator, min_closure_days=args.min_closure_days)

    panel_path = INTERIM_DIR / f"{operator.key}_panel.parquet"
    panel.to_parquet(panel_path, index=False, compression="snappy")

    registry_dir = PROJECT_ROOT / "reference"
    registry_dir.mkdir(exist_ok=True)
    registry_path = registry_dir / f"{operator.key}_stations.csv"
    raw = pd.read_parquet(RAW_DIR / f"{operator.key}_ridership.parquet")
    build_station_registry(canonicalize(raw)).to_csv(registry_path, index=False, encoding="utf-8")

    print("\nPanel built.")
    print(f"  panel    : {panel_path}")
    print(f"  registry : {registry_path}")
    print(f"  rows     : {len(panel):,}")
    print(f"  span     : {panel['date'].min():%Y-%m-%d} .. {panel['date'].max():%Y-%m-%d}")
    print(f"  target   : {panel['target'].notna().sum():,} usable station-days "
          f"({panel['target'].notna().mean():.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
