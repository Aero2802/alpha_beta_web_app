#!/usr/bin/env python3
"""Build full-factorial WTT3 aero tables from alpha/beta/TAS design points."""

from __future__ import annotations

import csv
import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


# --------------------------
# User-configurable settings
# --------------------------
DESIGN_SPACE_CSV = Path("alpha_beta_TAS_WTT3_matrix_latest.csv")
TIME_PER_STEP_SECONDS = 12.0

# Automatic tare-sweep insertion.
# A tare sweep is always inserted at the start of each test.
# Set TARE_INTERVAL_MINUTES <= 0 to disable additional periodic tare sweeps.
TARE_INTERVAL_MINUTES = 60.0
TARE_DWELL_TIME: Optional[float] = None
TARE_MAX_TRAVERSE_TIME: Optional[float] = None

# Keep these output names exactly as requested.
TEST_OUTPUT_FILES = {
    "passive": Path("WTT3_full_aero_table_passive.csv"),
    "front_lateral": Path("WTT3_full_aero_table_front_lateral.csv"),
    "hoop": Path("WTT3_full_aero_table_hoop.csv"),
    "interactions": Path("WTT3_full_aero_table_interactions.csv"),
}

# Full-factorial values to edit for each test.
TEST_FACTORS = {
    "passive": {
        "HOOP_ALPHA": [0],
        "HOOP_BETA": [-30, -20, -10, 0, 10, 20, 30],
        "FLT_OMEGA": [0],
        "MT_OMEGA": [0],
        "DWELL_TIME": [10],
        "MAX_TRAVERSE_TIME": [6],
    },
    "front_lateral": {
        "HOOP_ALPHA": [0],
        "HOOP_BETA": [0],
        "FLT_OMEGA": [-15000, -11250, 7500, 11250, 15000],
        "MT_OMEGA": [0],
        "DWELL_TIME": [10],
        "MAX_TRAVERSE_TIME": [6],
    },
    "hoop": {
        "HOOP_ALPHA": [0],
        "HOOP_BETA": [-30, -20, -10, 0, 10, 20, 30],
        "FLT_OMEGA": [0],
        "MT_OMEGA": [-7500, -5625, 3750, 5625, 7500],
        "DWELL_TIME": [10],
        "MAX_TRAVERSE_TIME": [6],
    },
    "interactions": {
        "HOOP_ALPHA": [0],
        "HOOP_BETA": [0],
        "FLT_OMEGA": [-15000, -11250, 7500, 11250, 15000],
        "MT_OMEGA": [-7500, -5625, 3750, 5625, 7500],
        "DWELL_TIME": [10],
        "MAX_TRAVERSE_TIME": [6],
    },
}


OUTPUT_COLUMNS = [
    "SETPOINT",
    "ALPHA",
    "BETA",
    "TAS",
    "HOOP_ALPHA",
    "HOOP_BETA",
    "FLT_OMEGA",
    "MT_OMEGA",
    "DWELL_TIME",
    "MAX_TRAVERSE_TIME",
]

BASE_COLUMNS = ["ALPHA", "BETA", "TAS"]
FACTOR_COLUMNS = OUTPUT_COLUMNS[4:]


def numeric_sort_key(raw_value: str) -> Tuple[int, object]:
    try:
        return (0, float(raw_value))
    except ValueError:
        return (1, raw_value)


def load_design_space(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Design-space CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as infile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {csv_path}")

        missing = [col for col in BASE_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required base columns in {csv_path}: {missing}")

        return [{col: row[col] for col in BASE_COLUMNS} for row in reader]


def ensure_valid_factor_config(test_name: str, factors: Dict[str, Iterable[object]]) -> None:
    missing = [col for col in FACTOR_COLUMNS if col not in factors]
    if missing:
        raise ValueError(f"Missing factor columns for '{test_name}': {missing}")

    for col in FACTOR_COLUMNS:
        values = list(factors[col])
        if not values:
            raise ValueError(f"Factor '{col}' for '{test_name}' cannot be empty")
        factors[col] = values


def unique_beta_values(base_points: List[Dict[str, str]]) -> List[str]:
    betas = {point["BETA"].strip() for point in base_points}
    return sorted(betas, key=numeric_sort_key)


def resolve_tare_value(override: Optional[float], fallback_values: List[object]) -> object:
    if override is not None:
        return override
    return fallback_values[0]


def build_tare_sweep_rows(
    unique_betas: List[str],
    start_setpoint: int,
    factors: Dict[str, List[object]],
) -> List[Dict[str, object]]:
    dwell_value = resolve_tare_value(TARE_DWELL_TIME, factors["DWELL_TIME"])
    max_traverse_value = resolve_tare_value(
        TARE_MAX_TRAVERSE_TIME, factors["MAX_TRAVERSE_TIME"]
    )

    rows: List[Dict[str, object]] = []
    setpoint = start_setpoint
    for beta_value in unique_betas:
        rows.append(
            {
                "SETPOINT": setpoint,
                "ALPHA": 0,
                "BETA": beta_value,
                "TAS": 0,
                "HOOP_ALPHA": 0,
                "HOOP_BETA": 0,
                "FLT_OMEGA": 0,
                "MT_OMEGA": 0,
                "DWELL_TIME": dwell_value,
                "MAX_TRAVERSE_TIME": max_traverse_value,
            }
        )
        setpoint += 1
    return rows


def build_rows(
    base_points: List[Dict[str, str]],
    factors: Dict[str, List[object]],
    unique_betas: List[str],
) -> Tuple[List[Dict[str, object]], int]:
    combos = list(itertools.product(*(factors[col] for col in FACTOR_COLUMNS)))
    rows: List[Dict[str, object]] = []
    setpoint = 1

    tare_cases_inserted = 0
    tare_enabled = bool(unique_betas)
    periodic_tare_enabled = tare_enabled and TARE_INTERVAL_MINUTES > 0
    elapsed_seconds = 0.0
    tare_interval_seconds = TARE_INTERVAL_MINUTES * 60.0
    next_tare_at_seconds = tare_interval_seconds

    if tare_enabled:
        tare_rows = build_tare_sweep_rows(unique_betas, setpoint, factors)
        rows.extend(tare_rows)
        tare_count = len(tare_rows)
        tare_cases_inserted += tare_count
        setpoint += tare_count
        elapsed_seconds += tare_count * TIME_PER_STEP_SECONDS

    for base in base_points:
        for combo in combos:
            row: Dict[str, object] = {
                "SETPOINT": setpoint,
                "ALPHA": base["ALPHA"],
                "BETA": base["BETA"],
                "TAS": base["TAS"],
            }
            row.update({col: val for col, val in zip(FACTOR_COLUMNS, combo)})
            rows.append(row)
            setpoint += 1
            elapsed_seconds += TIME_PER_STEP_SECONDS

            if periodic_tare_enabled and elapsed_seconds >= next_tare_at_seconds:
                tare_rows = build_tare_sweep_rows(unique_betas, setpoint, factors)
                rows.extend(tare_rows)
                tare_count = len(tare_rows)
                tare_cases_inserted += tare_count
                setpoint += tare_count
                elapsed_seconds += tare_count * TIME_PER_STEP_SECONDS
                next_tare_at_seconds = elapsed_seconds + tare_interval_seconds

    return rows, tare_cases_inserted


def write_table(csv_path: Path, rows: List[Dict[str, object]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def format_duration(total_seconds: float) -> str:
    total_minutes = int(round(total_seconds / 60.0))
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}h {minutes:02d}m"


def summarize_design_space_by_tas(
    base_points: List[Dict[str, str]],
) -> List[Tuple[str, int]]:
    by_tas: Dict[str, Set[Tuple[str, str]]] = {}
    for point in base_points:
        tas = point["TAS"].strip()
        alpha_beta = (point["ALPHA"].strip(), point["BETA"].strip())
        by_tas.setdefault(tas, set()).add(alpha_beta)

    return [(tas, len(by_tas[tas])) for tas in sorted(by_tas.keys(), key=numeric_sort_key)]


def main() -> None:
    base_points = load_design_space(DESIGN_SPACE_CSV)
    if not base_points:
        raise ValueError(f"No rows found in {DESIGN_SPACE_CSV}")

    summary_rows = []
    all_unique_betas = unique_beta_values(base_points)

    for test_name, output_file in TEST_OUTPUT_FILES.items():
        factors = dict(TEST_FACTORS[test_name])
        ensure_valid_factor_config(test_name, factors)

        factor_combos = 1
        for col in FACTOR_COLUMNS:
            factor_combos *= len(factors[col])

        core_cases = len(base_points) * factor_combos
        rows, tare_cases = build_rows(base_points, factors, all_unique_betas)
        write_table(output_file, rows)

        total_cases = len(rows)
        est_seconds = total_cases * TIME_PER_STEP_SECONDS
        summary_rows.append(
            {
                "test": test_name,
                "factor_combos": factor_combos,
                "core_cases": core_cases,
                "tare_cases": tare_cases,
                "total_cases": total_cases,
                "est_seconds": est_seconds,
                "output": output_file,
            }
        )

    tas_design_space = summarize_design_space_by_tas(base_points)

    print("\nWTT3 DOE Build Summary")
    print("=" * 122)
    print(f"Design space CSV   : {DESIGN_SPACE_CSV}")
    print(f"Design points      : {len(base_points):,}")
    print(f"Time per step      : {TIME_PER_STEP_SECONDS:.2f} s")
    print(f"Tare at start      : enabled ({len(all_unique_betas)} BETA points)")
    if TARE_INTERVAL_MINUTES > 0:
        print(f"Tare interval      : every {TARE_INTERVAL_MINUTES:.2f} min")
    else:
        print("Tare interval      : disabled (start tare only)")
    print("-" * 122)
    print("Alpha-Beta Design Space by TAS")
    print(f"{'TAS':>10}{'Alpha-Beta Points':>22}")
    for tas, point_count in tas_design_space:
        print(f"{tas:>10}{point_count:>22,}")
    print("-" * 122)
    print(
        f"{'Test':<16}{'Factor Combos':>16}{'Core Cases':>14}{'Tare Cases':>12}"
        f"{'Total Cases':>14}{'Est. Test Time':>18}  {'Output CSV'}"
    )
    print("-" * 122)

    total_cases_all = 0
    total_seconds_all = 0.0
    total_tare_cases = 0
    for row in summary_rows:
        total_cases_all += row["total_cases"]
        total_seconds_all += row["est_seconds"]
        total_tare_cases += row["tare_cases"]
        print(
            f"{row['test']:<16}{row['factor_combos']:>16,}{row['core_cases']:>14,}"
            f"{row['tare_cases']:>12,}{row['total_cases']:>14,}"
            f"{format_duration(row['est_seconds']):>18}  {row['output']}"
        )

    print("-" * 122)
    print(
        f"{'TOTAL':<16}{'':>16}{'':>14}{total_tare_cases:>12,}{total_cases_all:>14,}"
        f"{format_duration(total_seconds_all):>18}"
    )
    print("=" * 122)


if __name__ == "__main__":
    main()
