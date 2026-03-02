#!/usr/bin/env python3
"""Shareable Streamlit app for symmetric alpha-beta point design and CSV export.

Run locally:
    streamlit run wind_tunnel_alpha_beta_web_app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Set, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


Coord = Tuple[int, int]  # (alpha, beta)


@dataclass
class DesignConfig:
    alpha_max_abs: int = 180
    beta_max_abs: int = 90
    n_alpha_positive: int = 6
    n_beta_positive: int = 6
    alpha_exponent: float = 1.0
    beta_exponent: float = 1.0
    snap_step: int = 5


def generate_clustered_integer_levels(max_abs: int, n_positive: int, exponent: float) -> list[int]:
    if max_abs < 1:
        return []

    n_positive = max(1, min(n_positive, max_abs))
    exponent = max(1.0, exponent)

    raw_values: list[int] = []
    for i in range(1, n_positive + 1):
        q = i / n_positive
        value = int(round(max_abs * (q**exponent)))
        value = max(1, min(max_abs, value))
        raw_values.append(value)

    levels = sorted(set(raw_values))

    if len(levels) < n_positive:
        used = set(levels)
        for candidate in range(1, max_abs + 1):
            if candidate not in used:
                levels.append(candidate)
                used.add(candidate)
                if len(levels) == n_positive:
                    break
        levels = sorted(levels)

    if max_abs not in levels:
        if len(levels) == n_positive:
            levels[-1] = max_abs
        else:
            levels.append(max_abs)
        levels = sorted(set(levels))

    if len(levels) > n_positive:
        keep = levels[: max(0, n_positive - 1)] + [max_abs]
        levels = sorted(set(keep))
        while len(levels) < n_positive:
            for candidate in range(1, max_abs + 1):
                if candidate not in levels:
                    levels.append(candidate)
                    break
            levels = sorted(levels)

    return levels


def symmetry_orbit(coord: Coord) -> Set[Coord]:
    alpha, beta = coord
    a = abs(int(alpha))
    b = abs(int(beta))
    return {(sa * a, sb * b) for sa in (-1, 1) for sb in (-1, 1)}


def snap_to_step(value: float, step: int) -> int:
    return int(round(value / step) * step)


def build_symmetric_points(cfg: DesignConfig) -> tuple[Set[Coord], list[int], list[int]]:
    alpha_pos = generate_clustered_integer_levels(cfg.alpha_max_abs, cfg.n_alpha_positive, cfg.alpha_exponent)
    beta_pos = generate_clustered_integer_levels(cfg.beta_max_abs, cfg.n_beta_positive, cfg.beta_exponent)

    alpha_nonnegative = [0] + alpha_pos
    beta_nonnegative = [0] + beta_pos

    q1: Set[Coord] = {(a, b) for a in alpha_nonnegative for b in beta_nonnegative}

    all_points: Set[Coord] = set()
    for alpha, beta in q1:
        all_points |= symmetry_orbit((alpha, beta))

    return all_points, alpha_nonnegative, beta_nonnegative


def points_to_df(all_coords: Set[Coord], active_coords: Set[Coord]) -> pd.DataFrame:
    rows = []
    for alpha, beta in sorted(all_coords, key=lambda p: (p[0], p[1])):
        rows.append(
            {
                "alpha": alpha,
                "beta": beta,
                "status": "active" if (alpha, beta) in active_coords else "removed",
                "coord_key": f"{alpha},{beta}",
            }
        )
    return pd.DataFrame(rows)


def active_points_csv_df(active_coords: Set[Coord]) -> pd.DataFrame:
    points = sorted(active_coords, key=lambda p: (p[0], p[1]))
    return pd.DataFrame(
        {
            "point_id": range(1, len(points) + 1),
            "alpha": [p[0] for p in points],
            "beta": [p[1] for p in points],
        }
    )


def parse_point_beta_alpha(raw: str) -> tuple[float, float]:
    text = raw.strip().replace(" ", "")
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError("Use format (beta,alpha)")
    beta_raw = float(parts[0])
    alpha_raw = float(parts[1])
    return beta_raw, alpha_raw


def extract_selected_coords(selection_event) -> Set[Coord]:
    coords: Set[Coord] = set()
    if not selection_event:
        return coords

    points = []
    if isinstance(selection_event, dict):
        points = selection_event.get("selection", {}).get("points", [])
    else:
        selection = getattr(selection_event, "selection", None)
        points = getattr(selection, "points", []) if selection is not None else []

    for p in points:
        customdata = None
        if isinstance(p, dict):
            customdata = p.get("customdata")
        else:
            customdata = getattr(p, "customdata", None)

        alpha = beta = None
        if isinstance(customdata, (list, tuple)) and len(customdata) >= 2:
            alpha = int(customdata[0])
            beta = int(customdata[1])
        elif isinstance(customdata, str) and "," in customdata:
            a, b = customdata.split(",", 1)
            alpha = int(a)
            beta = int(b)

        if alpha is not None and beta is not None:
            coords.add((alpha, beta))

    return coords


def apply_symmetry(coords: Iterable[Coord], all_coords: Set[Coord]) -> Set[Coord]:
    result: Set[Coord] = set()
    for coord in coords:
        result |= (symmetry_orbit(coord) & all_coords)
    return result


def add_orbit(alpha_raw: float, beta_raw: float, cfg: DesignConfig) -> None:
    alpha = snap_to_step(alpha_raw, cfg.snap_step)
    beta = snap_to_step(beta_raw, cfg.snap_step)
    key = (abs(alpha), abs(beta))
    orbit = symmetry_orbit((key[0], key[1]))

    st.session_state.all_coords |= orbit
    st.session_state.active_coords |= orbit
    st.session_state.selected_coords = set(orbit)
    st.session_state.notice = f"Added/restored orbit at snapped point alpha={alpha}, beta={beta}."


def init_state() -> None:
    if "cfg" not in st.session_state:
        st.session_state.cfg = DesignConfig()
    if "all_coords" not in st.session_state or "active_coords" not in st.session_state:
        pts, a_lvls, b_lvls = build_symmetric_points(st.session_state.cfg)
        st.session_state.all_coords = pts
        st.session_state.active_coords = set(pts)
        st.session_state.alpha_levels = a_lvls
        st.session_state.beta_levels = b_lvls
    if "selected_coords" not in st.session_state:
        st.session_state.selected_coords = set()
    if "output_filename" not in st.session_state:
        st.session_state.output_filename = "alpha_beta_symmetric_points.csv"
    if "manual_point" not in st.session_state:
        st.session_state.manual_point = "(0,0)"
    if "notice" not in st.session_state:
        st.session_state.notice = ""


def main() -> None:
    st.set_page_config(page_title="Alpha-Beta DOE Web Tool", layout="wide")
    init_state()

    st.title("'Alpha-Beta' Parameter Space Web Tool")
    st.caption("Shareable browser app: users can generate/edit points and download CSV locally.")

    cfg: DesignConfig = st.session_state.cfg

    with st.sidebar:
        st.subheader("Generation Parameters")
        alpha_max = st.number_input("alpha_max", min_value=1, value=cfg.alpha_max_abs, step=1)
        beta_max = st.number_input("beta_max", min_value=1, value=cfg.beta_max_abs, step=1)
        n_alpha = st.number_input("n_alpha", min_value=1, value=cfg.n_alpha_positive, step=1)
        n_beta = st.number_input("n_beta", min_value=1, value=cfg.n_beta_positive, step=1)
        alpha_exp = st.number_input("alpha_exp", min_value=1.0, value=float(cfg.alpha_exponent), step=0.1)
        beta_exp = st.number_input("beta_exp", min_value=1.0, value=float(cfg.beta_exponent), step=0.1)
        snap_step = st.number_input("snap_step", min_value=1, value=cfg.snap_step, step=1)

        if st.button("Regenerate Grid", use_container_width=True):
            st.session_state.cfg = DesignConfig(
                alpha_max_abs=int(alpha_max),
                beta_max_abs=int(beta_max),
                n_alpha_positive=int(min(n_alpha, alpha_max)),
                n_beta_positive=int(min(n_beta, beta_max)),
                alpha_exponent=float(alpha_exp),
                beta_exponent=float(beta_exp),
                snap_step=int(snap_step),
            )
            pts, a_lvls, b_lvls = build_symmetric_points(st.session_state.cfg)
            st.session_state.all_coords = pts
            st.session_state.active_coords = set(pts)
            st.session_state.selected_coords = set()
            st.session_state.alpha_levels = a_lvls
            st.session_state.beta_levels = b_lvls
            st.session_state.notice = "Grid regenerated from parameters."

        st.divider()
        st.subheader("Add Point")
        st.text_input("point(beta,alpha)", key="manual_point", help="x first (beta), y second (alpha). Example: (15,-10)")
        if st.button("Add Point (Symmetric Orbit)", use_container_width=True):
            try:
                beta_raw, alpha_raw = parse_point_beta_alpha(st.session_state.manual_point)
                add_orbit(alpha_raw=alpha_raw, beta_raw=beta_raw, cfg=st.session_state.cfg)
            except Exception:
                st.session_state.notice = "Add point error: use format (beta,alpha), e.g. (15,-10)."

        st.divider()
        st.subheader("Output")
        st.text_input(
            "output_file",
            key="output_filename",
            help="Download filename for CSV saved on each user's machine.",
        )

    if st.session_state.notice:
        st.info(st.session_state.notice)

    all_coords: Set[Coord] = st.session_state.all_coords
    active_coords: Set[Coord] = st.session_state.active_coords
    selected_coords: Set[Coord] = st.session_state.selected_coords

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total points", len(all_coords))
    c2.metric("Active points", len(active_coords))
    c3.metric("Removed points", len(all_coords) - len(active_coords))
    c4.metric("Selected points", len(selected_coords))

    df = points_to_df(all_coords, active_coords)
    active_df = df[df["status"] == "active"]
    removed_df = df[df["status"] == "removed"]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=active_df["beta"],
            y=active_df["alpha"],
            mode="markers",
            marker={"size": 8, "color": "#1565c0"},
            name="active",
            customdata=active_df[["alpha", "beta"]].values,
        )
    )
    if not removed_df.empty:
        fig.add_trace(
            go.Scattergl(
                x=removed_df["beta"],
                y=removed_df["alpha"],
                mode="markers",
                marker={"size": 8, "color": "#b71c1c", "symbol": "x"},
                name="removed",
                customdata=removed_df[["alpha", "beta"]].values,
            )
        )

    if selected_coords:
        sel_points = sorted(selected_coords)
        fig.add_trace(
            go.Scattergl(
                x=[p[1] for p in sel_points],
                y=[p[0] for p in sel_points],
                mode="markers",
                marker={"size": 13, "color": "rgba(0,0,0,0)", "line": {"color": "#39ff14", "width": 2}},
                name="selected",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="'Alpha-Beta' Parameter Space",
        xaxis_title="beta (deg)",
        yaxis_title="alpha (deg)",
        dragmode="lasso",
        height=680,
        legend={"orientation": "h", "y": 1.03, "x": 0.0},
    )
    fig.update_xaxes(showgrid=True, zeroline=True)
    fig.update_yaxes(showgrid=True, zeroline=True)

    st.caption("Select points with box/lasso, then apply Remove/Restore. Symmetry is enforced automatically.")
    selection_event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode=["box", "lasso"],
        key="ab_plot",
    )
    new_selection = extract_selected_coords(selection_event)
    if new_selection:
        st.session_state.selected_coords = new_selection
        selected_coords = new_selection

    a1, a2, a3, a4, a5 = st.columns(5)
    if a1.button("Remove Selected", use_container_width=True):
        to_remove = apply_symmetry(st.session_state.selected_coords, st.session_state.all_coords)
        st.session_state.active_coords -= to_remove
        st.session_state.notice = f"Removed {len(to_remove)} points (symmetry enforced)."
    if a2.button("Restore Selected", use_container_width=True):
        to_add = apply_symmetry(st.session_state.selected_coords, st.session_state.all_coords)
        st.session_state.active_coords |= to_add
        st.session_state.notice = f"Restored {len(to_add)} points (symmetry enforced)."
    if a3.button("Clear Selection", use_container_width=True):
        st.session_state.selected_coords = set()
    if a4.button("Restore All", use_container_width=True):
        st.session_state.active_coords = set(st.session_state.all_coords)
        st.session_state.notice = "All points restored."
    if a5.button("Finalize (Drop Removed)", use_container_width=True):
        st.session_state.all_coords = set(st.session_state.active_coords)
        st.session_state.selected_coords = set()
        st.session_state.notice = "Removed points dropped from design space."

    st.subheader("CSV Export")
    csv_df = active_points_csv_df(st.session_state.active_coords)
    st.dataframe(csv_df, use_container_width=True, height=220)

    csv_name = st.session_state.output_filename.strip() or "alpha_beta_symmetric_points.csv"
    if not csv_name.lower().endswith(".csv"):
        csv_name = f"{csv_name}.csv"

    st.download_button(
        label="Download Active Points CSV",
        data=csv_df.to_csv(index=False).encode("utf-8"),
        file_name=Path(csv_name).name,
        mime="text/csv",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
