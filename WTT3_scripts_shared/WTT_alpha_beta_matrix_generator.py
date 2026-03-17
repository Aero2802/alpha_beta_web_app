#!/usr/bin/env python3
"""Interactive symmetric alpha-beta point designer for wind tunnel planning.

Features:
- Symmetric alpha/beta grid generation from first-quadrant clustered levels.
- Delete/restore with symmetry enforcement across all quadrants.
- Drag active points to move symmetric orbits with 5 deg snapping.
- Hover highlighting + coordinate annotation.
- On-plot parameter editing and regeneration (no terminal edits required).
- Finalize button to permanently drop removed points from design space.
- Optional aero-map overlay (Fx/Fy/Fz/Mx/My/Mz) on the beta-alpha plot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np


Coord = Tuple[int, int]  # (alpha, beta)


@dataclass
class DesignConfig:
    alpha_max_abs: int = 180
    beta_max_abs: int = 90
    n_alpha_positive: int = 6
    n_beta_positive: int = 6
    alpha_exponent: float = 1.0
    beta_exponent: float = 1.0


# ----------------------
# Generation and IO
# ----------------------
def default_output_name() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"alpha_beta_symmetric_points_{ts}.csv"


def _last_output_state_file() -> Path:
    return Path(__file__).with_name(".doe_last_output.json")


def load_last_output_path(default_path: Path) -> Path:
    state_file = _last_output_state_file()
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
        raw = str(data.get("output_csv", "")).strip()
        if raw:
            return Path(raw).expanduser()
    except Exception:
        pass
    return default_path


def save_last_output_path(path: Path) -> None:
    state_file = _last_output_state_file()
    payload = {"output_csv": str(path.expanduser())}
    try:
        state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        # Non-fatal; continue without persistence.
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Symmetric alpha-beta point designer")
    parser.add_argument("--no-gui", action="store_true", help="Generate and save without interactive editor")
    parser.add_argument("--output", type=str, default="", help="Output CSV path")
    parser.add_argument(
        "--tas-values",
        type=str,
        default="12,18,24",
        help="Comma-separated TAS values, e.g. '20,30,40'.",
    )
    parser.add_argument("--alpha-max", type=int, default=180)
    parser.add_argument("--beta-max", type=int, default=90)
    parser.add_argument("--n-alpha", type=int, default=6)
    parser.add_argument("--n-beta", type=int, default=6)
    parser.add_argument("--alpha-exp", type=float, default=1.0)
    parser.add_argument("--beta-exp", type=float, default=1.0)
    return parser.parse_args()


def generate_clustered_integer_levels(max_abs: int, n_positive: int, exponent: float) -> List[int]:
    if max_abs < 1:
        return []

    n_positive = max(1, min(n_positive, max_abs))
    exponent = max(1.0, exponent)

    raw_values: List[int] = []
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


def build_first_quadrant_grid(alpha_nonnegative: Sequence[int], beta_nonnegative: Sequence[int]) -> Set[Coord]:
    return {(int(a), int(b)) for a in alpha_nonnegative for b in beta_nonnegative}


def mirror_to_all_quadrants(first_quadrant_points: Iterable[Coord]) -> Set[Coord]:
    mirrored: Set[Coord] = set()
    for alpha, beta in first_quadrant_points:
        mirrored.add((alpha, beta))
        mirrored.add((-alpha, beta))
        mirrored.add((alpha, -beta))
        mirrored.add((-alpha, -beta))
    return mirrored


def build_symmetric_points(cfg: DesignConfig) -> Tuple[Set[Coord], List[int], List[int]]:
    alpha_pos = generate_clustered_integer_levels(cfg.alpha_max_abs, cfg.n_alpha_positive, cfg.alpha_exponent)
    beta_pos = generate_clustered_integer_levels(cfg.beta_max_abs, cfg.n_beta_positive, cfg.beta_exponent)

    alpha_nonnegative = [0] + alpha_pos
    beta_nonnegative = [0] + beta_pos

    q1 = build_first_quadrant_grid(alpha_nonnegative, beta_nonnegative)
    all_points = mirror_to_all_quadrants(q1)

    return all_points, alpha_nonnegative, beta_nonnegative


def parse_tas_values_text(raw: str) -> List[float]:
    tokens = [t.strip() for t in str(raw).split(",")]
    values: List[float] = []
    seen: Set[float] = set()
    for token in tokens:
        if not token:
            continue
        value = float(token)
        if not np.isfinite(value):
            raise ValueError("TAS values must be finite numbers")
        key = float(round(value, 6))
        if key not in seen:
            seen.add(key)
            values.append(key)
    if not values:
        raise ValueError("Provide at least one TAS value")
    return values


def format_tas_value(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def save_points_csv(path: Path, rows: Iterable[Tuple[int, int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = sorted(
        {(int(alpha), int(beta), float(round(tas, 6))) for alpha, beta, tas in rows},
        key=lambda p: (p[2], p[0], p[1]),
    )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["SETPOINT", "ALPHA", "BETA", "TAS"])
        writer.writeheader()
        for idx, (alpha, beta, tas) in enumerate(points, start=1):
            writer.writerow(
                {
                    "SETPOINT": idx,
                    "ALPHA": alpha,
                    "BETA": beta,
                    "TAS": format_tas_value(tas),
                }
            )


def symmetry_orbit(coord: Coord) -> Set[Coord]:
    alpha, beta = coord
    a = abs(int(alpha))
    b = abs(int(beta))
    return {(sa * a, sb * b) for sa in (-1, 1) for sb in (-1, 1)}


def snap_to_step(value: float, step: int = 5) -> int:
    return int(round(value / step) * step)


def to_plot_array(coords: Iterable[Coord]) -> np.ndarray:
    arr = [(float(beta), float(alpha)) for alpha, beta in coords]  # x=beta, y=alpha
    if not arr:
        return np.empty((0, 2), dtype=float)
    return np.array(arr, dtype=float)


def to_sphere_array(coords: Iterable[Coord]) -> np.ndarray:
    # Map (alpha,beta) to unit sphere using x=cos(alpha)cos(beta), y=cos(alpha)sin(beta), z=sin(alpha)
    arr = []
    for alpha, beta in coords:
        a = np.deg2rad(float(alpha))
        b = np.deg2rad(float(beta))
        x = np.cos(a) * np.cos(b)
        y = np.cos(a) * np.sin(b)
        z = np.sin(a)
        arr.append((x, y, z))
    if not arr:
        return np.empty((0, 3), dtype=float)
    return np.array(arr, dtype=float)


# ----------------------
# Interactive Editor
# ----------------------
class InteractivePointEditor:
    def __init__(self, cfg: DesignConfig, output_csv: Path, tas_values: Sequence[float] | None = None):
        self.cfg = cfg
        self.output_csv = output_csv
        self.tas_values: List[float] = [float(round(v, 6)) for v in (tas_values or [12.0, 18.0, 24.0])]
        if not self.tas_values:
            self.tas_values = [0.0]
        self.tas_idx = 0
        self.tas_design_spaces: dict[float, tuple[Set[Coord], Set[Coord], Set[Tuple[int, int]]]] = {}

        self.all_coords: Set[Coord] = set()
        self.active_coords: Set[Coord] = set()
        self.history: List[Tuple[Set[Coord], Set[Coord], DesignConfig, Set[Tuple[int, int]]]] = []
        self.selected_orbit_keys: Set[Tuple[int, int]] = set()

        self.saved = False
        self.snap_step = 5
        self.enforce_symmetry = True

        # GUI state
        self.fig = None
        self.ax = None
        self.ax_sphere = None
        self.ax_summary = None
        self.ax_instructions = None
        self.active_scatter = None
        self.removed_scatter = None
        self.selected_scatter = None
        self.hover_ring = None
        self.drag_preview_scatter = None
        self.hover_annot = None
        self.status_text = None
        self.message_text = None
        self.param_help_text = None
        self.legend = None
        self.param_axis_help = {}
        self.sphere_view_elev = 22.0
        self.sphere_view_azim = 35.0

        # widgets
        self.tb_alpha_max = None
        self.tb_beta_max = None
        self.tb_n_alpha = None
        self.tb_n_beta = None
        self.tb_alpha_exp = None
        self.tb_beta_exp = None
        self.tb_output_dir = None
        self.tb_output_file = None
        self.tb_add_point = None
        self.tb_tas_values = None
        self.btn_symmetry_toggle = None
        self.btn_overlay_cycle = None
        self.btn_vehicle_toggle = None
        self.btn_tas_cycle = None
        self.selection_marker_offset = (1.5, 1.5)  # (x=beta, y=alpha) offset for selected outlines

        # Aero overlay state
        self.aero_overlay_options = ["None", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]
        self.aero_overlay_idx = 0
        self.aero_overlay_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, str]] = {}
        self.aero_overlay_source_path: Path | None = None
        self.aero_overlay_error: str | None = None
        self.aero_overlay_rows: list[dict] | None = None
        self.aero_overlay_columns: list[str] | None = None
        self.aero_overlay_mesh = None
        self.aero_overlay_colorbar = None
        self.ax_overlay_cbar = None
        self.aero_overlay_drawn_name = "None"
        self.overlay_cbar_rect = (0.698, 0.615, 0.015, 0.275)
        self.aero_overlay_units = {
            "Fx": "N",
            "Fy": "N",
            "Fz": "N",
            "Mx": "N-m",
            "My": "N-m",
            "Mz": "N-m",
        }

        # Vehicle STL overlay in sphere view
        default_stl = Path("/Users/paras.singh/Documents/EV3_Droid/Wind_tunnel_testing/Codes/WTT3/test/droid_ev3.stl")
        if not default_stl.exists():
            default_stl = Path(__file__).resolve().with_name("droid_ev3.stl")
        self.vehicle_stl_path = default_stl
        self.vehicle_mesh_tris: np.ndarray | None = None
        self.vehicle_mesh_face_scalar: np.ndarray | None = None
        self.vehicle_mesh_face_colors: np.ndarray | None = None
        self.vehicle_mesh_error: str | None = None
        self.vehicle_mesh_attempted = False
        self.show_vehicle_cad = False

        # interaction state
        self.mode: str | None = None
        self.pressed_coord: Coord | None = None
        self.press_pixel: Tuple[float, float] | None = None
        self.press_data: Tuple[float, float] | None = None
        self.drag_started = False
        self.drag_axis: str | None = None
        self.rect_start_data: Tuple[float, float] | None = None
        self.rect_patch = None

        self.preview_all_coords: Set[Coord] | None = None
        self.preview_active_coords: Set[Coord] | None = None
        self.preview_selected_coords: Set[Coord] | None = None
        self.preview_hover: Coord | None = None
        self.last_preview_drag_target: tuple[int, int] | None = None
        self.last_preview_group_target: tuple[str, int] | None = None
        self.last_sphere_refresh_ts = 0.0
        self.last_sphere_hover_coord: Coord | None = None
        self.sphere_refresh_interval_s = 0.25
        self._nearest_plot_coords = np.empty((0, 2), dtype=float)  # x=beta, y=alpha
        self._nearest_is_active = np.empty((0,), dtype=bool)
        self._sphere_initialized = False
        self._sphere_active_scatter = None
        self._sphere_removed_scatter = None
        self._sphere_selected_scatter = None
        self._sphere_hover_scatter = None
        self._sphere_boundary_lines = []
        self._sphere_last_bounds: tuple[float, float, float, float] | None = None
        self._sphere_static_vehicle_enabled: bool | None = None
        self._coord_to_sphere_cache: dict[Coord, tuple[float, float, float]] = {}

        self._regenerate_from_cfg(clear_history=True)
        self._capture_current_tas_state()
        base_all = set(self.all_coords)
        base_active = set(self.active_coords)
        for tas in self.tas_values:
            key = float(round(tas, 6))
            if key not in self.tas_design_spaces:
                self.tas_design_spaces[key] = (set(base_all), set(base_active), set())

    # ---------- state + history ----------
    def _current_tas(self) -> float:
        if not self.tas_values:
            return 0.0
        return self.tas_values[self.tas_idx]

    def _tas_values_to_text(self) -> str:
        return ",".join(format_tas_value(v) for v in self.tas_values)

    def _capture_current_tas_state(self) -> None:
        tas = float(round(self._current_tas(), 6))
        self.tas_design_spaces[tas] = (
            set(self.all_coords),
            set(self.active_coords),
            set(self.selected_orbit_keys),
        )

    def _load_tas_state(self, tas: float) -> None:
        key = float(round(tas, 6))
        if key in self.tas_design_spaces:
            all_coords, active_coords, selected_keys = self.tas_design_spaces[key]
            self.all_coords = set(all_coords)
            self.active_coords = set(active_coords)
            self.selected_orbit_keys = set(selected_keys)
            self._prune_selection()
            return

        points, _, _ = build_symmetric_points(self.cfg)
        self.all_coords = set(points)
        self.active_coords = set(points)
        self.selected_orbit_keys.clear()
        self.tas_design_spaces[key] = (set(self.all_coords), set(self.active_coords), set())

    def _update_tas_button_label(self) -> None:
        if self.btn_tas_cycle is None:
            return
        self.btn_tas_cycle.label.set_text(f"TAS: {format_tas_value(self._current_tas())}")

    def _cycle_tas(self, event=None) -> None:
        if not self.tas_values:
            return
        step = 1
        if event is not None and getattr(event, "button", None) == 3:
            step = -1

        self._capture_current_tas_state()
        self.tas_idx = (self.tas_idx + step) % len(self.tas_values)
        self._load_tas_state(self._current_tas())
        self._clear_preview()
        self.history.clear()
        self._update_tas_button_label()
        self._set_message(f"Active TAS: {format_tas_value(self._current_tas())}")
        self._refresh()

    def _apply_tas_values(self, _event=None) -> None:
        if self.tb_tas_values is None:
            return
        try:
            values = parse_tas_values_text(self.tb_tas_values.text)
        except ValueError as err:
            self._set_message(f"TAS values error: {err}")
            self.fig.canvas.draw_idle()
            return

        current_tas = float(round(self._current_tas(), 6))
        self._capture_current_tas_state()

        new_spaces: dict[float, tuple[Set[Coord], Set[Coord], Set[Tuple[int, int]]]] = {}
        for tas in values:
            key = float(round(tas, 6))
            if key in self.tas_design_spaces:
                new_spaces[key] = self.tas_design_spaces[key]
            else:
                points, _, _ = build_symmetric_points(self.cfg)
                new_spaces[key] = (set(points), set(points), set())

        self.tas_values = values
        self.tas_design_spaces = new_spaces
        if current_tas in self.tas_values:
            self.tas_idx = self.tas_values.index(current_tas)
        else:
            self.tas_idx = 0
        self._load_tas_state(self._current_tas())
        self._clear_preview()
        self.history.clear()
        self._update_tas_button_label()
        self._sync_textboxes_from_cfg()
        self._set_message(f"Applied {len(self.tas_values)} TAS values. Active TAS: {format_tas_value(self._current_tas())}")
        self._refresh()

    def _clone_cfg(self) -> DesignConfig:
        return DesignConfig(
            alpha_max_abs=int(self.cfg.alpha_max_abs),
            beta_max_abs=int(self.cfg.beta_max_abs),
            n_alpha_positive=int(self.cfg.n_alpha_positive),
            n_beta_positive=int(self.cfg.n_beta_positive),
            alpha_exponent=float(self.cfg.alpha_exponent),
            beta_exponent=float(self.cfg.beta_exponent),
        )

    def _orbit_key(self, coord: Coord) -> Tuple[int, int]:
        return (abs(int(coord[0])), abs(int(coord[1])))

    def _active_orbit_keys(self) -> Set[Tuple[int, int]]:
        return {self._orbit_key(c) for c in self.active_coords}

    def _selected_coords_for_keys(self, keys: Set[Tuple[int, int]], all_coords: Set[Coord], active_coords: Set[Coord]) -> Set[Coord]:
        selected: Set[Coord] = set()
        for a_abs, b_abs in keys:
            selected |= (symmetry_orbit((a_abs, b_abs)) & active_coords & all_coords)
        return selected

    def _prune_selection(self) -> None:
        self.selected_orbit_keys &= self._active_orbit_keys()

    def _clear_selection(self) -> None:
        self.selected_orbit_keys.clear()

    def _toggle_orbit_selection(self, coord: Coord) -> None:
        key = self._orbit_key(coord)
        if key in self.selected_orbit_keys:
            self.selected_orbit_keys.remove(key)
            self._set_message(f"Unselected orbit |alpha|={key[0]}, |beta|={key[1]}")
        else:
            self.selected_orbit_keys.add(key)
            self._set_message(f"Selected orbit |alpha|={key[0]}, |beta|={key[1]}")
        self._clear_preview()
        self._refresh()

    @staticmethod
    def _event_has_shift(event) -> bool:
        key = getattr(event, "key", None)
        return isinstance(key, str) and ("shift" in key.lower())

    def _push_history(self) -> None:
        self.history.append((set(self.all_coords), set(self.active_coords), self._clone_cfg(), set(self.selected_orbit_keys)))

    def _undo_last(self) -> None:
        if not self.history:
            return
        self.all_coords, self.active_coords, self.cfg, self.selected_orbit_keys = self.history.pop()
        self._prune_selection()
        self._sync_textboxes_from_cfg()
        self._set_message("Undo applied.")
        self._clear_preview()
        self._refresh()

    def _clear_preview(self) -> None:
        self.preview_all_coords = None
        self.preview_active_coords = None
        self.preview_selected_coords = None
        self.last_preview_drag_target = None
        self.last_preview_group_target = None
        if self.drag_preview_scatter is not None:
            self.drag_preview_scatter.set_offsets(np.empty((0, 2), dtype=float))

    def _effective_sets(self) -> Tuple[Set[Coord], Set[Coord]]:
        if self.preview_all_coords is not None and self.preview_active_coords is not None:
            return self.preview_all_coords, self.preview_active_coords
        return self.all_coords, self.active_coords

    def _set_message(self, msg: str) -> None:
        if self.message_text is not None:
            self.message_text.set_text(msg)

    def _update_drag_preview_marker(self, target_alpha: int, target_beta: int, source_coord: Coord | None = None) -> None:
        if self.drag_preview_scatter is None:
            return
        if self.enforce_symmetry and (source_coord is None or self._is_complete_orbit(source_coord)):
            preview_coords = symmetry_orbit((target_alpha, target_beta))
        else:
            preview_coords = {(target_alpha, target_beta)}
        self.drag_preview_scatter.set_offsets(to_plot_array(preview_coords))
        self.fig.canvas.draw_idle()

    # ---------- generation ----------
    def _regenerate_from_cfg(self, clear_history: bool = False) -> None:
        points, _, _ = build_symmetric_points(self.cfg)
        self.all_coords = set(points)
        self.active_coords = set(points)
        self._clear_selection()
        if clear_history:
            self.history.clear()

    def _orbit_in_all(self, coord: Coord, all_coords: Set[Coord] | None = None) -> Set[Coord]:
        base = self.all_coords if all_coords is None else all_coords
        return {c for c in symmetry_orbit(coord) if c in base}

    def _is_complete_orbit(self, coord: Coord, all_coords: Set[Coord] | None = None) -> bool:
        base = self.all_coords if all_coords is None else all_coords
        if coord not in base:
            return False
        return symmetry_orbit(coord) <= base

    def _group_in_all(self, coord: Coord, all_coords: Set[Coord] | None = None) -> Set[Coord]:
        base = self.all_coords if all_coords is None else all_coords
        if coord not in base:
            return set()
        if self.enforce_symmetry and self._is_complete_orbit(coord, base):
            return {c for c in symmetry_orbit(coord) if c in base}
        return {coord}

    def _mode_name(self) -> str:
        return "Mirror ON (symmetric edits)" if self.enforce_symmetry else "Mirror OFF (asymmetric edits)"

    def _update_mode_button_label(self) -> None:
        if self.btn_symmetry_toggle is None:
            return
        self.btn_symmetry_toggle.label.set_text("Mirror: ON" if self.enforce_symmetry else "Mirror: OFF")

    def _toggle_symmetry_mode(self, _event=None) -> None:
        self.enforce_symmetry = not self.enforce_symmetry
        self._clear_selection()
        self._clear_preview()
        self._update_mode_button_label()
        self._set_message(f"Edit mode changed: {self._mode_name()}")
        self._refresh()

    def _current_overlay_name(self) -> str:
        return self.aero_overlay_options[self.aero_overlay_idx]

    def _update_overlay_button_label(self) -> None:
        if self.btn_overlay_cycle is None:
            return
        self.btn_overlay_cycle.label.set_text(f"Overlay: {self._current_overlay_name()}")

    def _update_vehicle_button_label(self) -> None:
        if self.btn_vehicle_toggle is None:
            return
        self.btn_vehicle_toggle.label.set_text("Droid CAD: ON" if self.show_vehicle_cad else "Droid CAD: OFF")

    def _toggle_vehicle_cad(self, _event=None) -> None:
        self.show_vehicle_cad = not self.show_vehicle_cad
        self._sphere_initialized = False
        self._sphere_static_vehicle_enabled = None
        self._update_vehicle_button_label()
        self._set_message("Vehicle CAD in sphere: ON" if self.show_vehicle_cad else "Vehicle CAD in sphere: OFF")
        self._refresh(include_sphere=True, lightweight=False)

    def _cycle_overlay(self, event=None) -> None:
        step = 1
        if event is not None and getattr(event, "button", None) == 3:
            step = -1
        self.aero_overlay_idx = (self.aero_overlay_idx + step) % len(self.aero_overlay_options)
        self._update_overlay_button_label()
        self._set_message(f"Aero overlay: {self._current_overlay_name()}")
        self._refresh()

    @staticmethod
    def _find_latest_cfd_csv(base_dir: Path) -> Path | None:
        cfd_dir = base_dir / "CFD_data"
        if not cfd_dir.exists():
            return None
        csv_files = list(cfd_dir.glob("*.csv"))
        if not csv_files:
            return None
        return max(csv_files, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def _resolve_aero_column(columns: Sequence[str], desired: str) -> str | None:
        aliases = {"FY_Droid (N)_ang": "FY_Droid (N)_avg"}
        columns_set = set(columns)
        if desired in columns_set:
            return desired
        alias = aliases.get(desired)
        if alias and alias in columns_set:
            return alias
        return None

    @staticmethod
    def _build_aero_grid(rows: list[dict], value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sums_by_point: dict[tuple[float, float], float] = {}
        counts_by_point: dict[tuple[float, float], int] = {}

        for row in rows:
            try:
                alpha = float(row.get("Alpha", ""))
                beta = float(row.get("Beta", ""))
                value = float(row.get(value_col, ""))
            except (TypeError, ValueError):
                continue
            key = (alpha, beta)
            sums_by_point[key] = sums_by_point.get(key, 0.0) + value
            counts_by_point[key] = counts_by_point.get(key, 0) + 1

        if not sums_by_point:
            return np.array([]), np.array([]), np.empty((0, 0))

        averaged_points = []
        for (alpha, beta), total in sums_by_point.items():
            averaged_points.append((alpha, beta, total / counts_by_point[(alpha, beta)]))

        alpha_values = np.array(sorted({p[0] for p in averaged_points}), dtype=float)
        beta_values = np.array(sorted({p[1] for p in averaged_points}), dtype=float)
        alpha_index = {v: i for i, v in enumerate(alpha_values)}
        beta_index = {v: i for i, v in enumerate(beta_values)}

        grid = np.full((len(alpha_values), len(beta_values)), np.nan, dtype=float)
        for alpha, beta, value in averaged_points:
            grid[alpha_index[alpha], beta_index[beta]] = value

        return alpha_values, beta_values, grid

    def _ensure_aero_overlay_data(self, overlay_name: str) -> bool:
        if overlay_name in self.aero_overlay_data:
            return True
        if self.aero_overlay_error is not None:
            return False

        try:
            if self.aero_overlay_rows is None or self.aero_overlay_columns is None:
                base_dir = Path(__file__).resolve().parent
                source_path = self._find_latest_cfd_csv(base_dir)
                if source_path is None:
                    raise FileNotFoundError(f"No CFD CSV found in {(base_dir / 'CFD_data')}")
                with source_path.open("r", newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    if reader.fieldnames is None:
                        raise ValueError("Aero CSV has no header.")
                    self.aero_overlay_rows = list(reader)
                    self.aero_overlay_columns = [str(c) for c in reader.fieldnames if c is not None]
                    self.aero_overlay_source_path = source_path

            rows = self.aero_overlay_rows
            columns = self.aero_overlay_columns
            if rows is None or columns is None:
                raise ValueError("Aero data cache was not initialized.")
            if "Alpha" not in columns or "Beta" not in columns:
                raise ValueError("Aero CSV must contain Alpha and Beta columns.")

            mapping = {
                "Fx": "FX_Droid (N)_avg",
                "Fy": "FY_Droid (N)_ang",
                "Fz": "FZ_Droid (N)_avg",
                "Mx": "MX_Droid (N-m)_avg",
                "My": "MY_Droid (N-m)_avg",
                "Mz": "MZ_Droid (N-m)_avg",
            }
            desired_col = mapping.get(overlay_name)
            if desired_col is None:
                raise ValueError(f"Unknown overlay '{overlay_name}'.")

            source_col = self._resolve_aero_column(columns, desired_col)
            if source_col is None:
                raise ValueError(f"Missing aero column for {overlay_name}: '{desired_col}'")
            alpha_vals, beta_vals, grid = self._build_aero_grid(rows, source_col)
            if grid.size == 0:
                raise ValueError(f"No aero data points found for {overlay_name} ({source_col}).")
            self.aero_overlay_data[overlay_name] = (alpha_vals, beta_vals, grid, source_col)
            return True
        except Exception as err:
            self.aero_overlay_error = str(err)
            return False

    def _clear_overlay_artists(self) -> None:
        if self.aero_overlay_mesh is not None:
            try:
                self.aero_overlay_mesh.remove()
            except Exception:
                pass
            self.aero_overlay_mesh = None

        cax = self._ensure_overlay_cbar_axis()
        if cax is not None:
            cax.set_visible(False)
        if self.aero_overlay_colorbar is not None:
            self.aero_overlay_colorbar.ax.set_visible(False)
        self.aero_overlay_drawn_name = "None"

    def _ensure_overlay_cbar_axis(self):
        if self.fig is None:
            return None
        if self.ax_overlay_cbar is None or getattr(self.ax_overlay_cbar, "figure", None) is None:
            self.ax_overlay_cbar = self.fig.add_axes(list(self.overlay_cbar_rect))
            self.ax_overlay_cbar.set_visible(False)
        return self.ax_overlay_cbar

    def _refresh_overlay(self) -> None:
        cax = self._ensure_overlay_cbar_axis()
        if cax is None:
            return

        overlay = self._current_overlay_name()
        if overlay == "None":
            self._clear_overlay_artists()
            return

        if self.aero_overlay_mesh is not None and self.aero_overlay_drawn_name == overlay:
            return

        self._clear_overlay_artists()

        ok = self._ensure_aero_overlay_data(overlay)
        if not ok:
            self._set_message(f"Aero overlay unavailable: {self.aero_overlay_error}")
            return

        if overlay not in self.aero_overlay_data:
            return

        alpha_vals, beta_vals, grid, _source_col = self.aero_overlay_data[overlay]
        masked = np.ma.masked_invalid(grid)
        self.aero_overlay_mesh = self.ax.pcolormesh(
            beta_vals,
            alpha_vals,
            masked,
            shading="auto",
            cmap="viridis",
            alpha=0.45,
            zorder=0.1,
        )
        cax.set_visible(True)
        if self.aero_overlay_colorbar is None:
            self.aero_overlay_colorbar = self.fig.colorbar(self.aero_overlay_mesh, cax=cax)
        else:
            self.aero_overlay_colorbar.update_normal(self.aero_overlay_mesh)
            self.aero_overlay_colorbar.ax.set_visible(True)
        unit = self.aero_overlay_units.get(overlay, "")
        label = f"{overlay} ({unit})" if unit else overlay
        self.aero_overlay_colorbar.set_label(label, fontsize=8)
        self.aero_overlay_colorbar.ax.tick_params(labelsize=7)
        cax.set_title(label, fontsize=8, pad=4)
        self.aero_overlay_drawn_name = overlay

    @staticmethod
    def _parse_binary_stl(data: bytes) -> np.ndarray | None:
        if len(data) < 84:
            return None
        tri_count = struct.unpack_from("<I", data, 80)[0]
        expected = 84 + tri_count * 50
        if expected != len(data):
            return None

        tris = np.empty((tri_count, 3, 3), dtype=float)
        off = 84
        for i in range(tri_count):
            off += 12  # facet normal
            for j in range(3):
                x, y, z = struct.unpack_from("<3f", data, off)
                tris[i, j, 0] = x
                tris[i, j, 1] = y
                tris[i, j, 2] = z
                off += 12
            off += 2  # attribute byte count
        return tris

    @staticmethod
    def _parse_ascii_stl(data: bytes) -> np.ndarray | None:
        text = data.decode("utf-8", errors="ignore")
        verts: list[tuple[float, float, float]] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line.lower().startswith("vertex"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                x = float(parts[-3])
                y = float(parts[-2])
                z = float(parts[-1])
            except ValueError:
                continue
            verts.append((x, y, z))

        if len(verts) < 3:
            return None
        usable = (len(verts) // 3) * 3
        if usable == 0:
            return None
        return np.array(verts[:usable], dtype=float).reshape((-1, 3, 3))

    @staticmethod
    def _to_indexed_mesh(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flat_vertices = triangles.reshape(-1, 3)
        unique_vertices, inverse = np.unique(flat_vertices, axis=0, return_inverse=True)
        face_indices = inverse.reshape(-1, 3)
        return unique_vertices.astype(np.float64), face_indices

    @staticmethod
    def _build_vertex_neighbors(face_indices: np.ndarray, vertex_count: int) -> list[np.ndarray]:
        neighbors: list[set[int]] = [set() for _ in range(vertex_count)]
        for a, b, c in face_indices:
            neighbors[a].update((int(b), int(c)))
            neighbors[b].update((int(a), int(c)))
            neighbors[c].update((int(a), int(b)))
        return [np.fromiter(n, dtype=np.int32) for n in neighbors]

    @staticmethod
    def _laplacian_step(vertices: np.ndarray, neighbors: list[np.ndarray], weight: float) -> np.ndarray:
        updated = vertices.copy()
        for idx, nbrs in enumerate(neighbors):
            if len(nbrs) == 0:
                continue
            mean_neighbor = vertices[nbrs].mean(axis=0)
            updated[idx] = vertices[idx] + weight * (mean_neighbor - vertices[idx])
        return updated

    def _taubin_smooth(self, vertices: np.ndarray, neighbors: list[np.ndarray], iterations: int) -> np.ndarray:
        if iterations <= 0:
            return vertices
        smoothed = vertices.copy()
        for _ in range(iterations):
            smoothed = self._laplacian_step(smoothed, neighbors, 0.5)
            smoothed = self._laplacian_step(smoothed, neighbors, -0.53)
        return smoothed

    @staticmethod
    def _sample_face_indices(face_indices: np.ndarray, max_faces: int) -> np.ndarray:
        if max_faces <= 0 or len(face_indices) <= max_faces:
            return face_indices
        step = int(math.ceil(len(face_indices) / max_faces))
        return face_indices[::step]

    def _prepare_vehicle_mesh(self, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vertices, face_indices = self._to_indexed_mesh(tris)

        # Reference approach from plot_vehicle_stl_iso.py for smoother appearance.
        neighbors = self._build_vertex_neighbors(face_indices, len(vertices))
        vertices = self._taubin_smooth(vertices, neighbors, iterations=8)

        # Keep dense surface coverage to avoid patchy/hollow appearance.
        face_indices_to_plot = self._sample_face_indices(face_indices, max_faces=50000)
        triangles = vertices[face_indices_to_plot]

        # Center and normalize to fit in unit sphere while keeping shape proportions.
        center = np.mean(vertices, axis=0)
        triangles = triangles - center

        # Fixed CAD orientation adjustment: rotate 180 deg about +Z.
        # This flips X/Y while keeping Z unchanged.
        triangles[..., 0] *= -1.0
        triangles[..., 1] *= -1.0

        radii = np.linalg.norm(triangles.reshape(-1, 3), axis=1)
        rmax = float(np.max(radii)) if radii.size else 0.0
        if not np.isfinite(rmax) or rmax <= 1e-12:
            raise ValueError("STL mesh has invalid scale.")
        triangles *= (0.80 / rmax)

        # Solid light-blue body with subtle Lambertian shading.
        v1 = triangles[:, 1, :] - triangles[:, 0, :]
        v2 = triangles[:, 2, :] - triangles[:, 0, :]
        normals = np.cross(v1, v2)
        nmag = np.linalg.norm(normals, axis=1, keepdims=True)
        nmag[nmag < 1e-12] = 1e-12
        normals = normals / nmag

        light_dir = np.array([0.35, -0.25, 0.90], dtype=np.float64)
        light_dir = light_dir / np.linalg.norm(light_dir)
        intensity = np.clip(normals @ light_dir, 0.0, 1.0)
        shade = 0.70 + 0.30 * intensity

        base_rgb = np.array([0.73, 0.86, 0.97], dtype=np.float64)
        rgb = np.clip(base_rgb[None, :] * shade[:, None], 0.0, 1.0)
        alpha = np.full((rgb.shape[0], 1), 0.98, dtype=np.float64)
        face_colors = np.concatenate([rgb, alpha], axis=1)

        face_scalar = triangles[:, :, 2].mean(axis=1)
        return triangles.astype(np.float32), face_scalar.astype(np.float32), face_colors.astype(np.float32)

    def _ensure_vehicle_mesh(self) -> np.ndarray | None:
        if self.vehicle_mesh_tris is not None:
            return self.vehicle_mesh_tris
        if self.vehicle_mesh_attempted:
            return None

        self.vehicle_mesh_attempted = True
        try:
            if not self.vehicle_stl_path.exists():
                raise FileNotFoundError(f"Vehicle STL not found: {self.vehicle_stl_path}")

            data = self.vehicle_stl_path.read_bytes()
            tris = self._parse_binary_stl(data)
            if tris is None:
                tris = self._parse_ascii_stl(data)
            if tris is None or tris.size == 0:
                raise ValueError("Could not parse STL triangles.")

            prepared_tris, face_scalar, face_colors = self._prepare_vehicle_mesh(tris)
            self.vehicle_mesh_tris = prepared_tris
            self.vehicle_mesh_face_scalar = face_scalar
            self.vehicle_mesh_face_colors = face_colors
            self.vehicle_mesh_error = None
            return self.vehicle_mesh_tris
        except Exception as err:
            self.vehicle_mesh_error = str(err)
            self.vehicle_mesh_tris = None
            self.vehicle_mesh_face_scalar = None
            self.vehicle_mesh_face_colors = None
            return None

    # ---------- plotting ----------
    def _autoscale_axes(self, coords: Set[Coord]) -> None:
        if self.ax is None:
            return
        if not coords:
            target_xlim = (-20, 20)
            target_ylim = (-20, 20)
            if self.ax.get_xlim() != target_xlim:
                self.ax.set_xlim(*target_xlim)
            if self.ax.get_ylim() != target_ylim:
                self.ax.set_ylim(*target_ylim)
            return

        max_alpha = max(abs(alpha) for alpha, _ in coords)
        max_beta = max(abs(beta) for _, beta in coords)
        margin = max(5, self.snap_step)

        target_xlim = (-(max_beta + margin), (max_beta + margin))
        target_ylim = (-(max_alpha + margin), (max_alpha + margin))
        if self.ax.get_xlim() != target_xlim:
            self.ax.set_xlim(*target_xlim)
        if self.ax.get_ylim() != target_ylim:
            self.ax.set_ylim(*target_ylim)

    def _format_output_path(self, path: Path) -> str:
        path_str = str(path)
        if len(path_str) <= 64:
            return path_str
        return f"{path_str[:24]}...{path_str[-35:]}"

    def _update_nearest_cache_from_arrays(self, active_plot: np.ndarray, removed_plot: np.ndarray) -> None:
        if active_plot.size and removed_plot.size:
            self._nearest_plot_coords = np.vstack([active_plot, removed_plot])
            self._nearest_is_active = np.concatenate(
                [np.ones(len(active_plot), dtype=bool), np.zeros(len(removed_plot), dtype=bool)]
            )
        elif active_plot.size:
            self._nearest_plot_coords = active_plot
            self._nearest_is_active = np.ones(len(active_plot), dtype=bool)
        elif removed_plot.size:
            self._nearest_plot_coords = removed_plot
            self._nearest_is_active = np.zeros(len(removed_plot), dtype=bool)
        else:
            self._nearest_plot_coords = np.empty((0, 2), dtype=float)
            self._nearest_is_active = np.empty((0,), dtype=bool)

    def _coords_to_sphere_array_cached(self, coords: Set[Coord]) -> np.ndarray:
        if not coords:
            return np.empty((0, 3), dtype=float)
        arr = np.empty((len(coords), 3), dtype=float)
        for idx, coord in enumerate(coords):
            cached = self._coord_to_sphere_cache.get(coord)
            if cached is None:
                alpha, beta = coord
                a = np.deg2rad(float(alpha))
                b = np.deg2rad(float(beta))
                cached = (
                    float(np.cos(a) * np.cos(b)),
                    float(np.cos(a) * np.sin(b)),
                    float(np.sin(a)),
                )
                self._coord_to_sphere_cache[coord] = cached
            arr[idx, :] = cached
        return arr

    @staticmethod
    def _set_3d_scatter_points(scatter, xyz: np.ndarray) -> None:
        if scatter is None:
            return
        if xyz.size:
            scatter._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
        else:
            empty = np.empty((0,), dtype=float)
            scatter._offsets3d = (empty, empty, empty)

    def _ensure_sphere_artists(self) -> None:
        if self.ax_sphere is None:
            return
        ax = self.ax_sphere
        if (
            self._sphere_initialized
            and self._sphere_active_scatter is not None
            and getattr(self._sphere_active_scatter, "axes", None) is ax
            and self._sphere_static_vehicle_enabled == self.show_vehicle_cad
        ):
            return

        ax.cla()

        # Build the static sphere scene once; subsequent refreshes only update dynamic artists.
        lon = np.deg2rad(np.linspace(-180.0, 180.0, 72))
        lat = np.deg2rad(np.linspace(-90.0, 90.0, 42))
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        x = np.cos(lat_grid) * np.cos(lon_grid)
        y = np.cos(lat_grid) * np.sin(lon_grid)
        z = np.sin(lat_grid)
        ax.plot_surface(x, y, z, color="#eceff1", alpha=0.10, linewidth=0.0, shade=False)

        vehicle_tris = self._ensure_vehicle_mesh() if self.show_vehicle_cad else None
        if vehicle_tris is not None and vehicle_tris.size:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            face_colors = self.vehicle_mesh_face_colors
            if face_colors is None or len(face_colors) != len(vehicle_tris):
                face_colors = np.tile(np.array([[0.90, 0.89, 0.85, 0.98]], dtype=np.float32), (len(vehicle_tris), 1))

            vehicle_poly = Poly3DCollection(
                vehicle_tris,
                facecolors=face_colors,
                edgecolors="none",
                linewidths=0.0,
                alpha=1.0,
                antialiased=True,
            )
            vehicle_poly.set_zsort("average")
            ax.add_collection3d(vehicle_poly)

        lon_line = np.deg2rad(np.linspace(-180.0, 180.0, 160))
        for lat_deg in (-60, -30, 0, 30, 60):
            lat_rad = np.deg2rad(lat_deg)
            gx = np.cos(lat_rad) * np.cos(lon_line)
            gy = np.cos(lat_rad) * np.sin(lon_line)
            gz = np.full_like(gx, np.sin(lat_rad))
            ax.plot(gx, gy, gz, color="#b0bec5", linewidth=0.55, alpha=0.65)

        lat_line = np.deg2rad(np.linspace(-90.0, 90.0, 140))
        for lon_deg in (-120, -90, -60, -30, 0, 30, 60, 90, 120):
            lon_rad = np.deg2rad(lon_deg)
            gx = np.cos(lat_line) * np.cos(lon_rad)
            gy = np.cos(lat_line) * np.sin(lon_rad)
            gz = np.sin(lat_line)
            ax.plot(gx, gy, gz, color="#cfd8dc", linewidth=0.45, alpha=0.55)

        alpha_axis = np.deg2rad(np.linspace(-90.0, 90.0, 240))
        ax.plot(np.cos(alpha_axis), np.zeros_like(alpha_axis), np.sin(alpha_axis), color="black", linewidth=1.3, alpha=1.0)
        beta_axis = np.deg2rad(np.linspace(-180.0, 180.0, 360))
        ax.plot(np.cos(beta_axis), np.sin(beta_axis), np.zeros_like(beta_axis), color="black", linewidth=1.3, alpha=1.0)

        self._sphere_boundary_lines = [
            ax.plot([], [], [], color="#455a64", linewidth=1.1, alpha=0.9)[0],
            ax.plot([], [], [], color="#455a64", linewidth=1.1, alpha=0.9)[0],
            ax.plot([], [], [], color="#455a64", linewidth=1.1, alpha=0.9)[0],
            ax.plot([], [], [], color="#455a64", linewidth=1.1, alpha=0.9)[0],
        ]

        self._sphere_active_scatter = ax.scatter([], [], [], s=18, c="#1565c0", depthshade=False)
        self._sphere_removed_scatter = ax.scatter([], [], [], s=18, c="#b71c1c", marker="x", depthshade=False)
        self._sphere_selected_scatter = ax.scatter(
            [], [], [], s=36, facecolors="none", edgecolors="#39ff14", linewidths=1.2, depthshade=False
        )
        self._sphere_hover_scatter = ax.scatter(
            [], [], [], s=90, c="#ffb300", edgecolors="#000000", linewidths=0.8, depthshade=False, zorder=10
        )

        ax.scatter([0.0], [0.0], [1.0], c="#2e7d32", s=18, depthshade=False)
        ax.scatter([0.0], [0.0], [-1.0], c="#2e7d32", s=18, depthshade=False)
        ax.scatter([0.0], [1.0], [0.0], c="#ef6c00", s=18, depthshade=False)
        ax.scatter([0.0], [-1.0], [0.0], c="#ef6c00", s=18, depthshade=False)
        ax.text(0.0, 0.0, 1.08, "-alpha", color="#2e7d32", fontsize=8)
        ax.text(0.0, 0.0, -1.14, "+alpha", color="#2e7d32", fontsize=8)
        ax.text(0.0, 1.08, 0.0, "-beta", color="#ef6c00", fontsize=8)
        ax.text(0.0, -1.18, 0.0, "+beta", color="#ef6c00", fontsize=8)

        ax.set_title("Sphere (lat=alpha, lon=beta)", fontsize=9.2, y=1.06, pad=2.0)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_zlim(-1.05, 1.05)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.grid(False)
        ax.view_init(elev=self.sphere_view_elev, azim=self.sphere_view_azim)

        self._sphere_last_bounds = None
        self._sphere_static_vehicle_enabled = self.show_vehicle_cad
        self._sphere_initialized = True

    def _refresh_sphere(
        self,
        all_coords: Set[Coord],
        active_coords: Set[Coord],
        selected_coords: Set[Coord],
        hover_coord: Coord | None,
    ) -> None:
        if self.ax_sphere is None:
            return

        ax = self.ax_sphere
        if hasattr(ax, "elev") and hasattr(ax, "azim"):
            self.sphere_view_elev = float(ax.elev)
            self.sphere_view_azim = float(ax.azim)
        self._ensure_sphere_artists()

        removed_coords = all_coords - active_coords
        active_xyz = self._coords_to_sphere_array_cached(active_coords)
        removed_xyz = self._coords_to_sphere_array_cached(removed_coords)
        selected_xyz = self._coords_to_sphere_array_cached(selected_coords & active_coords)
        self._set_3d_scatter_points(self._sphere_active_scatter, active_xyz)
        self._set_3d_scatter_points(self._sphere_removed_scatter, removed_xyz)
        self._set_3d_scatter_points(self._sphere_selected_scatter, selected_xyz)

        if hover_coord is not None:
            hover_xyz = self._coords_to_sphere_array_cached({hover_coord})
            self._set_3d_scatter_points(self._sphere_hover_scatter, hover_xyz)
        else:
            self._set_3d_scatter_points(self._sphere_hover_scatter, np.empty((0, 3), dtype=float))

        if all_coords:
            alpha_vals = np.array([c[0] for c in all_coords], dtype=float)
            beta_vals = np.array([c[1] for c in all_coords], dtype=float)
            a_min, a_max = np.min(alpha_vals), np.max(alpha_vals)
            b_min, b_max = np.min(beta_vals), np.max(beta_vals)
            bounds = (float(a_min), float(a_max), float(b_min), float(b_max))
            if bounds != self._sphere_last_bounds and len(self._sphere_boundary_lines) == 4:
                beta_range = np.deg2rad(np.linspace(b_min, b_max, 120))
                for idx, a in enumerate((a_min, a_max)):
                    a_rad = np.deg2rad(a)
                    bx = np.cos(a_rad) * np.cos(beta_range)
                    by = np.cos(a_rad) * np.sin(beta_range)
                    bz = np.full_like(bx, np.sin(a_rad))
                    self._sphere_boundary_lines[idx].set_data_3d(bx, by, bz)

                alpha_range = np.deg2rad(np.linspace(a_min, a_max, 120))
                for idx, b in enumerate((b_min, b_max), start=2):
                    b_rad = np.deg2rad(b)
                    bx = np.cos(alpha_range) * np.cos(b_rad)
                    by = np.cos(alpha_range) * np.sin(b_rad)
                    bz = np.sin(alpha_range)
                    self._sphere_boundary_lines[idx].set_data_3d(bx, by, bz)
                self._sphere_last_bounds = bounds
        else:
            if self._sphere_boundary_lines:
                empty = np.empty((0,), dtype=float)
                for ln in self._sphere_boundary_lines:
                    ln.set_data_3d(empty, empty, empty)
            self._sphere_last_bounds = None

        ax.view_init(elev=self.sphere_view_elev, azim=self.sphere_view_azim)

    def _refresh(self, include_sphere: bool = True, lightweight: bool = False) -> None:
        all_coords, active_coords = self._effective_sets()
        removed_coords = all_coords - active_coords

        active_plot = to_plot_array(active_coords)
        removed_plot = to_plot_array(removed_coords)
        self.active_scatter.set_offsets(active_plot)
        self.removed_scatter.set_offsets(removed_plot)
        self._update_nearest_cache_from_arrays(active_plot, removed_plot)
        if self.preview_selected_coords is not None:
            selected_coords = self.preview_selected_coords & active_coords & all_coords
        else:
            selected_coords = self._selected_coords_for_keys(self.selected_orbit_keys, all_coords, active_coords)
        selected_plot = to_plot_array(selected_coords)
        if selected_plot.size:
            selected_plot[:, 0] += self.selection_marker_offset[0]
            selected_plot[:, 1] += self.selection_marker_offset[1]
        self.selected_scatter.set_offsets(selected_plot)

        if self.preview_hover is not None:
            alpha, beta = self.preview_hover
            self.hover_ring.set_offsets(np.array([[float(beta), float(alpha)]], dtype=float))
        else:
            self.hover_ring.set_offsets(np.empty((0, 2), dtype=float))

        if not lightweight:
            self._refresh_overlay()
            self._autoscale_axes(all_coords)
        if include_sphere:
            self._refresh_sphere(all_coords, active_coords, selected_coords, self.preview_hover)
            self.last_sphere_refresh_ts = time.monotonic()
            self.last_sphere_hover_coord = self.preview_hover

        if not lightweight:
            wrapped_path = self._format_output_path(self._current_output_path())
            if not self.show_vehicle_cad:
                stl_status = "hidden"
            elif self.vehicle_mesh_tris is not None:
                stl_status = f"loaded ({self.vehicle_mesh_tris.shape[0]} faces)"
            elif self.vehicle_mesh_error is not None:
                stl_status = "error"
            else:
                stl_status = "pending"
            status = "\n".join(
                [
                    "Point Summary",
                    f"TAS active:     {format_tas_value(self._current_tas())}",
                    f"TAS count:      {len(self.tas_values)}",
                    f"Total generated: {len(all_coords)}",
                    f"Active points:   {len(active_coords)}",
                    f"Removed points:  {len(removed_coords)}",
                    f"Edit mode:       {self._mode_name()}",
                    f"Aero overlay:    {self._current_overlay_name()}",
                    f"Vehicle STL:     {stl_status}",
                    f"Selected points: {len(selected_coords)}",
                    f"Undo stack:      {len(self.history)} actions",
                    f"Output CSV: {wrapped_path}",
                ]
            )
            self.status_text.set_text(status)
        self.fig.canvas.draw_idle()

    def _refresh_hover_only(self, include_sphere: bool = False) -> None:
        if self.preview_hover is not None:
            alpha, beta = self.preview_hover
            self.hover_ring.set_offsets(np.array([[float(beta), float(alpha)]], dtype=float))
        else:
            self.hover_ring.set_offsets(np.empty((0, 2), dtype=float))

        if include_sphere:
            all_coords, active_coords = self._effective_sets()
            if self.preview_selected_coords is not None:
                selected_coords = self.preview_selected_coords & active_coords & all_coords
            else:
                selected_coords = self._selected_coords_for_keys(self.selected_orbit_keys, all_coords, active_coords)
            self._refresh_sphere(all_coords, active_coords, selected_coords, self.preview_hover)
            self.last_sphere_refresh_ts = time.monotonic()
            self.last_sphere_hover_coord = self.preview_hover

        self.fig.canvas.draw_idle()

    def _find_nearest(self, event, pixel_tol: float = 9.0) -> Tuple[Coord | None, bool]:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return None, False

        coords_plot = self._nearest_plot_coords
        if coords_plot.size == 0:
            return None, False

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_span = max(abs(xlim[1] - xlim[0]), 1e-9)
        y_span = max(abs(ylim[1] - ylim[0]), 1e-9)
        bbox = self.ax.bbox
        x_scale = float(bbox.width) / x_span if bbox.width > 1e-9 else 1.0
        y_scale = float(bbox.height) / y_span if bbox.height > 1e-9 else 1.0

        dx_px = (coords_plot[:, 0] - float(event.xdata)) * x_scale
        dy_px = (coords_plot[:, 1] - float(event.ydata)) * y_scale
        dist2 = dx_px * dx_px + dy_px * dy_px
        i = int(np.argmin(dist2))
        if dist2[i] > float(pixel_tol * pixel_tol):
            return None, False
        beta = int(round(coords_plot[i, 0]))
        alpha = int(round(coords_plot[i, 1]))
        return (alpha, beta), bool(self._nearest_is_active[i])

    # ---------- editing actions ----------
    def _delete_coords(self, coords: Set[Coord]) -> None:
        to_remove = set(coords) & self.active_coords
        if not to_remove:
            return
        self._push_history()
        self.active_coords -= to_remove
        self._prune_selection()
        self._set_message(f"Removed {len(to_remove)} point(s).")
        self._clear_preview()
        self._refresh()

    def _restore_coords(self, coords: Set[Coord]) -> None:
        to_add = (set(coords) & self.all_coords) - self.active_coords
        if not to_add:
            return
        self._push_history()
        self.active_coords |= to_add
        self._prune_selection()
        self._set_message(f"Restored {len(to_add)} point(s).")
        self._clear_preview()
        self._refresh()

    def _delete_orbit(self, coord: Coord) -> None:
        self._delete_coords(self._group_in_all(coord))

    def _restore_orbit(self, coord: Coord) -> None:
        self._restore_coords(self._group_in_all(coord))

    def _delete_rectangle(self, x1: float, y1: float, x2: float, y2: float) -> None:
        x_lo, x_hi = sorted((x1, x2))
        y_lo, y_hi = sorted((y1, y2))

        selected_active: Set[Coord] = set()
        for alpha, beta in self.active_coords:
            x = float(beta)
            y = float(alpha)
            if x_lo <= x <= x_hi and y_lo <= y <= y_hi:
                selected_active.add((alpha, beta))

        if not selected_active:
            self._set_message("Rectangle selected no active points.")
            self.fig.canvas.draw_idle()
            return

        to_remove: Set[Coord] = set()
        for coord in selected_active:
            to_remove |= self._group_in_all(coord)
        self._delete_coords(to_remove)

    def _compute_moved_sets(self, source_coord: Coord, target_alpha: int, target_beta: int) -> Tuple[bool, Set[Coord], Set[Coord], str]:
        old_group = self._group_in_all(source_coord)
        if not old_group:
            return False, set(self.all_coords), set(self.active_coords), "Source point/group missing."

        if self.enforce_symmetry and self._is_complete_orbit(source_coord, self.all_coords):
            new_group = symmetry_orbit((target_alpha, target_beta))
        else:
            new_group = {(target_alpha, target_beta)}

        occupied_other = self.all_coords - old_group
        collision = new_group & occupied_other
        if collision:
            return False, set(self.all_coords), set(self.active_coords), "Move blocked: overlap with existing points."

        new_all = (self.all_coords - old_group) | new_group
        old_had_active = bool(old_group & self.active_coords)
        new_active = set(self.active_coords) - old_group
        if old_had_active:
            new_active |= new_group

        return True, new_all, new_active, ""

    def _preview_move(self, source_coord: Coord, target_alpha: int, target_beta: int, include_sphere: bool = False) -> None:
        is_orbit_move = self.enforce_symmetry and self._is_complete_orbit(source_coord, self.all_coords)
        ok, new_all, new_active, msg = self._compute_moved_sets(source_coord, target_alpha, target_beta)
        if not ok:
            self._set_message(msg)
            self._clear_preview()
            self._refresh(include_sphere=include_sphere, lightweight=True)
            return

        self.preview_all_coords = new_all
        self.preview_active_coords = new_active
        entity = "orbit" if is_orbit_move else "point"
        self._set_message(f"Preview move {entity} -> alpha={target_alpha}, beta={target_beta} (snap {self.snap_step} deg)")
        self._refresh(include_sphere=include_sphere, lightweight=True)

    def _commit_move(self, source_coord: Coord, target_alpha: int, target_beta: int) -> None:
        is_orbit_move = self.enforce_symmetry and self._is_complete_orbit(source_coord, self.all_coords)
        ok, new_all, new_active, msg = self._compute_moved_sets(source_coord, target_alpha, target_beta)
        if not ok:
            self._set_message(msg)
            self._clear_preview()
            self._refresh()
            return

        self._push_history()
        self.all_coords = new_all
        self.active_coords = new_active
        if is_orbit_move:
            moved_from = self._orbit_key(source_coord)
            moved_to = self._orbit_key((target_alpha, target_beta))
            if moved_from in self.selected_orbit_keys:
                self.selected_orbit_keys.remove(moved_from)
                self.selected_orbit_keys.add(moved_to)
        else:
            self._clear_selection()
        self._prune_selection()
        self._clear_preview()
        entity = "orbit" if is_orbit_move else "point"
        self._set_message(f"Moved {entity} -> alpha={target_alpha}, beta={target_beta}")
        self._refresh()

    def _compute_selected_move_sets(self, axis: str, delta_abs: int) -> Tuple[bool, Set[Coord], Set[Coord], Set[Tuple[int, int]], str]:
        if not self.enforce_symmetry:
            return False, set(self.all_coords), set(self.active_coords), set(self.selected_orbit_keys), "Group move works only with Mirror: ON."
        if axis not in {"horizontal", "vertical"}:
            return False, set(self.all_coords), set(self.active_coords), set(self.selected_orbit_keys), "Invalid move axis."
        if not self.selected_orbit_keys:
            return False, set(self.all_coords), set(self.active_coords), set(), "No selected points. Use Shift+click to select."

        key_to_old_group: dict[Tuple[int, int], Set[Coord]] = {}
        old_selected_coords: Set[Coord] = set()
        for key in self.selected_orbit_keys:
            old_group = symmetry_orbit(key) & self.all_coords
            if not old_group:
                continue
            key_to_old_group[key] = set(old_group)
            old_selected_coords |= old_group
        if not old_selected_coords:
            return False, set(self.all_coords), set(self.active_coords), set(self.selected_orbit_keys), "Selected points are no longer active."

        new_keys: Set[Tuple[int, int]] = set()
        new_selected_coords: Set[Coord] = set()
        for (a_abs, b_abs), old_group in key_to_old_group.items():
            if axis == "horizontal":
                new_a, new_b = a_abs, max(0, b_abs + delta_abs)
            else:
                new_a, new_b = max(0, a_abs + delta_abs), b_abs
            new_keys.add((new_a, new_b))
            source_complete = symmetry_orbit((a_abs, b_abs)) <= self.all_coords
            if source_complete:
                new_selected_coords |= symmetry_orbit((new_a, new_b))
            else:
                for old_a, old_b in old_group:
                    sign_a = 0 if old_a == 0 else (1 if old_a > 0 else -1)
                    sign_b = 0 if old_b == 0 else (1 if old_b > 0 else -1)
                    if axis == "horizontal":
                        moved_a_abs = abs(old_a)
                        moved_b_abs = max(0, abs(old_b) + delta_abs)
                    else:
                        moved_a_abs = max(0, abs(old_a) + delta_abs)
                        moved_b_abs = abs(old_b)
                    new_selected_coords.add((sign_a * moved_a_abs, sign_b * moved_b_abs))

        occupied_other = self.all_coords - old_selected_coords
        collision = new_selected_coords & occupied_other
        if collision:
            return (
                False,
                set(self.all_coords),
                set(self.active_coords),
                set(self.selected_orbit_keys),
                "Move blocked: selected group would overlap existing points.",
            )

        new_all = occupied_other | new_selected_coords
        new_active = (self.active_coords - old_selected_coords) | new_selected_coords
        return True, new_all, new_active, new_keys, ""

    def _preview_selected_move(self, axis: str, delta_abs: int, include_sphere: bool = False) -> None:
        ok, new_all, new_active, new_keys, msg = self._compute_selected_move_sets(axis, delta_abs)
        if not ok:
            self._set_message(msg)
            self._clear_preview()
            self._refresh(include_sphere=include_sphere, lightweight=True)
            return

        self.preview_all_coords = new_all
        self.preview_active_coords = new_active
        self.preview_selected_coords = self._selected_coords_for_keys(new_keys, new_all, new_active)
        self._set_message(f"Preview {axis} move for {len(new_keys)} selection key(s): Δ={delta_abs} deg")
        self._refresh(include_sphere=include_sphere, lightweight=True)

    def _commit_selected_move(self, axis: str, delta_abs: int) -> None:
        ok, new_all, new_active, new_keys, msg = self._compute_selected_move_sets(axis, delta_abs)
        if not ok:
            self._set_message(msg)
            self._clear_preview()
            self._refresh()
            return

        self._push_history()
        self.all_coords = new_all
        self.active_coords = new_active
        self.selected_orbit_keys = set(new_keys)
        self._prune_selection()
        self._clear_preview()
        self._set_message(f"Moved selected group ({axis}) by {delta_abs} deg.")
        self._refresh()

    def _add_orbit_from_raw_values(self, alpha_raw: float, beta_raw: float, source: str = "input") -> None:
        alpha = snap_to_step(alpha_raw, self.snap_step)
        beta = snap_to_step(beta_raw, self.snap_step)
        if self.enforce_symmetry:
            key = self._orbit_key((alpha, beta))
            target_coords = symmetry_orbit((key[0], key[1]))
            entity = f"orbit |alpha|={key[0]}, |beta|={key[1]}"
        else:
            key = None
            target_coords = {(alpha, beta)}
            entity = f"point alpha={alpha}, beta={beta}"

        if target_coords <= self.all_coords:
            if target_coords <= self.active_coords:
                self._set_message(f"{entity} already exists.")
            else:
                self._push_history()
                self.active_coords |= target_coords
                if self.enforce_symmetry and key is not None:
                    self.selected_orbit_keys.add(key)
                else:
                    self._clear_selection()
                self._set_message(f"Restored {entity} from {source}.")
                self._clear_preview()
                self._refresh()
            return

        self._push_history()
        self.all_coords |= target_coords
        self.active_coords |= target_coords
        if self.enforce_symmetry and key is not None:
            self.selected_orbit_keys.add(key)
        else:
            self._clear_selection()
        self._set_message(f"Added {entity} from {source}.")
        self._clear_preview()
        self._refresh()

    def _add_point_from_widgets(self, _event=None) -> None:
        if self.tb_add_point is None:
            return
        try:
            raw = self.tb_add_point.text.strip().replace(" ", "")
            if raw.startswith("(") and raw.endswith(")"):
                raw = raw[1:-1]
            parts = raw.split(",")
            if len(parts) != 2:
                raise ValueError("Need two comma-separated values")
            beta_raw = float(parts[0])   # x-axis first
            alpha_raw = float(parts[1])  # y-axis second
        except ValueError:
            self._set_message("Add point error: use format (beta,alpha), e.g. (15,-10).")
            self.fig.canvas.draw_idle()
            return

        self._add_orbit_from_raw_values(alpha_raw=alpha_raw, beta_raw=beta_raw, source="input")

    def _finalize_design_space(self, _event=None) -> None:
        removed = self.all_coords - self.active_coords
        if not removed:
            self._set_message("No removed points to finalize.")
            self.fig.canvas.draw_idle()
            return

        self._push_history()
        self.all_coords = set(self.active_coords)
        self.active_coords = set(self.active_coords)
        self._prune_selection()
        self._clear_preview()
        self._set_message("Finalize view applied: removed points permanently dropped.")
        self._refresh()

    # ---------- mouse + keyboard ----------
    def _update_param_hover_help(self, event) -> None:
        if self.param_help_text is None:
            return

        default_msg = "Hover any input field to see a brief description."
        help_msg = default_msg
        if event is not None and event.inaxes in self.param_axis_help:
            help_msg = self.param_axis_help[event.inaxes]

        if self.param_help_text.get_text() != help_msg:
            self.param_help_text.set_text(help_msg)
            self.fig.canvas.draw_idle()

    def _on_hover(self, event) -> None:
        if event.inaxes != self.ax:
            if self.preview_hover is None and not self.hover_annot.get_visible():
                return
            self.preview_hover = None
            self.hover_annot.set_visible(False)
            include_sphere = (time.monotonic() - self.last_sphere_refresh_ts) >= self.sphere_refresh_interval_s
            self._refresh_hover_only(include_sphere=include_sphere)
            return

        coord, _is_active = self._find_nearest(event, pixel_tol=8.0)
        if coord is None:
            if self.preview_hover is None and not self.hover_annot.get_visible():
                return
            self.preview_hover = None
            self.hover_annot.set_visible(False)
            include_sphere = (time.monotonic() - self.last_sphere_refresh_ts) >= self.sphere_refresh_interval_s
            self._refresh_hover_only(include_sphere=include_sphere)
            return

        alpha, beta = coord
        hover_text = f"beta={beta}, alpha={alpha}"

        # Skip no-op redraws when hover target and annotation state are unchanged.
        if (
            coord == self.preview_hover
            and self.hover_annot.get_visible()
            and self.hover_annot.get_text() == hover_text
            and (
                coord == self.last_sphere_hover_coord
                or (time.monotonic() - self.last_sphere_refresh_ts) < self.sphere_refresh_interval_s
            )
        ):
            return

        self.preview_hover = coord
        self.hover_annot.xy = (float(beta), float(alpha))
        self.hover_annot.set_text(hover_text)
        self.hover_annot.set_visible(True)
        include_sphere = (time.monotonic() - self.last_sphere_refresh_ts) >= self.sphere_refresh_interval_s
        self._refresh_hover_only(include_sphere=include_sphere)

    def _on_press(self, event) -> None:
        if event.inaxes != self.ax:
            return

        if event.button == 1:
            coord, is_active = self._find_nearest(event)
            self.press_pixel = (float(event.x), float(event.y)) if event.x is not None and event.y is not None else None
            self.press_data = (float(event.xdata), float(event.ydata)) if event.xdata is not None and event.ydata is not None else None
            self.drag_started = False
            self.drag_axis = None
            self.last_preview_drag_target = None
            self.last_preview_group_target = None

            # Shift+left-click toggles symmetric orbit selection (Mirror ON only).
            if coord is not None and is_active and self._event_has_shift(event):
                if self.enforce_symmetry:
                    self._toggle_orbit_selection(coord)
                else:
                    self._set_message("Shift multi-select works only with Mirror: ON.")
                    self._clear_preview()
                    self._refresh()
                self.mode = None
                self.pressed_coord = None
                self.press_pixel = None
                self.press_data = None
                return

            if coord is not None and is_active:
                if self.enforce_symmetry and self.selected_orbit_keys and (self._orbit_key(coord) in self.selected_orbit_keys):
                    self.mode = "drag_selected_group"
                    self._set_message(
                        "Moving selected group: drag mostly horizontal or vertical to axis-lock."
                    )
                else:
                    self.mode = "drag_active"
                    if self.enforce_symmetry and self._is_complete_orbit(coord, self.all_coords):
                        self._set_message("Drag active point to move symmetric orbit.")
                    else:
                        self._set_message("Drag active point to move this single point.")
                self.pressed_coord = coord
            elif coord is not None and not is_active:
                self.mode = "restore_click"
                self.pressed_coord = coord
                if self.enforce_symmetry:
                    self._set_message("Release click to restore symmetric orbit.")
                else:
                    self._set_message("Release click to restore this point.")
            else:
                self.mode = "rect_delete"
                self.pressed_coord = None
                if event.xdata is not None and event.ydata is not None:
                    self.rect_start_data = (float(event.xdata), float(event.ydata))
                    from matplotlib.patches import Rectangle

                    self.rect_patch = Rectangle(
                        (event.xdata, event.ydata),
                        0.0,
                        0.0,
                        fill=False,
                        edgecolor="#d32f2f",
                        linewidth=1.4,
                        linestyle="--",
                    )
                    self.ax.add_patch(self.rect_patch)
                    self.fig.canvas.draw_idle()

        elif event.button == 3:
            coord, is_active = self._find_nearest(event)
            if coord is not None and is_active:
                self._delete_orbit(coord)
        elif event.button == 2:
            # Middle-click adds a point at cursor: x=beta, y=alpha.
            if event.xdata is None or event.ydata is None:
                return
            beta_raw = float(event.xdata)
            alpha_raw = float(event.ydata)
            self._add_orbit_from_raw_values(alpha_raw=alpha_raw, beta_raw=beta_raw, source="middle-click")

    def _on_motion(self, event) -> None:
        self._update_param_hover_help(event)
        if self.mode in {"drag_active", "drag_selected_group", "rect_delete"}:
            # Avoid expensive hover-triggered sphere redraws during active drag operations.
            pass
        else:
            self._on_hover(event)

        if self.mode == "rect_delete":
            if self.rect_patch is None or self.rect_start_data is None or event.xdata is None or event.ydata is None:
                return

            x0, y0 = self.rect_start_data
            x1, y1 = float(event.xdata), float(event.ydata)
            self.rect_patch.set_x(min(x0, x1))
            self.rect_patch.set_y(min(y0, y1))
            self.rect_patch.set_width(abs(x1 - x0))
            self.rect_patch.set_height(abs(y1 - y0))
            self.fig.canvas.draw_idle()
            return

        if self.mode == "drag_active" and self.pressed_coord is not None:
            if self.press_pixel is None or event.x is None or event.y is None:
                return

            dx = float(event.x) - self.press_pixel[0]
            dy = float(event.y) - self.press_pixel[1]
            if math.hypot(dx, dy) > 3.0:
                self.drag_started = True

            if self.drag_started and event.xdata is not None and event.ydata is not None:
                target_beta = snap_to_step(float(event.xdata), self.snap_step)
                target_alpha = snap_to_step(float(event.ydata), self.snap_step)
                target = (target_alpha, target_beta)
                if target == self.last_preview_drag_target:
                    return
                self.last_preview_drag_target = target
                is_orbit_preview = self.enforce_symmetry and self._is_complete_orbit(self.pressed_coord, self.all_coords)
                self._set_message(
                    f"Preview move {'orbit' if is_orbit_preview else 'point'} -> alpha={target_alpha}, beta={target_beta} (snap {self.snap_step} deg)"
                )
                self._update_drag_preview_marker(target_alpha, target_beta, source_coord=self.pressed_coord)
            return

        if self.mode == "drag_selected_group" and self.pressed_coord is not None:
            if self.press_pixel is None or event.x is None or event.y is None:
                return

            dx_px = float(event.x) - self.press_pixel[0]
            dy_px = float(event.y) - self.press_pixel[1]
            if math.hypot(dx_px, dy_px) > 3.0:
                self.drag_started = True
                if self.drag_axis is None:
                    self.drag_axis = "horizontal" if abs(dx_px) >= abs(dy_px) else "vertical"

            if not self.drag_started or self.drag_axis is None:
                return
            if event.xdata is None or event.ydata is None:
                return

            if self.drag_axis == "horizontal":
                target_abs = abs(snap_to_step(float(event.xdata), self.snap_step))
                delta_abs = target_abs - abs(self.pressed_coord[1])
            else:
                target_abs = abs(snap_to_step(float(event.ydata), self.snap_step))
                delta_abs = target_abs - abs(self.pressed_coord[0])
            preview_key = (self.drag_axis, delta_abs)
            if preview_key == self.last_preview_group_target:
                return
            self.last_preview_group_target = preview_key
            self._preview_selected_move(self.drag_axis, delta_abs, include_sphere=False)

    def _cleanup_rectangle(self) -> None:
        if self.rect_patch is not None:
            self.rect_patch.remove()
            self.rect_patch = None
        self.rect_start_data = None

    def _on_release(self, event) -> None:
        if event.button != 1:
            return

        moved_pixels = 0.0
        if self.press_pixel is not None and event.x is not None and event.y is not None:
            moved_pixels = math.hypot(float(event.x) - self.press_pixel[0], float(event.y) - self.press_pixel[1])

        if self.mode == "restore_click" and self.pressed_coord is not None:
            if moved_pixels <= 3.0:
                self._restore_orbit(self.pressed_coord)

        elif self.mode == "drag_active" and self.pressed_coord is not None:
            if self.drag_started and event.xdata is not None and event.ydata is not None:
                target_beta = snap_to_step(float(event.xdata), self.snap_step)
                target_alpha = snap_to_step(float(event.ydata), self.snap_step)
                self._commit_move(self.pressed_coord, target_alpha, target_beta)
            else:
                self._clear_preview()
                self._refresh()

        elif self.mode == "drag_selected_group" and self.pressed_coord is not None:
            if self.drag_started and self.drag_axis is not None and event.xdata is not None and event.ydata is not None:
                if self.drag_axis == "horizontal":
                    target_abs = abs(snap_to_step(float(event.xdata), self.snap_step))
                    delta_abs = target_abs - abs(self.pressed_coord[1])
                else:
                    target_abs = abs(snap_to_step(float(event.ydata), self.snap_step))
                    delta_abs = target_abs - abs(self.pressed_coord[0])
                self._commit_selected_move(self.drag_axis, delta_abs)
            else:
                self._clear_preview()
                self._refresh()

        elif self.mode == "rect_delete" and self.rect_start_data is not None:
            if event.xdata is not None and event.ydata is not None:
                x0, y0 = self.rect_start_data
                x1, y1 = float(event.xdata), float(event.ydata)
                if abs(x1 - x0) >= 0.25 and abs(y1 - y0) >= 0.25:
                    self._delete_rectangle(x0, y0, x1, y1)
                else:
                    self._set_message("Rectangle too small; no deletion applied.")
                    self.fig.canvas.draw_idle()

        self._cleanup_rectangle()
        self.mode = None
        self.pressed_coord = None
        self.press_pixel = None
        self.press_data = None
        self.drag_started = False
        self.drag_axis = None

    def _on_key(self, event) -> None:
        if self._is_textbox_editing(event):
            return
        if event.key == "u":
            self._undo_last()
        elif event.key == "a":
            self._restore_all()
        elif event.key == "s":
            self._save_only()
        elif event.key == "m":
            self._toggle_symmetry_mode()
        elif event.key == "c":
            self._clear_selection()
            self._set_message("Selection cleared.")
            self._clear_preview()
            self._refresh()
        elif event.key == "t":
            self._cycle_tas()
        elif event.key == "v":
            self._toggle_vehicle_cad()

    def _iter_textboxes(self):
        for tb in [
            self.tb_alpha_max,
            self.tb_beta_max,
            self.tb_n_alpha,
            self.tb_n_beta,
            self.tb_alpha_exp,
            self.tb_beta_exp,
            self.tb_output_dir,
            self.tb_output_file,
            self.tb_add_point,
            self.tb_tas_values,
        ]:
            if tb is not None:
                yield tb

    def _is_textbox_editing(self, event=None) -> bool:
        event_axes = getattr(event, "inaxes", None)
        for tb in self._iter_textboxes():
            if event_axes is not None and event_axes is getattr(tb, "ax", None):
                return True
            if bool(getattr(tb, "capturekeystrokes", False)):
                return True
        return False

    # ---------- top-level actions ----------
    def _restore_all(self, _event=None) -> None:
        if self.active_coords == self.all_coords:
            return
        self._push_history()
        self.active_coords = set(self.all_coords)
        self._prune_selection()
        self._clear_preview()
        self._set_message("All points restored.")
        self._refresh()

    def _current_output_path(self) -> Path:
        if self.tb_output_dir is None or self.tb_output_file is None:
            return self.output_csv
        output_dir = self.tb_output_dir.text.strip()
        output_file = self.tb_output_file.text.strip()
        if not output_dir:
            output_dir = str(self.output_csv.parent)
        if not output_file:
            output_file = self.output_csv.name

        # Keep output_file as a filename only; prevent accidental path separators.
        output_file = Path(output_file).name.strip()
        if not output_file or output_file in {".", ".."}:
            output_file = self.output_csv.name
        if "\x00" in output_file:
            output_file = output_file.replace("\x00", "")
        if not output_file.lower().endswith(".csv"):
            output_file = f"{output_file}.csv"
        if len(output_file) > 200:
            output_file = f"{output_file[:196]}.csv"

        return (Path(output_dir).expanduser() / output_file).expanduser()

    def _gather_export_rows(self) -> List[Tuple[int, int, float]]:
        self._capture_current_tas_state()
        rows: List[Tuple[int, int, float]] = []
        for tas in self.tas_values:
            key = float(round(tas, 6))
            state = self.tas_design_spaces.get(key)
            if state is None:
                continue
            _all_coords, active_coords, _selected = state
            for alpha, beta in active_coords:
                rows.append((int(alpha), int(beta), key))
        return rows

    def _load_doe_from_output_path(self, _event=None) -> None:
        path = self._current_output_path()
        if not path.exists():
            self._set_message(f"Load failed: file not found -> {path}")
            self.fig.canvas.draw_idle()
            return

        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    raise ValueError("CSV has no header row.")

                fields = {}
                for name in reader.fieldnames:
                    if name is None:
                        continue
                    key = name.strip().lower().lstrip("\ufeff")
                    fields[key] = name

                def find_col(target: str) -> str | None:
                    exact = fields.get(target)
                    if exact is not None:
                        return exact
                    for k, original in fields.items():
                        if target in k:
                            return original
                    return None

                alpha_col = find_col("alpha")
                beta_col = find_col("beta")
                status_col = find_col("status")
                tas_col = find_col("tas")
                if alpha_col is None or beta_col is None:
                    raise ValueError("CSV must contain 'alpha' and 'beta' columns.")

                loaded_all_by_tas: dict[float, Set[Coord]] = {}
                loaded_active_by_tas: dict[float, Set[Coord]] = {}
                tas_order: List[float] = []
                tas_targets = [float(round(v, 6)) for v in self.tas_values] if tas_col is None else []
                if tas_col is None and not tas_targets:
                    tas_targets = [0.0]
                if tas_col is None:
                    for tas_key in tas_targets:
                        loaded_all_by_tas[tas_key] = set()
                        loaded_active_by_tas[tas_key] = set()
                        tas_order.append(tas_key)

                for row in reader:
                    alpha_raw = str(row.get(alpha_col, "")).strip()
                    beta_raw = str(row.get(beta_col, "")).strip()
                    if not alpha_raw or not beta_raw:
                        continue

                    if tas_col is None:
                        tas_keys = list(tas_targets)
                    else:
                        tas_raw = str(row.get(tas_col, "")).strip()
                        tas_value = float(tas_raw) if tas_raw else 0.0
                        tas_key = float(round(tas_value, 6))
                        tas_keys = [tas_key]

                    alpha = int(round(float(alpha_raw)))
                    beta = int(round(float(beta_raw)))
                    coord = (alpha, beta)

                    if status_col is None:
                        is_active = True
                    else:
                        raw = str(row.get(status_col, "")).strip().lower()
                        is_active = raw in {"1", "true", "yes", "y", "active", "keep"}

                    for tas_key in tas_keys:
                        if tas_key not in loaded_all_by_tas:
                            loaded_all_by_tas[tas_key] = set()
                            loaded_active_by_tas[tas_key] = set()
                            tas_order.append(tas_key)
                        loaded_all_by_tas[tas_key].add(coord)
                        if is_active:
                            loaded_active_by_tas[tas_key].add(coord)

            if not loaded_all_by_tas:
                raise ValueError("CSV has no valid point rows.")

            tas_spaces: dict[float, tuple[Set[Coord], Set[Coord], Set[Tuple[int, int]]]] = {}
            for tas_key in tas_order:
                loaded_all = loaded_all_by_tas[tas_key]
                loaded_active = loaded_active_by_tas[tas_key]
                loaded_active &= loaded_all
                if not loaded_active:
                    loaded_active = set(loaded_all)
                tas_spaces[tas_key] = (set(loaded_all), set(loaded_active), set())

            current_tas = float(round(self._current_tas(), 6))
            self._push_history()
            self.tas_values = list(tas_order)
            self.tas_design_spaces = tas_spaces
            if current_tas in self.tas_values:
                self.tas_idx = self.tas_values.index(current_tas)
            else:
                self.tas_idx = 0
            self._load_tas_state(self._current_tas())
            self._clear_preview()
            self.output_csv = path
            save_last_output_path(self.output_csv)
            self._sync_textboxes_from_cfg()
            self._update_tas_button_label()
            if tas_col is None:
                self._set_message(
                    f"Loaded DOE without TAS column; applied points to all {len(self.tas_values)} TAS values from {path.name}. Active TAS={format_tas_value(self._current_tas())}"
                )
            else:
                self._set_message(
                    f"Loaded DOE: {len(self.tas_values)} TAS values from {path.name}; active TAS={format_tas_value(self._current_tas())}"
                )
            self._refresh()
        except Exception as err:
            self._set_message(f"Load failed: {err}")
            self.fig.canvas.draw_idle()

    def _save_only(self, _event=None) -> None:
        try:
            self.output_csv = self._current_output_path()
            rows = self._gather_export_rows()
            save_points_csv(self.output_csv, rows)
            save_last_output_path(self.output_csv)
            self._sync_textboxes_from_cfg()
            self.saved = True
            self._set_message(
                f"Saved {len(rows)} rows across {len(self.tas_values)} TAS values -> {self.output_csv.name}"
            )
            self.fig.canvas.draw_idle()
        except Exception as err:
            self.saved = False
            self._set_message(f"Save failed: {err}")
            self.fig.canvas.draw_idle()

    # ---------- parameter widgets ----------
    def _sync_textboxes_from_cfg(self) -> None:
        if self.tb_alpha_max is None:
            return

        updates = [
            (self.tb_alpha_max, str(self.cfg.alpha_max_abs)),
            (self.tb_beta_max, str(self.cfg.beta_max_abs)),
            (self.tb_n_alpha, str(self.cfg.n_alpha_positive)),
            (self.tb_n_beta, str(self.cfg.n_beta_positive)),
            (self.tb_alpha_exp, f"{self.cfg.alpha_exponent:g}"),
            (self.tb_beta_exp, f"{self.cfg.beta_exponent:g}"),
        ]

        for tb, value in updates:
            eventson_prev = tb.eventson
            tb.eventson = False
            tb.set_val(value)
            tb.eventson = eventson_prev

        if self.tb_output_dir is not None and self.tb_output_file is not None:
            for tb, value in [
                (self.tb_output_dir, str(self.output_csv.parent)),
                (self.tb_output_file, str(self.output_csv.name)),
            ]:
                eventson_prev = tb.eventson
                tb.eventson = False
                tb.set_val(value)
                tb.eventson = eventson_prev

        if self.tb_tas_values is not None:
            eventson_prev = self.tb_tas_values.eventson
            self.tb_tas_values.eventson = False
            self.tb_tas_values.set_val(self._tas_values_to_text())
            self.tb_tas_values.eventson = eventson_prev

    def _apply_params(self, _event=None) -> None:
        def parse_int(tb, name: str, min_v: int) -> int:
            try:
                v = int(tb.text.strip())
            except ValueError as exc:
                raise ValueError(f"{name} must be integer") from exc
            if v < min_v:
                raise ValueError(f"{name} must be >= {min_v}")
            return v

        def parse_float(tb, name: str, min_v: float) -> float:
            try:
                v = float(tb.text.strip())
            except ValueError as exc:
                raise ValueError(f"{name} must be numeric") from exc
            if v < min_v:
                raise ValueError(f"{name} must be >= {min_v}")
            return v

        try:
            alpha_max = parse_int(self.tb_alpha_max, "alpha_max", 1)
            beta_max = parse_int(self.tb_beta_max, "beta_max", 1)
            n_alpha = parse_int(self.tb_n_alpha, "n_alpha", 1)
            n_beta = parse_int(self.tb_n_beta, "n_beta", 1)
            alpha_exp = parse_float(self.tb_alpha_exp, "alpha_exp", 1.0)
            beta_exp = parse_float(self.tb_beta_exp, "beta_exp", 1.0)

            n_alpha = min(n_alpha, alpha_max)
            n_beta = min(n_beta, beta_max)

            self._push_history()
            self.cfg = DesignConfig(
                alpha_max_abs=alpha_max,
                beta_max_abs=beta_max,
                n_alpha_positive=n_alpha,
                n_beta_positive=n_beta,
                alpha_exponent=alpha_exp,
                beta_exponent=beta_exp,
            )
            self._regenerate_from_cfg(clear_history=False)
            self._capture_current_tas_state()
            self._clear_preview()
            self._set_message("Parameters applied: grid regenerated and shown.")
            self._refresh()
        except ValueError as err:
            self._set_message(f"Parameter error: {err}")
            self.fig.canvas.draw_idle()

    def _get_screen_size_px(self) -> Tuple[int | None, int | None]:
        if self.fig is None or getattr(self.fig, "canvas", None) is None:
            return None, None
        manager = getattr(self.fig.canvas, "manager", None)
        window = getattr(manager, "window", None) if manager is not None else None
        if window is None:
            return None, None

        try:
            if hasattr(window, "winfo_screenwidth") and hasattr(window, "winfo_screenheight"):
                return int(window.winfo_screenwidth()), int(window.winfo_screenheight())
        except Exception:
            pass

        try:
            screen_fn = getattr(window, "screen", None)
            if callable(screen_fn):
                screen = screen_fn()
                size_fn = getattr(screen, "size", None)
                if callable(size_fn):
                    size = size_fn()
                    width = getattr(size, "width", None)
                    height = getattr(size, "height", None)
                    if width is not None and height is not None:
                        return int(width), int(height)
        except Exception:
            pass

        return None, None

    # ---------- gui build ----------
    def run(self) -> bool:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Button, TextBox
            import matplotlib.ticker as mticker
        except ImportError as exc:
            raise RuntimeError("matplotlib is required for GUI mode. Install with pip install matplotlib") from exc

        self.fig = plt.figure(figsize=(16, 9))
        dpi = float(self.fig.get_dpi())
        screen_w_px, screen_h_px = self._get_screen_size_px()
        if screen_w_px is not None and screen_h_px is not None:
            # Fit to the current display to avoid cramped widgets on smaller laptop screens.
            target_w_in = min(16.0, max(12.0, (screen_w_px * 0.92) / dpi))
            target_h_in = min(9.2, max(7.2, (screen_h_px * 0.86) / dpi))
            self.fig.set_size_inches(target_w_in, target_h_in, forward=True)

        fig_w_px = self.fig.get_figwidth() * dpi
        fig_h_px = self.fig.get_figheight() * dpi
        compact_ui = (fig_w_px < 1520.0) or (fig_h_px < 930.0)
        font_scale = 0.86 if compact_ui else 1.0

        if compact_ui:
            plot_rect = [0.05, 0.18, 0.59, 0.76]
            x_panel = 0.665
            w_panel = 0.305
            self.overlay_cbar_rect = (0.647, 0.615, 0.012, 0.26)
            sphere_rect = [x_panel, 0.695, w_panel, 0.295]
            summary_rect = [x_panel, 0.585, w_panel, 0.100]
            instructions_rect = [x_panel, 0.525, w_panel, 0.055]
            param_help_rect = [x_panel, 0.503, w_panel, 0.020]
            tb_y0 = 0.462
            tb_h = 0.024
            tb_gap = 0.0056
            right_btn_h = 0.018
            right_btn_y = [0.130, 0.109, 0.088, 0.067, 0.046, 0.025]
            primary_btn_y = 0.02
            primary_btn_h = 0.075
            primary_btn_gap = 0.010
        else:
            plot_rect = [0.05, 0.16, 0.64, 0.78]
            x_panel = 0.73
            w_panel = 0.25
            self.overlay_cbar_rect = (0.698, 0.615, 0.015, 0.275)
            sphere_rect = [x_panel, 0.695, w_panel, 0.295]
            summary_rect = [x_panel, 0.590, w_panel, 0.095]
            instructions_rect = [x_panel, 0.538, w_panel, 0.050]
            param_help_rect = [x_panel, 0.515, w_panel, 0.020]
            tb_y0 = 0.472
            tb_h = 0.0285
            tb_gap = 0.0062
            right_btn_h = 0.020
            right_btn_y = [0.138, 0.115, 0.092, 0.069, 0.046, 0.023]
            primary_btn_y = 0.02
            primary_btn_h = 0.085
            primary_btn_gap = 0.012

        # Main plot + right control column (non-overlapping layout)
        self.ax = self.fig.add_axes(plot_rect)
        # Keep overlay color legend in the gap between main plot and sphere.
        self.ax_overlay_cbar = self.fig.add_axes(list(self.overlay_cbar_rect))
        self.ax_overlay_cbar.set_visible(False)
        self.ax_sphere = self.fig.add_axes(sphere_rect, projection="3d")
        self.ax_summary = self.fig.add_axes(summary_rect)
        self.ax_summary.axis("off")
        self.ax_instructions = self.fig.add_axes(instructions_rect)
        self.ax_instructions.axis("off")
        ax_param_help = self.fig.add_axes(param_help_rect)
        ax_param_help.axis("off")

        self.active_scatter = self.ax.scatter([], [], s=42, c="#1565c0", label="active", zorder=3)
        self.removed_scatter = self.ax.scatter([], [], s=40, c="#b71c1c", marker="x", label="removed", zorder=2)
        self.selected_scatter = self.ax.scatter(
            [], [], s=110, facecolors="none", edgecolors="#39ff14", linewidths=2.2, label="selected", zorder=3.5
        )
        self.hover_ring = self.ax.scatter([], [], s=180, facecolors="none", edgecolors="#ffb300", linewidths=2.2, zorder=4)
        self.drag_preview_scatter = self.ax.scatter(
            [], [], s=95, facecolors="#ffca28", edgecolors="#f57f17", linewidths=1.0, alpha=0.95, zorder=4.2
        )

        self.ax.axhline(0, color="black", linewidth=1.0)
        self.ax.axvline(0, color="black", linewidth=1.0)
        self.ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=16))
        self.ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=16))
        self.ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        self.ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        self.ax.grid(True, which="major", alpha=0.48, linewidth=1.05)
        self.ax.grid(True, which="minor", alpha=0.28, linewidth=0.75, linestyle=":")
        self.ax.set_xlabel("beta (deg)")
        self.ax.set_ylabel("alpha (deg)")
        self.ax.set_title("'Alpha-Beta' Parameter Space")
        self.ax.set_aspect("auto")
        self.ax.tick_params(labelsize=9.0 * font_scale)
        self.ax.xaxis.label.set_size(10.0 * font_scale)
        self.ax.yaxis.label.set_size(10.0 * font_scale)
        self.ax.title.set_fontsize(11.0 * font_scale)

        self.legend = self.ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.13),
            ncol=3,
            frameon=True,
            borderaxespad=0.0,
        )
        if self.legend is not None:
            for txt in self.legend.get_texts():
                txt.set_fontsize(8.8 * font_scale)

        self.hover_annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.95},
            fontsize=9 * font_scale,
        )
        self.hover_annot.set_visible(False)

        self.status_text = self.ax_summary.text(
            0.0,
            1.0,
            "",
            va="top",
            ha="left",
            fontsize=8.4 * font_scale,
            family="monospace",
            clip_on=True,
        )

        self.ax_instructions.text(
            0.0,
            1.0,
            (
                "Edit params on right, then press Apply.\n"
                "Shift+Left-click active: toggle multi-select (Mirror ON) | Key c: clear selection\n"
                "Left-drag empty: delete box | Right-click: delete point/group | Middle-click: add point/group\n"
                "Set TAS list (comma-separated), Apply TAS List, then use TAS button or key t to switch\n"
                "Overlay button cycles: None -> Fx -> Fy -> Fz -> Mx -> My -> Mz | Key v: CAD show/hide\n"
                "Use output_dir/output_file + Load DOE to resume from a saved CSV"
            ),
            va="top",
            ha="left",
            fontsize=7.2 * font_scale,
            clip_on=True,
        )
        self.param_help_text = ax_param_help.text(
            0.0,
            1.0,
            "Hover any input field to see a brief description.",
            va="top",
            ha="left",
            fontsize=7.4 * font_scale,
            color="#4e342e",
            clip_on=True,
        )

        # Parameter text boxes (right panel)
        y0 = tb_y0
        h = tb_h
        gap = tb_gap

        ax_tb_alpha_max = self.fig.add_axes([x_panel, y0 - 0 * (h + gap), w_panel, h])
        ax_tb_beta_max = self.fig.add_axes([x_panel, y0 - 1 * (h + gap), w_panel, h])
        ax_tb_n_alpha = self.fig.add_axes([x_panel, y0 - 2 * (h + gap), w_panel, h])
        ax_tb_n_beta = self.fig.add_axes([x_panel, y0 - 3 * (h + gap), w_panel, h])
        ax_tb_alpha_exp = self.fig.add_axes([x_panel, y0 - 4 * (h + gap), w_panel, h])
        ax_tb_beta_exp = self.fig.add_axes([x_panel, y0 - 5 * (h + gap), w_panel, h])
        ax_tb_output_dir = self.fig.add_axes([x_panel, y0 - 6 * (h + gap), w_panel, h])
        ax_tb_output_file = self.fig.add_axes([x_panel, y0 - 7 * (h + gap), w_panel, h])

        # Add point row as a single coordinate input.
        add_row_y = y0 - 8 * (h + gap)
        ax_tb_add_point = self.fig.add_axes([x_panel, add_row_y, w_panel, h])
        ax_tb_tas_values = self.fig.add_axes([x_panel, y0 - 9 * (h + gap), w_panel, h])

        self.tb_alpha_max = TextBox(ax_tb_alpha_max, "alpha_max ", initial=str(self.cfg.alpha_max_abs))
        self.tb_beta_max = TextBox(ax_tb_beta_max, "beta_max ", initial=str(self.cfg.beta_max_abs))
        self.tb_n_alpha = TextBox(ax_tb_n_alpha, "n_alpha  ", initial=str(self.cfg.n_alpha_positive))
        self.tb_n_beta = TextBox(ax_tb_n_beta, "n_beta   ", initial=str(self.cfg.n_beta_positive))
        self.tb_alpha_exp = TextBox(ax_tb_alpha_exp, "alpha_exp", initial=f"{self.cfg.alpha_exponent:g}")
        self.tb_beta_exp = TextBox(ax_tb_beta_exp, "beta_exp ", initial=f"{self.cfg.beta_exponent:g}")
        self.tb_output_dir = TextBox(ax_tb_output_dir, "output_dir ", initial=str(self.output_csv.parent))
        self.tb_output_file = TextBox(ax_tb_output_file, "output_file", initial=str(self.output_csv.name))
        self.tb_add_point = TextBox(ax_tb_add_point, "point\n(beta, alpha) ", initial="(0,0)")
        self.tb_tas_values = TextBox(ax_tb_tas_values, "tas_values", initial=self._tas_values_to_text())

        self.param_axis_help = {
            ax_tb_alpha_max: "alpha_max: maximum absolute alpha extent (deg) for generated base grid.",
            ax_tb_beta_max: "beta_max: maximum absolute beta extent (deg) for generated base grid.",
            ax_tb_n_alpha: "n_alpha: number of positive alpha levels (excluding zero).",
            ax_tb_n_beta: "n_beta: number of positive beta levels (excluding zero).",
            ax_tb_alpha_exp: "alpha_exp: clustering exponent for alpha; >1 gives denser points near zero.",
            ax_tb_beta_exp: "beta_exp: clustering exponent for beta; >1 gives denser points near zero.",
            ax_tb_output_dir: "output_dir: folder used for save and load operations.",
            ax_tb_output_file: "output_file: CSV filename used for save and load operations.",
            ax_tb_add_point: "point(beta,alpha): x first then y; signs respected. Snaps to 5 deg. Mirror toggle controls symmetry.",
            ax_tb_tas_values: "tas_values: comma-separated TAS values (example: 20,30,40). Each TAS has its own editable design space.",
        }

        # Action buttons in right panel
        btn_h = right_btn_h
        ax_btn_load = self.fig.add_axes([x_panel, right_btn_y[0], w_panel, btn_h])
        ax_btn_apply = self.fig.add_axes([x_panel, right_btn_y[1], w_panel, btn_h])
        ax_btn_tas_apply = self.fig.add_axes([x_panel, right_btn_y[2], w_panel, btn_h])
        ax_btn_finalize = self.fig.add_axes([x_panel, right_btn_y[3], w_panel, btn_h])
        ax_btn_add = self.fig.add_axes([x_panel, right_btn_y[4], w_panel, btn_h])
        ax_btn_clear_sel = self.fig.add_axes([x_panel, right_btn_y[5], w_panel, btn_h])
        btn_load = Button(ax_btn_load, "Load DOE (from output path)")
        btn_apply = Button(ax_btn_apply, "Apply Params")
        btn_apply_tas = Button(ax_btn_tas_apply, "Apply TAS List")
        btn_finalize = Button(ax_btn_finalize, "Finalize View (Drop Removed)")
        btn_add = Button(ax_btn_add, "Add Point")
        btn_clear_sel = Button(ax_btn_clear_sel, "Clear Selection")
        btn_load.on_clicked(self._load_doe_from_output_path)
        btn_apply.on_clicked(self._apply_params)
        btn_apply_tas.on_clicked(self._apply_tas_values)
        btn_finalize.on_clicked(self._finalize_design_space)
        btn_add.on_clicked(self._add_point_from_widgets)
        btn_clear_sel.on_clicked(
            lambda _evt: (self._clear_selection(), self._clear_preview(), self._set_message("Selection cleared."), self._refresh())
        )

        # Bottom-row primary controls under plot only (no right-panel intersection)
        btn_y = primary_btn_y
        btn_h = primary_btn_h
        start_x = plot_rect[0]
        total_w = plot_rect[2]
        gap_btn = primary_btn_gap
        btn_w = (total_w - 5 * gap_btn) / 6.0

        ax_btn_mode = self.fig.add_axes([start_x + 0 * (btn_w + gap_btn), btn_y, btn_w, btn_h])
        ax_btn_restore = self.fig.add_axes([start_x + 1 * (btn_w + gap_btn), btn_y, btn_w, btn_h])
        ax_btn_overlay = self.fig.add_axes([start_x + 2 * (btn_w + gap_btn), btn_y, btn_w, btn_h])
        ax_btn_vehicle = self.fig.add_axes([start_x + 3 * (btn_w + gap_btn), btn_y, btn_w, btn_h])
        ax_btn_tas = self.fig.add_axes([start_x + 4 * (btn_w + gap_btn), btn_y, btn_w, btn_h])
        ax_btn_save = self.fig.add_axes([start_x + 5 * (btn_w + gap_btn), btn_y, btn_w, btn_h])

        self.btn_symmetry_toggle = Button(ax_btn_mode, "")
        self._update_mode_button_label()
        self.btn_overlay_cycle = Button(ax_btn_overlay, "")
        self._update_overlay_button_label()
        self.btn_vehicle_toggle = Button(ax_btn_vehicle, "")
        self._update_vehicle_button_label()
        self.btn_tas_cycle = Button(ax_btn_tas, "")
        self._update_tas_button_label()
        btn_restore = Button(ax_btn_restore, "Restore All")
        btn_save = Button(ax_btn_save, "Save")

        # Scale control text for compact displays to avoid overlap.
        textboxes = [
            self.tb_alpha_max,
            self.tb_beta_max,
            self.tb_n_alpha,
            self.tb_n_beta,
            self.tb_alpha_exp,
            self.tb_beta_exp,
            self.tb_output_dir,
            self.tb_output_file,
            self.tb_add_point,
            self.tb_tas_values,
        ]
        for tb in textboxes:
            if tb is None:
                continue
            tb.label.set_fontsize(8.0 * font_scale)
            if hasattr(tb, "text_disp"):
                tb.text_disp.set_fontsize(8.4 * font_scale)

        panel_buttons = [btn_load, btn_apply, btn_apply_tas, btn_finalize, btn_add, btn_clear_sel]
        for btn in panel_buttons:
            btn.label.set_fontsize(8.0 * font_scale)
        main_buttons = [self.btn_symmetry_toggle, self.btn_overlay_cycle, self.btn_vehicle_toggle, self.btn_tas_cycle, btn_restore, btn_save]
        for btn in main_buttons:
            btn.label.set_fontsize(8.8 * font_scale)

        self.btn_symmetry_toggle.on_clicked(self._toggle_symmetry_mode)
        self.btn_overlay_cycle.on_clicked(self._cycle_overlay)
        self.btn_vehicle_toggle.on_clicked(self._toggle_vehicle_cad)
        self.btn_tas_cycle.on_clicked(self._cycle_tas)
        btn_restore.on_clicked(self._restore_all)
        btn_save.on_clicked(self._save_only)

        for tb in [self.tb_alpha_max, self.tb_beta_max, self.tb_n_alpha, self.tb_n_beta, self.tb_alpha_exp, self.tb_beta_exp]:
            tb.on_submit(lambda _text: self._apply_params())
        self.tb_output_dir.on_submit(lambda _text: self._refresh())
        self.tb_output_file.on_submit(lambda _text: self._refresh())
        self.tb_add_point.on_submit(lambda _text: self._add_point_from_widgets())
        self.tb_tas_values.on_submit(lambda _text: self._apply_tas_values())

        self.message_text = self.fig.text(
            x_panel,
            0.033,
            "Ready.",
            ha="left",
            va="center",
            fontsize=9.0 * font_scale,
            color="#2e7d32",
        )

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._refresh()
        plt.show()
        return self.saved


def main() -> None:
    args = parse_args()

    try:
        tas_values = parse_tas_values_text(args.tas_values)
    except ValueError as err:
        print(f"Error: invalid --tas-values ({err})")
        return

    cfg = DesignConfig(
        alpha_max_abs=max(1, args.alpha_max),
        beta_max_abs=max(1, args.beta_max),
        n_alpha_positive=max(1, min(args.n_alpha, max(1, args.alpha_max))),
        n_beta_positive=max(1, min(args.n_beta, max(1, args.beta_max))),
        alpha_exponent=max(1.0, args.alpha_exp),
        beta_exponent=max(1.0, args.beta_exp),
    )

    if args.output:
        output_csv = Path(args.output).expanduser()
    else:
        output_csv = load_last_output_path(Path.cwd() / default_output_name())

    if args.no_gui:
        points, _, _ = build_symmetric_points(cfg)
        export_rows: list[tuple[int, int, float]] = []
        for tas in tas_values:
            export_rows.extend((alpha, beta, tas) for alpha, beta in points)
        save_points_csv(output_csv, export_rows)
        print("Saved without GUI")
        print(f"alpha levels: {[0] + generate_clustered_integer_levels(cfg.alpha_max_abs, cfg.n_alpha_positive, cfg.alpha_exponent)}")
        print(f"beta levels:  {[0] + generate_clustered_integer_levels(cfg.beta_max_abs, cfg.n_beta_positive, cfg.beta_exponent)}")
        print(f"tas values:   {[format_tas_value(v) for v in tas_values]}")
        print(f"total rows:   {len(export_rows)}")
        print(f"output: {output_csv}")
        return

    editor = InteractivePointEditor(cfg, output_csv, tas_values=tas_values)
    try:
        saved = editor.run()
    except RuntimeError as err:
        print(f"Error: {err}")
        return

    if saved:
        print(f"Saved {len(editor._gather_export_rows())} rows to: {editor.output_csv}")
    else:
        print("Exited without saving")


if __name__ == "__main__":
    main()
