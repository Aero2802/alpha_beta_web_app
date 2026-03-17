# Wind Tunnel Test Matrix Tool

This folder contains a workflow for creating a wind-tunnel test design space.

The workflow has two stages:

1. `WTT_alpha_beta_matrix_generator.py` - Build or edit the base `ALPHA` / `BETA` / `TAS` design space.
2. `build_WTT3_DOE_matrix.py` - Expand that design space into full DOE tables for each wind-tunnel test type.

## Files in this folder

- `WTT_alpha_beta_matrix_generator.py`
  Interactive editor for creating a alpha-beta-TAS design space.

- `build_WTT3_DOE_matrix.py`
  Expands the base design space into the final test matrices with parameters like thruster RPMs, hoop angle, etc.

- `alpha_beta_TAS_WTT3_matrix_latest.csv`
  Current base design-space CSV used by the DOE builder.

- `WTT3_full_aero_table_passive.csv`
- `WTT3_full_aero_table_front_lateral.csv`
- `WTT3_full_aero_table_hoop.csv`
- `WTT3_full_aero_table_interactions.csv`
  Aerotable input CSVs generated from the DOE builder.

- `CFD_data/`
  Optional CFD map data used only by the editor overlay view.

- `droid_ev3.stl` and `droid.stl`
  Optional CAD files used by the editor 3D/sphere view.

## What the workflow produces

The builder currently generates four test families:

- `passive`
- `front_lateral`
- `hoop`
- `interactions`

## Recommended workflow

### Step 1: Create the base design space

Use `WTT_alpha_beta_matrix_generator.py` to create or modify the alpha-beta-TAS points.

This script generates a symmetric grid, lets you delete or add points interactively, and saves the final active points to CSV.

### Step 2: Build the final WTT3 matrices

Use `build_WTT3_DOE_matrix.py` to expand the base design space into the full factorial test tables for each test family.

This script:

- reads the base CSV
- multiplies each base point by the per-test factor combinations
- inserts tare sweeps automatically
- writes the four output CSV files
- prints a duration summary to the terminal

## Dependencies

The editor requires Python plus these packages:

- `numpy`
- `matplotlib`

Example setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
```

## How the generated grid works

The editor builds a first-quadrant grid and mirrors it into all quadrants.

- `0` is always included for both alpha and beta.
- Positive alpha and beta levels are generated from the max range and number of levels.
- The levels are mirrored to create symmetric positive and negative points.
- Higher exponent values cluster more points near zero.

With the default settings:

- alpha levels become `[-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180]`
- beta levels become `[-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]`
- total points per TAS become `13 x 13 = 169`

If you use three TAS values, the default unsimplified export would be `169 x 3 = 507` rows before any manual deletions.

## GUI workflow

The editor window is intended to be used in this order:

1. Set the geometry parameters on the right panel.
2. Press `Apply` to regenerate the base grid.
3. Set the TAS list and press `Apply TAS List`.
4. Switch between TAS values with the `TAS` button or the `t` key.
5. Delete, restore, move, or add points until the design space is correct.
6. Press `Save` to write the CSV.
7. If required, save directly as `alpha_beta_TAS_WTT3_matrix_latest.csv`.

## Editor controls

### Mouse controls

- Left-drag on empty area: draw a delete rectangle.
- Right-click on an active point: delete that point or its symmetric group.
- Middle-click on the plot: add a point or symmetric group at the cursor.
- Left-drag on an active point: move the point.
- Shift + left-click on an active point: toggle multi-selection when mirror mode is on.

### Keyboard shortcuts

- `s`: save
- `u`: undo
- `a`: restore all removed points
- `m`: toggle mirror mode
- `c`: clear selection
- `t`: cycle TAS value
- `v`: show or hide the vehicle CAD overlay

## Mirror mode

Mirror mode is one of the most important editor settings.

When mirror mode is on:

- deleting a point affects its full symmetric orbit
- adding a point adds the full symmetric orbit
- dragging a point moves the symmetric orbit
- shift-selection works on symmetric groups

When mirror mode is off:

- edits apply only to the single point you act on

For most wind-tunnel matrix creation, keep mirror mode on unless you intentionally want asymmetry.

## TAS-specific editing

Each TAS value keeps its own editable design space.

That means:

- you can remove a point at `TAS=24` without removing it at `TAS=12`
- the editor stores separate active-point sets for each TAS
- the final saved CSV combines the active points from all TAS values

This is useful when high-angle points are safe or meaningful at one airspeed but not another.

## Loading an existing DOE/design CSV

The editor can resume from a saved CSV by using the `output_dir` and `output_file` fields, then pressing `Load DOE (from output path)`.

The loader accepts CSVs that contain at least:

- `alpha`
- `beta`

It can also read:

- `tas`
- `status`

Notes:

- column matching is case-insensitive in practice because the loader searches field names flexibly
- if no `TAS` column is present, the loaded points are applied to all current TAS values
- if no `status` column is present, all loaded points are treated as active

## Save behavior

When you press `Save`, the editor writes rows with header:

```text
SETPOINT,ALPHA,BETA,TAS
```

Important details:

- rows are re-numbered from `1`
- points are sorted by `TAS`, then `ALPHA`, then `BETA`
- only active points are exported
- the last used output path is remembered in `.doe_last_output.json`

## Optional overlays in the editor

The editor has two optional visual aids.

### CFD overlay

The overlay button cycles through:

- `None`
- `Fx`
- `Fy`
- `Fz`
- `Mx`
- `My`
- `Mz`

Behavior:

- the editor automatically looks inside `CFD_data/`
- it loads the most recently modified CFD CSV in that folder
- the CSV must contain `Alpha` and `Beta` columns
- the script then looks for the expected force/moment columns in that file

This overlay is for visual guidance only. It does not change the saved design-space points.

### Vehicle CAD overlay

The `v` key or the vehicle toggle button shows or hides the STL-based vehicle view.

The editor first tries to use `droid_ev3.stl` in this folder.

## Stage 2: Building the final test matrices

Once the base design space CSV is ready, run:

```bash
python3 build_WTT3_DOE_matrix.py
```

This script reads the file defined by the `DESIGN_SPACE_CSV` constant. In the current code, that file is:

```text
alpha_beta_TAS_WTT3_matrix_latest.csv
```

So the simplest workflow is:

1. save the editor output as `alpha_beta_TAS_WTT3_matrix_latest.csv`
2. run `python3 build_WTT3_DOE_matrix.py`

If you want to use a different base filename, edit the `DESIGN_SPACE_CSV` constant in `build_WTT3_DOE_matrix.py`.

## What the builder adds

The builder takes each `ALPHA` / `BETA` / `TAS` row and expands it using the factor lists defined in `TEST_FACTORS`.

The output columns are:

```text
SETPOINT,ALPHA,BETA,TAS,HOOP_ALPHA,HOOP_BETA,FLT_OMEGA,MT_OMEGA,DWELL_TIME,MAX_TRAVERSE_TIME
```

## Tare sweep behavior

The builder inserts tare rows automatically.

### Start-of-test tare

At the start of every output table, the script inserts a tare sweep over all unique `BETA` values found in the base design-space CSV.

Each tare row is created with:

- `ALPHA = 0`
- `TAS = 0`
- `HOOP_ALPHA = 0`
- `HOOP_BETA = 0`
- `FLT_OMEGA = 0`
- `MT_OMEGA = 0`
- `DWELL_TIME =` first configured dwell value, unless `TARE_DWELL_TIME` is set
- `MAX_TRAVERSE_TIME =` first configured traverse value, unless `TARE_MAX_TRAVERSE_TIME` is set

### Periodic tare

Periodic tare insertion is controlled by:

- `TIME_PER_STEP_SECONDS = 12.0`
- `TARE_INTERVAL_MINUTES = 60.0`

Current behavior:

- each row is assumed to take `12` seconds
- after approximately every `60` minutes of accumulated test time, a new tare sweep is inserted
- set `TARE_INTERVAL_MINUTES <= 0` to disable periodic tare and keep only the start tare

## Output files from the builder

The builder writes these files:

- `WTT3_full_aero_table_passive.csv`
- `WTT3_full_aero_table_front_lateral.csv`
- `WTT3_full_aero_table_hoop.csv`
- `WTT3_full_aero_table_interactions.csv`
