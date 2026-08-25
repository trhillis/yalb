import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "v"
STEP_PATTERN = re.compile(r"output_velocity_rank\d+_step(\d+)\.txt$")

# Ghia et al. (1982), Table I: horizontal velocity on the vertical
# cavity centerline for Re = 1000. Velocities are normalized by U_lid.
GHIA_Y = np.array([
    1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516,
    0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719,
    0.1016, 0.0703, 0.0625, 0.0547, 0.0000,
])

GHIA_U_RE1000 = np.array([
    1.00000, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304,
    0.18719, 0.05702, -0.06080, -0.10648, -0.27805, -0.38289,
    -0.29730, -0.22220, -0.20196, -0.18109, 0.00000,
])


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the LBM vertical-centerline velocity with "
            "Ghia et al. (1982) at Re=1000."
        )
    )
    parser.add_argument(
        "--step",
        type=int,
        help="output step to compare (default: most recently written step)",
    )
    parser.add_argument(
        "--u-lid",
        type=float,
        default=0.1,
        help="lid velocity used by the simulation (default: 0.1)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"velocity-output directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="plot filename (default: compare_re1000_step<step>.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="also open an interactive plot window",
    )
    return parser.parse_args()


def available_steps(data_dir):
    step_modification_times = {}
    for path in data_dir.glob("output_velocity_rank*_step*.txt"):
        match = STEP_PATTERN.match(path.name)
        if match:
            step = int(match.group(1))
            step_modification_times[step] = max(
                step_modification_times.get(step, 0.0),
                path.stat().st_mtime,
            )
    return step_modification_times


def load_velocity(data_dir, step):
    files = sorted(data_dir.glob(f"output_velocity_rank*_step{step}.txt"))
    if not files:
        raise FileNotFoundError(
            f"No velocity files found for step {step} in {data_dir}"
        )

    data = np.vstack([np.loadtxt(path, ndmin=2) for path in files])
    global_row = data[:, 0].astype(int)
    column = data[:, 1].astype(int)

    rows = int(global_row.max()) + 1
    cols = int(column.max()) + 1
    expected_cells = rows * cols
    if data.shape[0] != expected_cells:
        raise ValueError(
            f"Step {step} contains {data.shape[0]} cells, but a complete "
            f"{rows}x{cols} field requires {expected_cells}."
        )

    u = np.full((rows, cols), np.nan)
    v = np.full((rows, cols), np.nan)
    u[global_row, column] = data[:, 2]
    v[global_row, column] = data[:, 3]
    if np.isnan(u).any() or np.isnan(v).any():
        raise ValueError(f"Step {step} has missing grid cells")

    return u, v, len(files)


def main():
    args = parse_args()
    step_modification_times = available_steps(args.data_dir)
    if not step_modification_times:
        raise FileNotFoundError(f"No velocity output found in {args.data_dir}")

    steps = sorted(step_modification_times)
    step = (
        args.step
        if args.step is not None
        else max(steps, key=lambda value: (step_modification_times[value], value))
    )
    if step not in step_modification_times:
        raise ValueError(
            f"Step {step} is unavailable. Available range: "
            f"{steps[0]} to {steps[-1]}."
        )

    u, _, number_of_files = load_velocity(args.data_dir, step)
    rows, cols = u.shape
    if rows != 129 or cols != 129:
        print(
            f"Warning: loaded a {rows}x{cols} grid; the requested validation "
            "case uses 129x129."
        )

    center_col = cols // 2
    u_centerline = u[:, center_col] / args.u_lid

    # Solid nodes occupy rows 0 and rows-1. With halfway bounce-back, the
    # physical walls lie halfway between those solid nodes and the adjacent
    # fluid nodes. Map interior row r using y/H = (r - 0.5)/(rows - 2).
    y = np.empty(rows)
    y[0] = 0.0
    y[-1] = 1.0
    y[1:-1] = (np.arange(1, rows - 1) - 0.5) / (rows - 2)
    u_at_ghia = np.interp(GHIA_Y, y, u_centerline)

    # The moving-wall bounce-back condition acts on adjacent fluid cells. The
    # stored macroscopic velocity on the solid top node is therefore zero.
    # Compare errors only at Ghia's interior sample positions.
    interior = (GHIA_Y > 0.0) & (GHIA_Y < 1.0)
    errors = u_at_ghia[interior] - GHIA_U_RE1000[interior]
    rmse = np.sqrt(np.mean(errors**2))
    max_error = np.max(np.abs(errors))

    print(
        f"Loaded step {step} from {number_of_files} file(s): "
        f"{rows}x{cols}, center column {center_col}"
    )
    print(f"Interior-point RMSE:      {rmse:.6e}")
    print(f"Interior-point max error: {max_error:.6e}")
    print("\n       y/H     Ghia u/U      LBM u/U        error")
    for index, (position, reference, numerical) in enumerate(
        zip(GHIA_Y, GHIA_U_RE1000, u_at_ghia)
    ):
        error_text = (
            f"{numerical - reference:+13.6f}"
            if interior[index]
            else "   prescribed"
        )
        print(
            f"{position:10.4f} {reference:12.6f} "
            f"{numerical:12.6f} {error_text}"
        )

    # Plot prescribed wall velocities without altering the interior error data.
    u_centerline_plot = u_centerline.copy()
    u_centerline_plot[0] = 0.0
    u_centerline_plot[-1] = 1.0

    plt.figure(figsize=(6, 6))
    plt.plot(
        u_centerline_plot,
        y,
        label=f"LBM, {rows}x{cols}, step {step}",
    )
    plt.scatter(
        GHIA_U_RE1000,
        GHIA_Y,
        marker="o",
        facecolors="none",
        edgecolors="black",
        label="Ghia et al., Re = 1000",
    )
    plt.xlabel(r"$u/U_{\mathrm{lid}}$")
    plt.ylabel(r"$y/L$")
    plt.title("Horizontal velocity on vertical cavity centerline")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output = args.output or PROJECT_ROOT / f"compare_re1000_step{step}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    print(f"\nSaved plot to {output}")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
