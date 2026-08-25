import glob
import numpy as np
import matplotlib.pyplot as plt

rows = 100
cols = 100
step = 10000

files = sorted(
    glob.glob(f"data/v/output_velocity_rank*_step{step}.txt")
)

if not files:
    raise FileNotFoundError(
        f"No velocity files found for step {step}"
    )

print("Loading:")
for filename in files:
    print(filename)

# Load and combine all MPI-rank files
parts = [np.loadtxt(filename) for filename in files]
data = np.vstack(parts)

global_row = data[:, 0].astype(int)
col = data[:, 1].astype(int)

horizontal_velocity = data[:, 2]
vertical_velocity = data[:, 3]

# Arrays use [vertical row, horizontal column]
u = np.zeros((rows, cols))
v = np.zeros((rows, cols))

for k in range(len(data)):
    row = global_row[k]
    column = col[k]
    print(k)
    u[row, column] = horizontal_velocity[k]
    v[row, column] = vertical_velocity[k]

speed = np.sqrt(u**2 + v**2)

# Horizontal coordinate is the column.
# Vertical coordinate is the global row.
x = np.linspace(0.0, 1.0, cols)
y = np.linspace(0.0, 1.0, rows)

X, Y = np.meshgrid(x, y)

plt.figure(figsize=(7, 6))

stream = plt.streamplot(
    X,
    Y,
    u,
    v,
    color=speed,
    cmap="viridis",
    density=2.0,
    linewidth=1.0,
    arrowsize=1.0
)

plt.colorbar(
    stream.lines,
    label="Velocity magnitude |u|"
)

plt.xlabel("x/L")
plt.ylabel("y/L")
plt.title(f"Lid-driven cavity flow, step {step}")

plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.gca().set_aspect("equal")

plt.tight_layout()
plt.show()