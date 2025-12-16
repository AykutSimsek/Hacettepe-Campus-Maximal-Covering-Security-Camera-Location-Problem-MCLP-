import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

# ==============================
# SETTINGS
# ==============================
EXCEL_FILE = "new.xlsx"
DEMAND_SHEET = "demand"
OUTPUT_IMAGE_FILE = "grid_heatmap.png"
HEATMAP_TITLE = "Campus Demand Distribution Heatmap (Grid View)"

# GRID DEFINITION
TARGET_COLS = 80   # X direction
TARGET_ROWS = 130  # Y direction

# Coordinate adjustment
subtract_one = 0

print("--- PROCESS STARTED ---")

# ==============================
# LOAD DATA
# ==============================
df = pd.read_excel(EXCEL_FILE, sheet_name=DEMAND_SHEET)
df.columns = [c.strip().lower() for c in df.columns]

required_cols = {"x", "y", "weight"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")

# ==============================
# CREATE MATRIX
# ==============================
heatmap_matrix = np.full((TARGET_ROWS, TARGET_COLS), np.nan)

for _, row in df.iterrows():
    try:
        c = int(row["x"]) - subtract_one
        r = int(row["y"]) - subtract_one
        w = row["weight"]

        if 0 <= r < TARGET_ROWS and 0 <= c < TARGET_COLS:
            if pd.notna(w) and w > 0:
                heatmap_matrix[r, c] = w
    except:
        continue

# ==============================
# PLOT SETUP
# ==============================
# 1. Calculate precise aspect ratio
aspect_ratio = TARGET_ROWS / TARGET_COLS
base_width = 10  # inches
fig_height = base_width * aspect_ratio

fig, ax = plt.subplots(figsize=(base_width, fig_height), dpi=100)

# 2. Strict Limits (Exactly 0 to 80, 0 to 130)
ax.set_xlim(0, TARGET_COLS)
ax.set_ylim(TARGET_ROWS, 0) # Inverted Y (0 at top)

# 3. Remove any internal Matplotlib margins
ax.margins(0)

# 4. Configure Grid
# Major ticks (Labels)
ax.set_xticks(np.arange(0, TARGET_COLS + 1, 10))
ax.set_yticks(np.arange(0, TARGET_ROWS + 1, 10))

# Minor ticks (Grid lines at 0.5 positions to center pixels)
ax.set_xticks(np.arange(0, TARGET_COLS, 1), minor=True)
ax.set_yticks(np.arange(0, TARGET_ROWS, 1), minor=True)

# Draw grid
ax.grid(which="minor", color="black", linestyle="-", linewidth=0.3, alpha=0.3, zorder=2)
ax.tick_params(which="minor", size=0) # Hide minor tick marks

# 5. Labels & Title
ax.set_xlabel("X Index")
ax.set_ylabel("Y Index")
ax.set_title(HEATMAP_TITLE, fontsize=12, fontweight='bold', pad=15)

# ==============================
# HEATMAP
# ==============================
heat_cmap = LinearSegmentedColormap.from_list("gyr", ["green", "yellow", "red"])
heat_cmap.set_bad(color="white") # Background color

positive_vals = heatmap_matrix[np.isfinite(heatmap_matrix)]
norm = None
if positive_vals.size > 0:
    norm = PowerNorm(gamma=0.4, vmin=positive_vals.min(), vmax=positive_vals.max())

# Draw Heatmap
# 'extent' maps the data pixels strictly to [0, 80] and [0, 130]
# origin='upper' ensures (0,0) is top-left
img = ax.imshow(
    heatmap_matrix,
    extent=[0, TARGET_COLS, TARGET_ROWS, 0],
    origin='upper', 
    cmap=heat_cmap,
    norm=norm,
    interpolation="nearest",
    aspect='equal', # Forces square pixels
    zorder=1
)

# ==============================
# COLORBAR & SAVE
# ==============================
if positive_vals.size > 0:
    cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Weight")

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches="tight", dpi=150)
plt.close()

print(f"File saved: {OUTPUT_IMAGE_FILE}")