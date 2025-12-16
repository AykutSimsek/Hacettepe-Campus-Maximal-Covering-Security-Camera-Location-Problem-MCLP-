import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from PIL import Image

# ==============================
# SETTINGS
# ==============================
MAP_IMAGE_FILE = "kroki.jpg"
EXCEL_FILE = "new.xlsx"
DEMAND_SHEET = "demand"
OUTPUT_IMAGE_FILE = "campus_heatmap.png"
HEATMAP_TITLE = "Campus Demand Distribution Heatmap"

# GRID DEFINITION
TARGET_COLS = 80   # X direction
TARGET_ROWS = 130  # Y direction

# Coordinate adjustment setting
# If Excel x,y starts from 1 -> set to 1
# If Excel x,y starts from 0 -> set to 0
subtract_one = 0

print("--- PROCESS STARTED ---")

# ==============================
# LOAD IMAGE
# ==============================
pil_img = Image.open(MAP_IMAGE_FILE)
img_width, img_height = pil_img.size
map_img = mpimg.imread(MAP_IMAGE_FILE)

print(f"Image size: {img_width}px x {img_height}px")

# ==============================
# LOAD DATA
# ==============================
df = pd.read_excel(EXCEL_FILE, sheet_name=DEMAND_SHEET)
# Standardize column names
df.columns = [c.strip().lower() for c in df.columns]

required_cols = {"x", "y", "weight"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")

print(f"Demand points loaded: {len(df)}")

# ==============================
# CREATE MATRICES
# ==============================
heatmap_matrix = np.full((TARGET_ROWS, TARGET_COLS), np.nan)

mapped = 0
positive_count = 0

for _, row in df.iterrows():
    try:
        c = int(row["x"]) - subtract_one
        r = int(row["y"]) - subtract_one
        w = row["weight"]

        if 0 <= r < TARGET_ROWS and 0 <= c < TARGET_COLS:
            if pd.notna(w) and w > 0:
                heatmap_matrix[r, c] = w
                positive_count += 1
            mapped += 1
    except:
        continue

print(f"Mapped cells: {mapped}")
print(f"Positive weights: {positive_count}")

# ==============================
# PLOT
# ==============================
fig, ax = plt.subplots(
    figsize=(img_width / 100, img_height / 100),
    dpi=100
)

# BACKGROUND
ax.imshow(map_img)
ax.axis("off")

# --- ADD TITLE ---
# pad=20 adds space between the title and the image
ax.set_title(HEATMAP_TITLE, fontsize=8, fontweight='bold', pad=20)

# --- POSITIVE WEIGHTS (HEATMAP) ---
heat_cmap = LinearSegmentedColormap.from_list(
    "green_yellow_red",
    ["green", "yellow", "red"]
)
heat_cmap.set_bad(color="none")  # NaN areas are transparent

# Calculate normalization for sensitivity
positive_vals = heatmap_matrix[np.isfinite(heatmap_matrix)]
norm = None

if positive_vals.size > 0:
    # PowerNorm (gamma < 1) increases sensitivity for small values
    # It "stretches" the lower end of the color scale
    norm = PowerNorm(gamma=0.4, vmin=positive_vals.min(), vmax=positive_vals.max())

ax.imshow(
    heatmap_matrix,
    extent=[0, img_width, img_height, 0],
    cmap=heat_cmap,
    norm=norm,        # Use the sensitive normalization here
    alpha=0.6,
    interpolation="nearest",
    zorder=2
)

# ==============================
# COLORBAR (SAFE)
# ==============================
if positive_vals.size > 0:
    sm = plt.cm.ScalarMappable(
        cmap=heat_cmap,
        norm=norm
    )
    sm.set_array([])
    # Adjusted fraction and pad slightly to accommodate title better if needed
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Weight (Power Scale)")

# ==============================
# SAVE
# ==============================
# bbox_inches="tight" ensures the title is included in the saved image
plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches="tight", pad_inches=0.1)
plt.close()

print("--- PROCESS FINISHED ---")
print(f"Output saved as: {OUTPUT_IMAGE_FILE}")