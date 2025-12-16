import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# ==============================
# SETTINGS
# ==============================
MAP_IMAGE_FILE = "kroki.jpg"
EXCEL_FILE = "new.xlsx"
DEMAND_SHEET = "demand"
CAMERA_SHEET = "camera"
OUTPUT_IMAGE_FILE = "grid_locations_with_strips.png"
CHART_TITLE = "Infrastructure Map"

# GRID DEFINITION
TARGET_COLS = 80   # X direction (0 to 79)
TARGET_ROWS = 130  # Y direction (0 to 129)
SUBTRACT_ONE = 0

print("--- PROCESS STARTED ---")

# ==============================
# 1. LOAD IMAGE
# ==============================
try:
    pil_img = Image.open(MAP_IMAGE_FILE)
    img_width, img_height = pil_img.size
    map_img = mpimg.imread(MAP_IMAGE_FILE)
    print(f"Image Loaded: {img_width}x{img_height} pixels")
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find {MAP_IMAGE_FILE}")

# ==============================
# 2. LOAD DATA
# ==============================
df_dem = pd.read_excel(EXCEL_FILE, sheet_name=DEMAND_SHEET)
df_dem.columns = [c.strip().lower() for c in df_dem.columns]
df_dem = df_dem[df_dem['weight'] > 0].copy()

df_cam = pd.read_excel(EXCEL_FILE, sheet_name=CAMERA_SHEET)
df_cam.columns = [c.strip().lower() for c in df_cam.columns]

# ==============================
# 3. PREPARE COORDINATES (GRID UNITS)
# ==============================
# Demand: Centers (Index + 0.5)
df_dem['plot_x'] = (df_dem['x'] - SUBTRACT_ONE) + 0.5
df_dem['plot_y'] = (df_dem['y'] - SUBTRACT_ONE) + 0.5

# Camera: Corners (Index exactly)
df_cam['plot_x'] = (df_cam['x'] - SUBTRACT_ONE)
df_cam['plot_y'] = (df_cam['y'] - SUBTRACT_ONE)

# ==============================
# 4. PLOTTING
# ==============================
fig_dpi = 120
# Increased base width to allow plot to expand horizontally
fig_w = 14
# Calculate height to match aspect ratio of the 80x130 grid
fig_h = fig_w * (TARGET_ROWS / TARGET_COLS)

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=fig_dpi)

# --- CONFIGURING "EXCEL-STYLE" STRIPS ---
# 1. Move X-Axis to the Top
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

# 2. Set Limits (Grid Units)
ax.set_xlim(0, TARGET_COLS)
ax.set_ylim(TARGET_ROWS, 0) # Inverted Y (0 at top)

# 3. Configure Ticks (The "Strips")
major_ticks_x = np.arange(0, TARGET_COLS + 1, 5)
major_ticks_y = np.arange(0, TARGET_ROWS + 1, 5)

ax.set_xticks(major_ticks_x)
ax.set_yticks(major_ticks_y)
ax.tick_params(axis='both', which='major', labelsize=8)
ax.grid(False)

# 4. Remove internal margins
ax.margins(0)

# --- PLOT CONTENT ---

# A. Background Image
ax.imshow(
    map_img, 
    extent=[0, TARGET_COLS, TARGET_ROWS, 0], 
    aspect='auto', 
    zorder=0
)

# B. Demand Points (Blue Centers)
ax.scatter(
    df_dem['plot_x'], 
    df_dem['plot_y'], 
    s=10, 
    color='cyan', 
    marker='o', 
    label='Demand (Centers)',
    alpha=0.6,
    edgecolors='blue',
    linewidth=0.5,
    zorder=2,
    clip_on=False
)

# C. Camera Points (Red Corners)
ax.scatter(
    df_cam['plot_x'], 
    df_cam['plot_y'], 
    s=20, 
    color='#ff0033', 
    marker='D', 
    label='Cameras (Corners)',
    alpha=0.9,
    edgecolors='white',
    linewidth=0.8,
    zorder=3,
    clip_on=False 
)

# ==============================
# 5. SAVE
# ==============================
ax.set_xlabel("Column Index (0-79)", fontsize=10, labelpad=10)
ax.set_ylabel("Row Index (0-129)", fontsize=10, labelpad=10)
# Lift title slightly less than before
ax.set_title(CHART_TITLE, y=1.06, fontsize=14, fontweight='bold')

# --- LEGEND MOVED OUTSIDE ---
# bbox_to_anchor=(1.02, 1): Closer to the plot edge to allow horizontal expansion
legend = ax.legend(loc='upper left', framealpha=0.9, bbox_to_anchor=(1.02, 1))
legend.get_frame().set_edgecolor('black')

# Increased padding slightly to ensure everything is included
plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches='tight', pad_inches=0.2)
plt.close()

print(f"Map with strips and outside legend saved to: {OUTPUT_IMAGE_FILE}")