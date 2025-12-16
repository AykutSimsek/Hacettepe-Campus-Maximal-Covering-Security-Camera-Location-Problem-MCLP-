import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import math
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import sparse
from matplotlib.lines import Line2D
import time
import random





# --- SETTINGS ---
MAP_IMAGE_FILE = 'kroki.jpg'       # Campus map image
OUTPUT_IMAGE_FILE = 'grid_kroki.png'

# Desired grid configuration
TARGET_ROWS = 130  # Number of rows (along Y-axis)
TARGET_COLS = 80   # Number of columns (along X-axis)

print("--- PROCESS STARTED ---")

# 1. LOAD IMAGE AND GET DIMENSIONS
try:
    # Use PIL to obtain original pixel dimensions
    pil_img = Image.open(MAP_IMAGE_FILE)
    img_width, img_height = pil_img.size
    print(f"Image dimensions: {img_width}px (width) x {img_height}px (height)")

    # Read image for Matplotlib
    map_img = mpimg.imread(MAP_IMAGE_FILE)
except FileNotFoundError:
    print(f"ERROR: '{MAP_IMAGE_FILE}' was not found.")
    exit()

# 2. COMPUTE GRID SPACING (IN PIXELS)
# Divide total width by number of columns and total height by number of rows
step_x = img_width / TARGET_COLS
step_y = img_height / TARGET_ROWS

print(f"Single grid cell size: {step_x:.2f}px x {step_y:.2f}px")

# 3. CREATE THE PLOT
# Adjust figure size based on the image aspect ratio
fig_dpi = 200
fig_width = 20  # inches
fig_height = fig_width * (img_height / img_width)

fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=fig_dpi)

# A) BACKGROUND IMAGE
# imshow displays the image in pixel coordinates by default
ax.imshow(map_img)

# B) DRAW GRID LINES
# Vertical lines (column boundaries)
for i in range(TARGET_COLS + 1):
    x_pos = i * step_x
    ax.axvline(
        x=x_pos,
        color='black',
        linestyle='-',
        linewidth=0.5,
        alpha=0.6
    )

# Horizontal lines (row boundaries)
for i in range(TARGET_ROWS + 1):
    y_pos = i * step_y
    ax.axhline(
        y=y_pos,
        color='black',
        linestyle='-',
        linewidth=0.5,
        alpha=0.6
    )

# 4. FINAL ADJUSTMENTS AND SAVE
# Fix axes to image boundaries
ax.set_xlim(0, img_width)
ax.set_ylim(img_height, 0)  # Invert Y-axis to match image coordinate system
ax.axis('off')              # Remove axes and frame

plt.tight_layout(pad=0)
plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches='tight', pad_inches=0)

print("-" * 30)
print("Process completed successfully!")
print(f"Gridded map saved as: {OUTPUT_IMAGE_FILE}")
print(f"Grid configuration: {TARGET_ROWS} rows x {TARGET_COLS} columns")
print("-" * 30)


















# --- SETTINGS ---
INPUT_FILE = 'new.xlsx'
SHEET_NAME = 'demand'
OUTPUT_FILE = 'demand_points.xlsx'
GRID_SIZE = 10  # meters

print("--- PROCESS STARTED ---")

# 1. LOAD DATA
try:
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    print(f"Data loaded successfully: {len(df)} rows.")
except FileNotFoundError:
    print(f"ERROR: The file '{INPUT_FILE}' was not found in the current directory.")
    exit()
except ValueError as e:
    print(f"ERROR: No sheet named '{SHEET_NAME}' was found. {e}")
    exit()

# 2. COORDINATE CALCULATION (ONLY IF weight != 0)
try:
    df['coord_x'] = np.where(
        df['weight'] != 0,
        (df['x'] * GRID_SIZE) + (GRID_SIZE / 2),
        np.nan
    )

    df['coord_y'] = np.where(
        df['weight'] != 0,
        (df['y'] * GRID_SIZE) + (GRID_SIZE / 2),
        np.nan
    )
except KeyError as e:
    print(f"ERROR: Missing column: {e}")
    print("Required columns: x, y, weight")
    exit()

# 3. CHECK AND SAVE
print("\n--- First 5 Rows (Verification) ---")
print(df[['x', 'y', 'weight', 'coord_x', 'coord_y']].head())

df.to_excel(OUTPUT_FILE, index=False)
print(f"\nFile successfully saved: {OUTPUT_FILE}")




















# --- SETTINGS ---
INPUT_FILE = 'new.xlsx'
CAMERA_SHEET = 'camera'
DEMAND_SHEET = 'demand'
OUTPUT_FILE = 'camera_points.xlsx'
GRID_SIZE = 10  # meters

print("--- GENERATING CANDIDATE CAMERA LOCATIONS ---")

# 1. LOAD CAMERA DATA
df_cam = pd.read_excel(INPUT_FILE, sheet_name=CAMERA_SHEET)
print(f"Camera data loaded: {len(df_cam)} rows.")

# 2. LOAD DEMAND DATA (WEIGHT SOURCE)
df_dem = pd.read_excel(INPUT_FILE, sheet_name=DEMAND_SHEET)
print(f"Demand data loaded: {len(df_dem)} rows.")

# Keep only necessary columns
df_dem = df_dem[['x', 'y', 'weight']]

# 3. MERGE ON (x, y)
df = df_cam.merge(df_dem, on=['x', 'y'], how='left')

# If grid not in demand, assume weight = 0
df['weight'] = df['weight'].fillna(0)

# 4. ASSIGN CANDIDATE IDS
df['candidate_id'] = range(1, len(df) + 1)

# 5. COMPUTE COORDINATES
# weight == 0 -> coordinates NaN
df['coord_x'] = np.where(
    df['weight'] != 0,
    df['x'] * GRID_SIZE,
    np.nan
)

df['coord_y'] = np.where(
    df['weight'] != 0,
    df['y'] * GRID_SIZE,
    np.nan
)

# 6. VERIFICATION
print("\n--- First 5 Rows (Verification) ---")
print(df[['candidate_id', 'x', 'y', 'weight', 'coord_x', 'coord_y']].head())

# 7. REMOVE WEIGHT AND SAVE
# Drop the 'weight' column before saving
df_final = df.drop(columns=['weight'])

df_final.to_excel(OUTPUT_FILE, index=False)
print(f"\nFile successfully saved (without weight column): {OUTPUT_FILE}")






















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



















# --- SETTINGS ---
DEMAND_FILE = 'demand_points.xlsx'     # Demand points (rows)
CAMERA_FILE = 'camera_points.xlsx'     # Candidate camera points (columns)
OUTPUT_COVERAGE_FILE = 'Coverage_Matrix.csv'

# Coverage parameters
R = 50.0        # Coverage radius (meters)
FOV = 90.0      # Field of View (degrees)
# Since FOV = 90°, it is reasonable to use four main orientations
CAMERA_ANGLES = [0, 90, 180, 270]

print("--- PROCESS STARTED ---")

# --- 1. LOAD DATA ---
try:
    # Demand data
    df_demand = pd.read_excel(DEMAND_FILE)
    # Standardize column names (lowercase, no spaces)
    df_demand.columns = [c.strip().lower() for c in df_demand.columns]

    # Camera candidate data
    df_camera = pd.read_excel(CAMERA_FILE)
    df_camera.columns = [c.strip().lower() for c in df_camera.columns]

    print(f"Demand points loaded: {len(df_demand)}")
    print(f"Camera candidates loaded: {len(df_camera)}")

except FileNotFoundError as e:
    print(f"ERROR: File not found - {e}")
    exit()

# --- 2. FAST COMPUTATION USING NUMPY ---
print("Computing coverage matrix (vectorized operations)...")

# Extract coordinates as NumPy arrays (coord_x, coord_y are in meters)
# Demand points (M points)
d_x = df_demand['coord_x'].values
d_y = df_demand['coord_y'].values

# Store demand IDs (use point_id if available, otherwise DataFrame index)
d_ids = (
    df_demand['point_id'].values
    if 'point_id' in df_demand.columns
    else df_demand.index.values
)

# Camera candidate points (N points)
c_x = df_camera['coord_x'].values
c_y = df_camera['coord_y'].values

# Store camera IDs
c_ids = (
    df_camera['candidate_id'].values
    if 'candidate_id' in df_camera.columns
    else df_camera.index.values
)

# --- A. DISTANCE CHECK ---
# Compute pairwise distances using broadcasting
# Results in an (M x N) matrix
delta_x = d_x[:, np.newaxis] - c_x
delta_y = d_y[:, np.newaxis] - c_y

# Squared Euclidean distance (faster than taking square roots)
dist_sq = delta_x ** 2 + delta_y ** 2
r_sq = R ** 2

# Distance mask (True if within radius R)
dist_mask = dist_sq <= r_sq

# --- B. ANGLE CHECK ---
# Compute angle for each demand–camera pair (radians, range: -pi to +pi)
# arctan2(y, x) returns the angle relative to the positive x-axis
angles_rad = np.arctan2(delta_y, delta_x)

# Convert to degrees and map to [0, 360)
angles_deg = (np.degrees(angles_rad) + 360) % 360

# Prepare containers for the final DataFrame
column_data = {}

print(f"Performing angular checks ({len(CAMERA_ANGLES)} orientations)...")

for camera_angle in CAMERA_ANGLES:
    # Mathematical angle conversion:
    # Geographic 0° (North) corresponds to 90° in mathematical coordinates
    center_angle = (90 - camera_angle) % 360

    # Compute the minimum angular difference
    # Formula for shortest angular distance:
    # (a - b + 180) % 360 - 180
    diff = np.abs((angles_deg - center_angle + 180) % 360 - 180)

    # Angular mask (within half of the FOV)
    angle_mask = diff <= (FOV / 2.0)

    # --- C. COMBINE DISTANCE AND ANGLE CONDITIONS ---
    # Coverage exists only if both distance and angle conditions are satisfied
    total_coverage = dist_mask & angle_mask

    # Generate column names: "CandidateID_Angle"
    current_col_names = [f"{cid}_{camera_angle}" for cid in c_ids]

    # Store coverage results column by column
    # total_coverage[:, i] corresponds to camera i for all demand points
    for i, col_name in enumerate(current_col_names):
        # Store as int8 to reduce memory usage
        column_data[col_name] = total_coverage[:, i].astype(np.int8)

print("Combining results and writing to CSV...")
# Create the coverage DataFrame
df_coverage = pd.DataFrame(column_data, index=d_ids)

# Name the index for clarity in Excel/CSV
df_coverage.index.name = 'Demand_Point_ID'

# Save to CSV
df_coverage.to_csv(OUTPUT_COVERAGE_FILE)

print("-" * 30)
print(f"Success! '{OUTPUT_COVERAGE_FILE}' has been created.")
print(f"Matrix size: {df_coverage.shape[0]} rows x {df_coverage.shape[1]} columns")
print(f"Parameters used: R = {R} m, FOV = {FOV} degrees")
print("-" * 30)























import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import sparse
from matplotlib.lines import Line2D
import time
import random

# ==============================
# 1. SETTINGS
# ==============================
INPUT_FILE = "new.xlsx"
DEMAND_SHEET = "demand"
CAMERA_SHEET = "camera"

# --- SPECIFIC OUTPUT FOR SOLUTION 1 ---
OUTPUT_IMAGE_FILE = "greedy_heatmap_soln1.png"
OUTPUT_CURVE_FILE = "greedy_growth_curve_soln1.png"
OUTPUT_EXCEL_FILE = "greedy_solution_final_soln1.xlsx"  # <--- As requested
HEATMAP_TITLE = "Solution 1: Randomized Greedy (300 Cameras)"

# Optimization Parameters
CAMERA_COUNT_P = 300  # <--- Run for exactly 300 cameras
CAMERA_RADIUS = 5
FOV_HALF_ANGLE = 45

# Randomization (RCL)
RCL_SIZE = 5
RANDOM_SEED = None  # Change to integer (e.g. 42) to fix the random choices

# Grid Settings
TARGET_COLS = 80
TARGET_ROWS = 130
SUBTRACT_ONE = 0 
BATCH_SIZE = 1000 

print(f"--- PROCESS STARTED: GENERATING {OUTPUT_EXCEL_FILE} ---")
start_time = time.time()

# ==============================
# 2. LOAD DATA
# ==============================
df_demand = pd.read_excel(INPUT_FILE, sheet_name=DEMAND_SHEET)
df_demand.columns = [c.strip().lower() for c in df_demand.columns]
df_demand = df_demand[df_demand['weight'] > 0].copy()
df_demand['x'] = df_demand['x'] - SUBTRACT_ONE
df_demand['y'] = df_demand['y'] - SUBTRACT_ONE

TOTAL_DEMAND_WEIGHT = df_demand['weight'].sum()

df_cam_locs = pd.read_excel(INPUT_FILE, sheet_name=CAMERA_SHEET)
df_cam_locs.columns = [c.strip().lower() for c in df_cam_locs.columns]
df_cam_locs['x'] = df_cam_locs['x'] - SUBTRACT_ONE
df_cam_locs['y'] = df_cam_locs['y'] - SUBTRACT_ONE

print(f"Demand Points: {len(df_demand)} | Total Weight: {TOTAL_DEMAND_WEIGHT:,.0f}")
print(f"Camera Locations: {len(df_cam_locs)}")

# ==============================
# 3. GENERATE CANDIDATES
# ==============================
candidates_list = []
directions = [0, 90, 180, 270]

for idx, row in df_cam_locs.iterrows():
    for angle in directions:
        candidates_list.append({
            'loc_id': idx,
            'x': row['x'],
            'y': row['y'],
            'dir_angle': angle
        })

df_candidates = pd.DataFrame(candidates_list)
df_candidates['candidate_id'] = range(len(df_candidates))
num_cands = len(df_candidates)

# ==============================
# 4. COMPUTE COVERAGE (BATCHED)
# ==============================
print("Computing Coverage Matrix...")
dem_x = df_demand['x'].values[:, np.newaxis]
dem_y = df_demand['y'].values[:, np.newaxis]
sparse_chunks = []

for start_idx in range(0, num_cands, BATCH_SIZE):
    end_idx = min(start_idx + BATCH_SIZE, num_cands)
    batch_df = df_candidates.iloc[start_idx:end_idx]
    
    cand_x = batch_df['x'].values[np.newaxis, :]
    cand_y = batch_df['y'].values[np.newaxis, :]
    cand_dirs = batch_df['dir_angle'].values[np.newaxis, :]
    
    dx = dem_x - cand_x
    dy = dem_y - cand_y
    distances = np.sqrt(dx**2 + dy**2)
    point_angles = np.degrees(np.arctan2(dx, -dy))
    point_angles = (point_angles + 360) % 360
    angle_diff = np.abs(point_angles - cand_dirs)
    angle_diff = np.minimum(angle_diff, 360 - angle_diff)
    
    mask = (distances <= CAMERA_RADIUS) & (angle_diff <= FOV_HALF_ANGLE) & (distances > 0)
    sparse_chunks.append(sparse.csc_matrix(mask))

coverage_matrix = sparse.hstack(sparse_chunks).tocsc()

# ==============================
# 5. RANDOMIZED GREEDY OPTIMIZATION
# ==============================
print(f"Optimizing for {CAMERA_COUNT_P} cameras using RCL={RCL_SIZE}...")

current_weights = df_demand['weight'].values.copy()
selected_indices = []

# List to track history for scenarios
history = []

for i in range(CAMERA_COUNT_P):
    # A. Calculate gains
    gains = coverage_matrix.T @ current_weights
    
    valid_gains_mask = gains > 0
    valid_indices = np.where(valid_gains_mask)[0]
    
    if len(valid_indices) == 0:
        print("No more gain possible (optimization stopped early).")
        break
        
    # B. RCL Selection (Top 5 Best)
    current_rcl_size = min(RCL_SIZE, len(valid_indices))
    # argpartition puts the top K elements at the end
    top_k_indices_unsorted = np.argpartition(gains[valid_indices], -current_rcl_size)[-current_rcl_size:]
    top_candidates = valid_indices[top_k_indices_unsorted]
    
    # C. Random Pick from Top K
    selected_idx = np.random.choice(top_candidates)
    selected_indices.append(selected_idx)
    
    # D. Update Weights
    covered_indices = coverage_matrix[:, selected_idx].indices
    current_weights[covered_indices] = 0
    
    # E. Record History
    current_covered_weight = TOTAL_DEMAND_WEIGHT - current_weights.sum()
    pct = (current_covered_weight / TOTAL_DEMAND_WEIGHT) * 100
    
    history.append({
        'camera_count': i + 1,
        'coverage_pct': pct,
        'marginal_gain': gains[selected_idx]
    })

# ==============================
# 6. SCENARIO REPORTING
# ==============================
end_time = time.time()
run_time = end_time - start_time
final_pct = history[-1]['coverage_pct']

print("\n" + "="*50)
print(f" SCENARIO ANALYSIS: COVERAGE FOR {OUTPUT_EXCEL_FILE}")
print("="*50)
print(f"{'Cameras':<10} | {'Coverage %':<15} | {'Gain':<10}")
print("-" * 50)

# Print specific milestones to understand growth
milestones = [10, 50, 100, 150, 200, 250, 300]
for m in milestones:
    if m <= len(history):
        rec = history[m-1] 
        print(f"{rec['camera_count']:<10} | {rec['coverage_pct']:<6.2f}%         | +{rec['marginal_gain']:.0f}")

print("-" * 50)
print(f"FINAL: {len(selected_indices)} cams | {final_pct:.2f}% | {run_time:.2f}s")
print("="*50 + "\n")

# ==============================
# 7. SAVE RESULTS TO EXCEL
# ==============================
print(f"Saving to {OUTPUT_EXCEL_FILE}...")

# 1. Camera Solution
df_results_cam = df_candidates.iloc[selected_indices].copy()
df_results_cam = df_results_cam[['candidate_id', 'loc_id', 'x', 'y', 'dir_angle']]
df_results_cam.rename(columns={'dir_angle': 'direction_degrees'}, inplace=True)

# 2. Demand Coverage Status
selected_matrix = coverage_matrix[:, selected_indices]
coverage_counts_per_point = np.array(selected_matrix.sum(axis=1)).flatten()

df_results_dem = df_demand.copy()
df_results_dem['cameras_covering'] = coverage_counts_per_point
df_results_dem['is_covered'] = np.where(df_results_dem['cameras_covering'] > 0, 'Yes', 'No')

# 3. Growth History
df_history = pd.DataFrame(history)

with pd.ExcelWriter(OUTPUT_EXCEL_FILE) as writer:
    df_results_cam.to_excel(writer, sheet_name='Selected_Cameras', index=False)
    df_results_dem.to_excel(writer, sheet_name='Demand_Coverage', index=False)
    df_history.to_excel(writer, sheet_name='Growth_History', index=False)

print("Excel file created successfully.")

# ==============================
# 8. VISUALIZATION: GROWTH CURVE
# ==============================
plt.figure(figsize=(10, 6), dpi=100)
x_vals = [h['camera_count'] for h in history]
y_vals = [h['coverage_pct'] for h in history]

plt.plot(x_vals, y_vals, color='#2980b9', linewidth=2, label='Coverage %')

for m in milestones:
    if m <= len(history):
        val = history[m-1]['coverage_pct']
        plt.plot(m, val, 'o', color='#e74c3c')
        plt.annotate(f"{val:.1f}%", (m, val), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

plt.title(f"Coverage Growth (Solution 1)\nMax: {final_pct:.2f}%", fontsize=14, fontweight='bold')
plt.xlabel("Number of Cameras")
plt.ylabel("Coverage Percentage (%)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_CURVE_FILE)
print(f"Curve saved: {OUTPUT_CURVE_FILE}")

# ==============================
# 9. VISUALIZATION: HEATMAP
# ==============================
final_heatmap_grid = np.zeros((TARGET_ROWS, TARGET_COLS))
dem_x_flat = df_demand['x'].values
dem_y_flat = df_demand['y'].values

for i, count in enumerate(coverage_counts_per_point):
    if count > 0:
        r, c = int(dem_y_flat[i]), int(dem_x_flat[i])
        if 0 <= r < TARGET_ROWS and 0 <= c < TARGET_COLS:
            final_heatmap_grid[r, c] = count

aspect_ratio = TARGET_ROWS / TARGET_COLS
base_width = 10 
fig_height = base_width * aspect_ratio
fig, ax = plt.subplots(figsize=(base_width, fig_height), dpi=120)

ax.set_xlim(0, TARGET_COLS)
ax.set_ylim(TARGET_ROWS, 0) 
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_xticks(np.arange(0, TARGET_COLS + 1, 10))
ax.set_yticks(np.arange(0, TARGET_ROWS + 1, 10))
ax.set_xticks(np.arange(0.5, TARGET_COLS, 1), minor=True)
ax.set_yticks(np.arange(0.5, TARGET_ROWS, 1), minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
ax.tick_params(which='minor', bottom=False, left=False)

colors = ['#ffffff00', '#800080', '#00ced1', '#ffd700'] 
cmap = ListedColormap(colors)
bounds = [0, 1, 2, 3, 100]
norm = BoundaryNorm(bounds, cmap.N)

ax.imshow(final_heatmap_grid, extent=[0, TARGET_COLS, TARGET_ROWS, 0], cmap=cmap, norm=norm, interpolation='nearest', zorder=1)

legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#800080', markersize=10, label='1 Cam'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#00ced1', markersize=10, label='2 Cams'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#ffd700', markersize=10, label='3+ Cams')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

ax.set_title(f"{HEATMAP_TITLE}\nCoverage: {final_pct:.1f}%", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches='tight')
print(f"Heatmap saved: {OUTPUT_IMAGE_FILE}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg # Not needed for grid view
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import sparse
from matplotlib.lines import Line2D
import time
import random  # <--- Added for randomization

















import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import sparse
from matplotlib.lines import Line2D
import time
import random

# ==============================
# 1. SETTINGS
# ==============================
INPUT_FILE = "new.xlsx"
DEMAND_SHEET = "demand"
CAMERA_SHEET = "camera"

# --- SPECIFIC OUTPUT FOR SOLUTION 1 ---
OUTPUT_IMAGE_FILE = "greedy_heatmap_soln2.png"
OUTPUT_CURVE_FILE = "greedy_growth_curve_soln2.png"
OUTPUT_EXCEL_FILE = "greedy_solution_final_soln2.xlsx"  # <--- As requested
HEATMAP_TITLE = "Solution 2: Randomized Greedy (300 Cameras)"

# Optimization Parameters
CAMERA_COUNT_P = 300  # <--- Run for exactly 300 cameras
CAMERA_RADIUS = 5
FOV_HALF_ANGLE = 45

# Randomization (RCL)
RCL_SIZE = 5
RANDOM_SEED = None  # Change to integer (e.g. 42) to fix the random choices

# Grid Settings
TARGET_COLS = 80
TARGET_ROWS = 130
SUBTRACT_ONE = 0 
BATCH_SIZE = 1000 

print(f"--- PROCESS STARTED: GENERATING {OUTPUT_EXCEL_FILE} ---")
start_time = time.time()

# ==============================
# 2. LOAD DATA
# ==============================
df_demand = pd.read_excel(INPUT_FILE, sheet_name=DEMAND_SHEET)
df_demand.columns = [c.strip().lower() for c in df_demand.columns]
df_demand = df_demand[df_demand['weight'] > 0].copy()
df_demand['x'] = df_demand['x'] - SUBTRACT_ONE
df_demand['y'] = df_demand['y'] - SUBTRACT_ONE

TOTAL_DEMAND_WEIGHT = df_demand['weight'].sum()

df_cam_locs = pd.read_excel(INPUT_FILE, sheet_name=CAMERA_SHEET)
df_cam_locs.columns = [c.strip().lower() for c in df_cam_locs.columns]
df_cam_locs['x'] = df_cam_locs['x'] - SUBTRACT_ONE
df_cam_locs['y'] = df_cam_locs['y'] - SUBTRACT_ONE

print(f"Demand Points: {len(df_demand)} | Total Weight: {TOTAL_DEMAND_WEIGHT:,.0f}")
print(f"Camera Locations: {len(df_cam_locs)}")

# ==============================
# 3. GENERATE CANDIDATES
# ==============================
candidates_list = []
directions = [0, 90, 180, 270]

for idx, row in df_cam_locs.iterrows():
    for angle in directions:
        candidates_list.append({
            'loc_id': idx,
            'x': row['x'],
            'y': row['y'],
            'dir_angle': angle
        })

df_candidates = pd.DataFrame(candidates_list)
df_candidates['candidate_id'] = range(len(df_candidates))
num_cands = len(df_candidates)

# ==============================
# 4. COMPUTE COVERAGE (BATCHED)
# ==============================
print("Computing Coverage Matrix...")
dem_x = df_demand['x'].values[:, np.newaxis]
dem_y = df_demand['y'].values[:, np.newaxis]
sparse_chunks = []

for start_idx in range(0, num_cands, BATCH_SIZE):
    end_idx = min(start_idx + BATCH_SIZE, num_cands)
    batch_df = df_candidates.iloc[start_idx:end_idx]
    
    cand_x = batch_df['x'].values[np.newaxis, :]
    cand_y = batch_df['y'].values[np.newaxis, :]
    cand_dirs = batch_df['dir_angle'].values[np.newaxis, :]
    
    dx = dem_x - cand_x
    dy = dem_y - cand_y
    distances = np.sqrt(dx**2 + dy**2)
    point_angles = np.degrees(np.arctan2(dx, -dy))
    point_angles = (point_angles + 360) % 360
    angle_diff = np.abs(point_angles - cand_dirs)
    angle_diff = np.minimum(angle_diff, 360 - angle_diff)
    
    mask = (distances <= CAMERA_RADIUS) & (angle_diff <= FOV_HALF_ANGLE) & (distances > 0)
    sparse_chunks.append(sparse.csc_matrix(mask))

coverage_matrix = sparse.hstack(sparse_chunks).tocsc()

# ==============================
# 5. RANDOMIZED GREEDY OPTIMIZATION
# ==============================
print(f"Optimizing for {CAMERA_COUNT_P} cameras using RCL={RCL_SIZE}...")

current_weights = df_demand['weight'].values.copy()
selected_indices = []

# List to track history for scenarios
history = []

for i in range(CAMERA_COUNT_P):
    # A. Calculate gains
    gains = coverage_matrix.T @ current_weights
    
    valid_gains_mask = gains > 0
    valid_indices = np.where(valid_gains_mask)[0]
    
    if len(valid_indices) == 0:
        print("No more gain possible (optimization stopped early).")
        break
        
    # B. RCL Selection (Top 5 Best)
    current_rcl_size = min(RCL_SIZE, len(valid_indices))
    # argpartition puts the top K elements at the end
    top_k_indices_unsorted = np.argpartition(gains[valid_indices], -current_rcl_size)[-current_rcl_size:]
    top_candidates = valid_indices[top_k_indices_unsorted]
    
    # C. Random Pick from Top K
    selected_idx = np.random.choice(top_candidates)
    selected_indices.append(selected_idx)
    
    # D. Update Weights
    covered_indices = coverage_matrix[:, selected_idx].indices
    current_weights[covered_indices] = 0
    
    # E. Record History
    current_covered_weight = TOTAL_DEMAND_WEIGHT - current_weights.sum()
    pct = (current_covered_weight / TOTAL_DEMAND_WEIGHT) * 100
    
    history.append({
        'camera_count': i + 1,
        'coverage_pct': pct,
        'marginal_gain': gains[selected_idx]
    })

# ==============================
# 6. SCENARIO REPORTING
# ==============================
end_time = time.time()
run_time = end_time - start_time
final_pct = history[-1]['coverage_pct']

print("\n" + "="*50)
print(f" SCENARIO ANALYSIS: COVERAGE FOR {OUTPUT_EXCEL_FILE}")
print("="*50)
print(f"{'Cameras':<10} | {'Coverage %':<15} | {'Gain':<10}")
print("-" * 50)

# Print specific milestones to understand growth
milestones = [10, 50, 100, 150, 200, 250, 300]
for m in milestones:
    if m <= len(history):
        rec = history[m-1] 
        print(f"{rec['camera_count']:<10} | {rec['coverage_pct']:<6.2f}%         | +{rec['marginal_gain']:.0f}")

print("-" * 50)
print(f"FINAL: {len(selected_indices)} cams | {final_pct:.2f}% | {run_time:.2f}s")
print("="*50 + "\n")

# ==============================
# 7. SAVE RESULTS TO EXCEL
# ==============================
print(f"Saving to {OUTPUT_EXCEL_FILE}...")

# 1. Camera Solution
df_results_cam = df_candidates.iloc[selected_indices].copy()
df_results_cam = df_results_cam[['candidate_id', 'loc_id', 'x', 'y', 'dir_angle']]
df_results_cam.rename(columns={'dir_angle': 'direction_degrees'}, inplace=True)

# 2. Demand Coverage Status
selected_matrix = coverage_matrix[:, selected_indices]
coverage_counts_per_point = np.array(selected_matrix.sum(axis=1)).flatten()

df_results_dem = df_demand.copy()
df_results_dem['cameras_covering'] = coverage_counts_per_point
df_results_dem['is_covered'] = np.where(df_results_dem['cameras_covering'] > 0, 'Yes', 'No')

# 3. Growth History
df_history = pd.DataFrame(history)

with pd.ExcelWriter(OUTPUT_EXCEL_FILE) as writer:
    df_results_cam.to_excel(writer, sheet_name='Selected_Cameras', index=False)
    df_results_dem.to_excel(writer, sheet_name='Demand_Coverage', index=False)
    df_history.to_excel(writer, sheet_name='Growth_History', index=False)

print("Excel file created successfully.")

# ==============================
# 8. VISUALIZATION: GROWTH CURVE
# ==============================
plt.figure(figsize=(10, 6), dpi=100)
x_vals = [h['camera_count'] for h in history]
y_vals = [h['coverage_pct'] for h in history]

plt.plot(x_vals, y_vals, color='#2980b9', linewidth=2, label='Coverage %')

for m in milestones:
    if m <= len(history):
        val = history[m-1]['coverage_pct']
        plt.plot(m, val, 'o', color='#e74c3c')
        plt.annotate(f"{val:.1f}%", (m, val), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

plt.title(f"Coverage Growth (Solution 1)\nMax: {final_pct:.2f}%", fontsize=14, fontweight='bold')
plt.xlabel("Number of Cameras")
plt.ylabel("Coverage Percentage (%)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_CURVE_FILE)
print(f"Curve saved: {OUTPUT_CURVE_FILE}")

# ==============================
# 9. VISUALIZATION: HEATMAP
# ==============================
final_heatmap_grid = np.zeros((TARGET_ROWS, TARGET_COLS))
dem_x_flat = df_demand['x'].values
dem_y_flat = df_demand['y'].values

for i, count in enumerate(coverage_counts_per_point):
    if count > 0:
        r, c = int(dem_y_flat[i]), int(dem_x_flat[i])
        if 0 <= r < TARGET_ROWS and 0 <= c < TARGET_COLS:
            final_heatmap_grid[r, c] = count

aspect_ratio = TARGET_ROWS / TARGET_COLS
base_width = 10 
fig_height = base_width * aspect_ratio
fig, ax = plt.subplots(figsize=(base_width, fig_height), dpi=120)

ax.set_xlim(0, TARGET_COLS)
ax.set_ylim(TARGET_ROWS, 0) 
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_xticks(np.arange(0, TARGET_COLS + 1, 10))
ax.set_yticks(np.arange(0, TARGET_ROWS + 1, 10))
ax.set_xticks(np.arange(0.5, TARGET_COLS, 1), minor=True)
ax.set_yticks(np.arange(0.5, TARGET_ROWS, 1), minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
ax.tick_params(which='minor', bottom=False, left=False)

colors = ['#ffffff00', '#800080', '#00ced1', '#ffd700'] 
cmap = ListedColormap(colors)
bounds = [0, 1, 2, 3, 100]
norm = BoundaryNorm(bounds, cmap.N)

ax.imshow(final_heatmap_grid, extent=[0, TARGET_COLS, TARGET_ROWS, 0], cmap=cmap, norm=norm, interpolation='nearest', zorder=1)

legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#800080', markersize=10, label='1 Cam'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#00ced1', markersize=10, label='2 Cams'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#ffd700', markersize=10, label='3+ Cams')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

ax.set_title(f"{HEATMAP_TITLE}\nCoverage: {final_pct:.1f}%", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches='tight')
print(f"Heatmap saved: {OUTPUT_IMAGE_FILE}")























import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import sparse
from matplotlib.lines import Line2D
import time
import random

# ==============================
# 1. SETTINGS
# ==============================
INPUT_FILE = "new.xlsx"
DEMAND_SHEET = "demand"
CAMERA_SHEET = "camera"

# --- SPECIFIC OUTPUT FOR SOLUTION 1 ---
OUTPUT_IMAGE_FILE = "greedy_heatmap_soln3.png"
OUTPUT_CURVE_FILE = "greedy_growth_curve_soln3.png"
OUTPUT_EXCEL_FILE = "greedy_solution_final_soln3.xlsx"  # <--- As requested
HEATMAP_TITLE = "Solution 3: Randomized Greedy (300 Cameras)"

# Optimization Parameters
CAMERA_COUNT_P = 300  # <--- Run for exactly 300 cameras
CAMERA_RADIUS = 5
FOV_HALF_ANGLE = 45

# Randomization (RCL)
RCL_SIZE = 5
RANDOM_SEED = None  # Change to integer (e.g. 42) to fix the random choices

# Grid Settings
TARGET_COLS = 80
TARGET_ROWS = 130
SUBTRACT_ONE = 0 
BATCH_SIZE = 1000 

print(f"--- PROCESS STARTED: GENERATING {OUTPUT_EXCEL_FILE} ---")
start_time = time.time()

# ==============================
# 2. LOAD DATA
# ==============================
df_demand = pd.read_excel(INPUT_FILE, sheet_name=DEMAND_SHEET)
df_demand.columns = [c.strip().lower() for c in df_demand.columns]
df_demand = df_demand[df_demand['weight'] > 0].copy()
df_demand['x'] = df_demand['x'] - SUBTRACT_ONE
df_demand['y'] = df_demand['y'] - SUBTRACT_ONE

TOTAL_DEMAND_WEIGHT = df_demand['weight'].sum()

df_cam_locs = pd.read_excel(INPUT_FILE, sheet_name=CAMERA_SHEET)
df_cam_locs.columns = [c.strip().lower() for c in df_cam_locs.columns]
df_cam_locs['x'] = df_cam_locs['x'] - SUBTRACT_ONE
df_cam_locs['y'] = df_cam_locs['y'] - SUBTRACT_ONE

print(f"Demand Points: {len(df_demand)} | Total Weight: {TOTAL_DEMAND_WEIGHT:,.0f}")
print(f"Camera Locations: {len(df_cam_locs)}")

# ==============================
# 3. GENERATE CANDIDATES
# ==============================
candidates_list = []
directions = [0, 90, 180, 270]

for idx, row in df_cam_locs.iterrows():
    for angle in directions:
        candidates_list.append({
            'loc_id': idx,
            'x': row['x'],
            'y': row['y'],
            'dir_angle': angle
        })

df_candidates = pd.DataFrame(candidates_list)
df_candidates['candidate_id'] = range(len(df_candidates))
num_cands = len(df_candidates)

# ==============================
# 4. COMPUTE COVERAGE (BATCHED)
# ==============================
print("Computing Coverage Matrix...")
dem_x = df_demand['x'].values[:, np.newaxis]
dem_y = df_demand['y'].values[:, np.newaxis]
sparse_chunks = []

for start_idx in range(0, num_cands, BATCH_SIZE):
    end_idx = min(start_idx + BATCH_SIZE, num_cands)
    batch_df = df_candidates.iloc[start_idx:end_idx]
    
    cand_x = batch_df['x'].values[np.newaxis, :]
    cand_y = batch_df['y'].values[np.newaxis, :]
    cand_dirs = batch_df['dir_angle'].values[np.newaxis, :]
    
    dx = dem_x - cand_x
    dy = dem_y - cand_y
    distances = np.sqrt(dx**2 + dy**2)
    point_angles = np.degrees(np.arctan2(dx, -dy))
    point_angles = (point_angles + 360) % 360
    angle_diff = np.abs(point_angles - cand_dirs)
    angle_diff = np.minimum(angle_diff, 360 - angle_diff)
    
    mask = (distances <= CAMERA_RADIUS) & (angle_diff <= FOV_HALF_ANGLE) & (distances > 0)
    sparse_chunks.append(sparse.csc_matrix(mask))

coverage_matrix = sparse.hstack(sparse_chunks).tocsc()

# ==============================
# 5. RANDOMIZED GREEDY OPTIMIZATION
# ==============================
print(f"Optimizing for {CAMERA_COUNT_P} cameras using RCL={RCL_SIZE}...")

current_weights = df_demand['weight'].values.copy()
selected_indices = []

# List to track history for scenarios
history = []

for i in range(CAMERA_COUNT_P):
    # A. Calculate gains
    gains = coverage_matrix.T @ current_weights
    
    valid_gains_mask = gains > 0
    valid_indices = np.where(valid_gains_mask)[0]
    
    if len(valid_indices) == 0:
        print("No more gain possible (optimization stopped early).")
        break
        
    # B. RCL Selection (Top 5 Best)
    current_rcl_size = min(RCL_SIZE, len(valid_indices))
    # argpartition puts the top K elements at the end
    top_k_indices_unsorted = np.argpartition(gains[valid_indices], -current_rcl_size)[-current_rcl_size:]
    top_candidates = valid_indices[top_k_indices_unsorted]
    
    # C. Random Pick from Top K
    selected_idx = np.random.choice(top_candidates)
    selected_indices.append(selected_idx)
    
    # D. Update Weights
    covered_indices = coverage_matrix[:, selected_idx].indices
    current_weights[covered_indices] = 0
    
    # E. Record History
    current_covered_weight = TOTAL_DEMAND_WEIGHT - current_weights.sum()
    pct = (current_covered_weight / TOTAL_DEMAND_WEIGHT) * 100
    
    history.append({
        'camera_count': i + 1,
        'coverage_pct': pct,
        'marginal_gain': gains[selected_idx]
    })

# ==============================
# 6. SCENARIO REPORTING
# ==============================
end_time = time.time()
run_time = end_time - start_time
final_pct = history[-1]['coverage_pct']

print("\n" + "="*50)
print(f" SCENARIO ANALYSIS: COVERAGE FOR {OUTPUT_EXCEL_FILE}")
print("="*50)
print(f"{'Cameras':<10} | {'Coverage %':<15} | {'Gain':<10}")
print("-" * 50)

# Print specific milestones to understand growth
milestones = [10, 50, 100, 150, 200, 250, 300]
for m in milestones:
    if m <= len(history):
        rec = history[m-1] 
        print(f"{rec['camera_count']:<10} | {rec['coverage_pct']:<6.2f}%         | +{rec['marginal_gain']:.0f}")

print("-" * 50)
print(f"FINAL: {len(selected_indices)} cams | {final_pct:.2f}% | {run_time:.2f}s")
print("="*50 + "\n")

# ==============================
# 7. SAVE RESULTS TO EXCEL
# ==============================
print(f"Saving to {OUTPUT_EXCEL_FILE}...")

# 1. Camera Solution
df_results_cam = df_candidates.iloc[selected_indices].copy()
df_results_cam = df_results_cam[['candidate_id', 'loc_id', 'x', 'y', 'dir_angle']]
df_results_cam.rename(columns={'dir_angle': 'direction_degrees'}, inplace=True)

# 2. Demand Coverage Status
selected_matrix = coverage_matrix[:, selected_indices]
coverage_counts_per_point = np.array(selected_matrix.sum(axis=1)).flatten()

df_results_dem = df_demand.copy()
df_results_dem['cameras_covering'] = coverage_counts_per_point
df_results_dem['is_covered'] = np.where(df_results_dem['cameras_covering'] > 0, 'Yes', 'No')

# 3. Growth History
df_history = pd.DataFrame(history)

with pd.ExcelWriter(OUTPUT_EXCEL_FILE) as writer:
    df_results_cam.to_excel(writer, sheet_name='Selected_Cameras', index=False)
    df_results_dem.to_excel(writer, sheet_name='Demand_Coverage', index=False)
    df_history.to_excel(writer, sheet_name='Growth_History', index=False)

print("Excel file created successfully.")

# ==============================
# 8. VISUALIZATION: GROWTH CURVE
# ==============================
plt.figure(figsize=(10, 6), dpi=100)
x_vals = [h['camera_count'] for h in history]
y_vals = [h['coverage_pct'] for h in history]

plt.plot(x_vals, y_vals, color='#2980b9', linewidth=2, label='Coverage %')

for m in milestones:
    if m <= len(history):
        val = history[m-1]['coverage_pct']
        plt.plot(m, val, 'o', color='#e74c3c')
        plt.annotate(f"{val:.1f}%", (m, val), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

plt.title(f"Coverage Growth (Solution 1)\nMax: {final_pct:.2f}%", fontsize=14, fontweight='bold')
plt.xlabel("Number of Cameras")
plt.ylabel("Coverage Percentage (%)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_CURVE_FILE)
print(f"Curve saved: {OUTPUT_CURVE_FILE}")

# ==============================
# 9. VISUALIZATION: HEATMAP
# ==============================
final_heatmap_grid = np.zeros((TARGET_ROWS, TARGET_COLS))
dem_x_flat = df_demand['x'].values
dem_y_flat = df_demand['y'].values

for i, count in enumerate(coverage_counts_per_point):
    if count > 0:
        r, c = int(dem_y_flat[i]), int(dem_x_flat[i])
        if 0 <= r < TARGET_ROWS and 0 <= c < TARGET_COLS:
            final_heatmap_grid[r, c] = count

aspect_ratio = TARGET_ROWS / TARGET_COLS
base_width = 10 
fig_height = base_width * aspect_ratio
fig, ax = plt.subplots(figsize=(base_width, fig_height), dpi=120)

ax.set_xlim(0, TARGET_COLS)
ax.set_ylim(TARGET_ROWS, 0) 
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_xticks(np.arange(0, TARGET_COLS + 1, 10))
ax.set_yticks(np.arange(0, TARGET_ROWS + 1, 10))
ax.set_xticks(np.arange(0.5, TARGET_COLS, 1), minor=True)
ax.set_yticks(np.arange(0.5, TARGET_ROWS, 1), minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
ax.tick_params(which='minor', bottom=False, left=False)

colors = ['#ffffff00', '#800080', '#00ced1', '#ffd700'] 
cmap = ListedColormap(colors)
bounds = [0, 1, 2, 3, 100]
norm = BoundaryNorm(bounds, cmap.N)

ax.imshow(final_heatmap_grid, extent=[0, TARGET_COLS, TARGET_ROWS, 0], cmap=cmap, norm=norm, interpolation='nearest', zorder=1)

legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#800080', markersize=10, label='1 Cam'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#00ced1', markersize=10, label='2 Cams'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#ffd700', markersize=10, label='3+ Cams')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

ax.set_title(f"{HEATMAP_TITLE}\nCoverage: {final_pct:.1f}%", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches='tight')
print(f"Heatmap saved: {OUTPUT_IMAGE_FILE}")




















# ==============================
# 1. SETTINGS
# ==============================
INPUT_FILE = "new.xlsx"            # Source of Demand/Camera locations
DEMAND_SHEET = "demand"
CAMERA_SHEET = "camera"

# --- CRITICAL CHANGE: LOAD FROM YOUR SAVED SOLUTION FILE ---
# Change this filename to visualize different solutions (soln1, soln2, etc.)
EXTERNAL_SOLUTION_FILE = "greedy_solution_final_soln1.xlsx" 

OUTPUT_IMAGE_FILE = "greedy_heatmap_gridview_soln1.png"
HEATMAP_TITLE = "Solution 1 - Greedy Solution for Campus Heatmap (Grid View)"

# Optimization Parameters (Used for matrix generation only)
CAMERA_RADIUS = 5
FOV_HALF_ANGLE = 45

# Grid Settings
TARGET_COLS = 80
TARGET_ROWS = 130
SUBTRACT_ONE = 0 
BATCH_SIZE = 1000 

print(f"--- PROCESS STARTED: LOADING {EXTERNAL_SOLUTION_FILE} ---")
start_time = time.time()

# ==============================
# 2. LOAD DATA (Standard Setup)
# ==============================
df_demand = pd.read_excel(INPUT_FILE, sheet_name=DEMAND_SHEET)
df_demand.columns = [c.strip().lower() for c in df_demand.columns]
df_demand = df_demand[df_demand['weight'] > 0].copy()
df_demand['x'] = df_demand['x'] - SUBTRACT_ONE
df_demand['y'] = df_demand['y'] - SUBTRACT_ONE

TOTAL_DEMAND_WEIGHT = df_demand['weight'].sum()

df_cam_locs = pd.read_excel(INPUT_FILE, sheet_name=CAMERA_SHEET)
df_cam_locs.columns = [c.strip().lower() for c in df_cam_locs.columns]
df_cam_locs['x'] = df_cam_locs['x'] - SUBTRACT_ONE
df_cam_locs['y'] = df_cam_locs['y'] - SUBTRACT_ONE

print(f"Demand Points: {len(df_demand)} | Total Weight: {TOTAL_DEMAND_WEIGHT:,.0f}")

# ==============================
# 3. GENERATE CANDIDATES (Must match original generation order!)
# ==============================
# We need to regenerate the exact same list of candidates to map IDs correctly
candidates_list = []
directions = [0, 90, 180, 270]

for idx, row in df_cam_locs.iterrows():
    for angle in directions:
        candidates_list.append({
            'loc_id': idx,
            'x': row['x'],
            'y': row['y'],
            'dir_angle': angle
        })

df_candidates = pd.DataFrame(candidates_list)
df_candidates['candidate_id'] = range(len(df_candidates))
num_cands = len(df_candidates)

# ==============================
# 4. COMPUTE COVERAGE MATRIX
# ==============================
# We need the matrix to calculate what the loaded cameras actually cover
print("Recomputing Coverage Matrix to verify solution...")

dem_x = df_demand['x'].values[:, np.newaxis]
dem_y = df_demand['y'].values[:, np.newaxis]
sparse_chunks = []

for start_idx in range(0, num_cands, BATCH_SIZE):
    end_idx = min(start_idx + BATCH_SIZE, num_cands)
    
    batch_df = df_candidates.iloc[start_idx:end_idx]
    cand_x = batch_df['x'].values[np.newaxis, :]
    cand_y = batch_df['y'].values[np.newaxis, :]
    cand_dirs = batch_df['dir_angle'].values[np.newaxis, :]
    
    dx = dem_x - cand_x
    dy = dem_y - cand_y
    distances = np.sqrt(dx**2 + dy**2)
    
    point_angles = np.degrees(np.arctan2(dx, -dy))
    point_angles = (point_angles + 360) % 360
    
    angle_diff = np.abs(point_angles - cand_dirs)
    angle_diff = np.minimum(angle_diff, 360 - angle_diff)
    
    mask = (distances <= CAMERA_RADIUS) & (angle_diff <= FOV_HALF_ANGLE) & (distances > 0)
    sparse_chunks.append(sparse.csc_matrix(mask))

coverage_matrix = sparse.hstack(sparse_chunks).tocsc()

# ==============================
# 5. LOAD EXTERNAL SOLUTION
# ==============================
print(f"Loading selected cameras from: {EXTERNAL_SOLUTION_FILE}")

try:
    # Read the file produced by your previous runs
    df_loaded_sol = pd.read_excel(EXTERNAL_SOLUTION_FILE, sheet_name='Selected_Cameras')
    
    # Extract the 'candidate_id' column
    # These IDs correspond to the rows in our df_candidates / columns in coverage_matrix
    selected_indices = df_loaded_sol['candidate_id'].values
    
    num_selected = len(selected_indices)
    print(f"Successfully loaded {num_selected} cameras.")
    
except Exception as e:
    print(f"ERROR: Could not read solution file. \n{e}")
    print("Ensure the file exists and has a sheet named 'Selected_Cameras' with a 'candidate_id' column.")
    exit()

# ==============================
# 6. CALCULATE STATISTICS
# ==============================
# Slice the matrix to get only the selected cameras
selected_matrix = coverage_matrix[:, selected_indices]

# Sum across rows to see how many cameras cover each demand point
coverage_counts_per_point = np.array(selected_matrix.sum(axis=1)).flatten()

# Calculate Covered Weight
# Demand points where coverage_counts > 0 are covered
is_covered = coverage_counts_per_point > 0
covered_weight = df_demand.loc[is_covered, 'weight'].sum()
coverage_pct = (covered_weight / TOTAL_DEMAND_WEIGHT) * 100

run_time = time.time() - start_time

print("-" * 40)
print(f"LOADED SOLUTION METRICS")
print(f"CAMERAS:  {num_selected}")
print(f"COVERAGE: {covered_weight:,.0f} / {TOTAL_DEMAND_WEIGHT:,.0f} ({coverage_pct:.2f}%)")
print("-" * 40)

# ==============================
# 7. VISUALIZATION (GRID VIEW)
# ==============================
dem_x = df_demand['x'].values
dem_y = df_demand['y'].values

final_heatmap_grid = np.zeros((TARGET_ROWS, TARGET_COLS))

for i, count in enumerate(coverage_counts_per_point):
    if count > 0:
        r, c = int(dem_y[i]), int(dem_x[i])
        if 0 <= r < TARGET_ROWS and 0 <= c < TARGET_COLS:
            final_heatmap_grid[r, c] = count

# Plot Setup
aspect_ratio = TARGET_ROWS / TARGET_COLS
base_width = 10 
fig_height = base_width * aspect_ratio

fig, ax = plt.subplots(figsize=(base_width, fig_height), dpi=120)

# Configure Axes
ax.set_xlim(0, TARGET_COLS)
ax.set_ylim(TARGET_ROWS, 0) # Inverted Y
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

ax.set_xticks(np.arange(0, TARGET_COLS + 1, 10))
ax.set_yticks(np.arange(0, TARGET_ROWS + 1, 10))
ax.set_xlabel("Column Index (X)")
ax.set_ylabel("Row Index (Y)")

# Grid Lines
ax.set_xticks(np.arange(0.5, TARGET_COLS, 1), minor=True)
ax.set_yticks(np.arange(0.5, TARGET_ROWS, 1), minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
ax.tick_params(which='minor', bottom=False, left=False)

# Colors
colors = ['#ffffff00', '#800080', '#00ced1', '#ffd700'] 
cmap = ListedColormap(colors)
bounds = [0, 1, 2, 3, 100]
norm = BoundaryNorm(bounds, cmap.N)

# Display Heatmap
ax.imshow(
    final_heatmap_grid,
    extent=[0, TARGET_COLS, TARGET_ROWS, 0],
    cmap=cmap,
    norm=norm,
    interpolation='nearest',
    zorder=1
)

# Legend
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#800080', markersize=10, label='1 Cam'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#00ced1', markersize=10, label='2 Cams'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#ffd700', markersize=10, label='3+ Cams')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

# Title
title_text = f"{HEATMAP_TITLE}\nCoverage: {coverage_pct:.1f}% | Cams: {num_selected}"
ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches='tight')
print(f"Heatmap saved: {OUTPUT_IMAGE_FILE}")
















# ==============================
# 1. SETTINGS
# ==============================
INPUT_FILE = "new.xlsx"            # Source of Demand/Camera locations
DEMAND_SHEET = "demand"
CAMERA_SHEET = "camera"

# --- CRITICAL CHANGE: LOAD FROM YOUR SAVED SOLUTION FILE ---
# Change this filename to visualize different solutions (soln1, soln2, etc.)
EXTERNAL_SOLUTION_FILE = "greedy_solution_final_soln2.xlsx" 

OUTPUT_IMAGE_FILE = "greedy_heatmap_gridview_soln2.png"
HEATMAP_TITLE = "Solution 2 - Greedy Solution for Campus Heatmap (Grid View)"

# Optimization Parameters (Used for matrix generation only)
CAMERA_RADIUS = 5
FOV_HALF_ANGLE = 45

# Grid Settings
TARGET_COLS = 80
TARGET_ROWS = 130
SUBTRACT_ONE = 0 
BATCH_SIZE = 1000 

print(f"--- PROCESS STARTED: LOADING {EXTERNAL_SOLUTION_FILE} ---")
start_time = time.time()

# ==============================
# 2. LOAD DATA (Standard Setup)
# ==============================
df_demand = pd.read_excel(INPUT_FILE, sheet_name=DEMAND_SHEET)
df_demand.columns = [c.strip().lower() for c in df_demand.columns]
df_demand = df_demand[df_demand['weight'] > 0].copy()
df_demand['x'] = df_demand['x'] - SUBTRACT_ONE
df_demand['y'] = df_demand['y'] - SUBTRACT_ONE

TOTAL_DEMAND_WEIGHT = df_demand['weight'].sum()

df_cam_locs = pd.read_excel(INPUT_FILE, sheet_name=CAMERA_SHEET)
df_cam_locs.columns = [c.strip().lower() for c in df_cam_locs.columns]
df_cam_locs['x'] = df_cam_locs['x'] - SUBTRACT_ONE
df_cam_locs['y'] = df_cam_locs['y'] - SUBTRACT_ONE

print(f"Demand Points: {len(df_demand)} | Total Weight: {TOTAL_DEMAND_WEIGHT:,.0f}")

# ==============================
# 3. GENERATE CANDIDATES (Must match original generation order!)
# ==============================
# We need to regenerate the exact same list of candidates to map IDs correctly
candidates_list = []
directions = [0, 90, 180, 270]

for idx, row in df_cam_locs.iterrows():
    for angle in directions:
        candidates_list.append({
            'loc_id': idx,
            'x': row['x'],
            'y': row['y'],
            'dir_angle': angle
        })

df_candidates = pd.DataFrame(candidates_list)
df_candidates['candidate_id'] = range(len(df_candidates))
num_cands = len(df_candidates)

# ==============================
# 4. COMPUTE COVERAGE MATRIX
# ==============================
# We need the matrix to calculate what the loaded cameras actually cover
print("Recomputing Coverage Matrix to verify solution...")

dem_x = df_demand['x'].values[:, np.newaxis]
dem_y = df_demand['y'].values[:, np.newaxis]
sparse_chunks = []

for start_idx in range(0, num_cands, BATCH_SIZE):
    end_idx = min(start_idx + BATCH_SIZE, num_cands)
    
    batch_df = df_candidates.iloc[start_idx:end_idx]
    cand_x = batch_df['x'].values[np.newaxis, :]
    cand_y = batch_df['y'].values[np.newaxis, :]
    cand_dirs = batch_df['dir_angle'].values[np.newaxis, :]
    
    dx = dem_x - cand_x
    dy = dem_y - cand_y
    distances = np.sqrt(dx**2 + dy**2)
    
    point_angles = np.degrees(np.arctan2(dx, -dy))
    point_angles = (point_angles + 360) % 360
    
    angle_diff = np.abs(point_angles - cand_dirs)
    angle_diff = np.minimum(angle_diff, 360 - angle_diff)
    
    mask = (distances <= CAMERA_RADIUS) & (angle_diff <= FOV_HALF_ANGLE) & (distances > 0)
    sparse_chunks.append(sparse.csc_matrix(mask))

coverage_matrix = sparse.hstack(sparse_chunks).tocsc()

# ==============================
# 5. LOAD EXTERNAL SOLUTION
# ==============================
print(f"Loading selected cameras from: {EXTERNAL_SOLUTION_FILE}")

try:
    # Read the file produced by your previous runs
    df_loaded_sol = pd.read_excel(EXTERNAL_SOLUTION_FILE, sheet_name='Selected_Cameras')
    
    # Extract the 'candidate_id' column
    # These IDs correspond to the rows in our df_candidates / columns in coverage_matrix
    selected_indices = df_loaded_sol['candidate_id'].values
    
    num_selected = len(selected_indices)
    print(f"Successfully loaded {num_selected} cameras.")
    
except Exception as e:
    print(f"ERROR: Could not read solution file. \n{e}")
    print("Ensure the file exists and has a sheet named 'Selected_Cameras' with a 'candidate_id' column.")
    exit()

# ==============================
# 6. CALCULATE STATISTICS
# ==============================
# Slice the matrix to get only the selected cameras
selected_matrix = coverage_matrix[:, selected_indices]

# Sum across rows to see how many cameras cover each demand point
coverage_counts_per_point = np.array(selected_matrix.sum(axis=1)).flatten()

# Calculate Covered Weight
# Demand points where coverage_counts > 0 are covered
is_covered = coverage_counts_per_point > 0
covered_weight = df_demand.loc[is_covered, 'weight'].sum()
coverage_pct = (covered_weight / TOTAL_DEMAND_WEIGHT) * 100

run_time = time.time() - start_time

print("-" * 40)
print(f"LOADED SOLUTION METRICS")
print(f"CAMERAS:  {num_selected}")
print(f"COVERAGE: {covered_weight:,.0f} / {TOTAL_DEMAND_WEIGHT:,.0f} ({coverage_pct:.2f}%)")
print("-" * 40)

# ==============================
# 7. VISUALIZATION (GRID VIEW)
# ==============================
dem_x = df_demand['x'].values
dem_y = df_demand['y'].values

final_heatmap_grid = np.zeros((TARGET_ROWS, TARGET_COLS))

for i, count in enumerate(coverage_counts_per_point):
    if count > 0:
        r, c = int(dem_y[i]), int(dem_x[i])
        if 0 <= r < TARGET_ROWS and 0 <= c < TARGET_COLS:
            final_heatmap_grid[r, c] = count

# Plot Setup
aspect_ratio = TARGET_ROWS / TARGET_COLS
base_width = 10 
fig_height = base_width * aspect_ratio

fig, ax = plt.subplots(figsize=(base_width, fig_height), dpi=120)

# Configure Axes
ax.set_xlim(0, TARGET_COLS)
ax.set_ylim(TARGET_ROWS, 0) # Inverted Y
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

ax.set_xticks(np.arange(0, TARGET_COLS + 1, 10))
ax.set_yticks(np.arange(0, TARGET_ROWS + 1, 10))
ax.set_xlabel("Column Index (X)")
ax.set_ylabel("Row Index (Y)")

# Grid Lines
ax.set_xticks(np.arange(0.5, TARGET_COLS, 1), minor=True)
ax.set_yticks(np.arange(0.5, TARGET_ROWS, 1), minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
ax.tick_params(which='minor', bottom=False, left=False)

# Colors
colors = ['#ffffff00', '#800080', '#00ced1', '#ffd700'] 
cmap = ListedColormap(colors)
bounds = [0, 1, 2, 3, 100]
norm = BoundaryNorm(bounds, cmap.N)

# Display Heatmap
ax.imshow(
    final_heatmap_grid,
    extent=[0, TARGET_COLS, TARGET_ROWS, 0],
    cmap=cmap,
    norm=norm,
    interpolation='nearest',
    zorder=1
)

# Legend
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#800080', markersize=10, label='1 Cam'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#00ced1', markersize=10, label='2 Cams'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#ffd700', markersize=10, label='3+ Cams')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

# Title
title_text = f"{HEATMAP_TITLE}\nCoverage: {coverage_pct:.1f}% | Cams: {num_selected}"
ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches='tight')
print(f"Heatmap saved: {OUTPUT_IMAGE_FILE}")






















# ==============================
# 1. SETTINGS
# ==============================
INPUT_FILE = "new.xlsx"            # Source of Demand/Camera locations
DEMAND_SHEET = "demand"
CAMERA_SHEET = "camera"

# --- CRITICAL CHANGE: LOAD FROM YOUR SAVED SOLUTION FILE ---
# Change this filename to visualize different solutions (soln1, soln2, etc.)
EXTERNAL_SOLUTION_FILE = "greedy_solution_final_soln3.xlsx" 

OUTPUT_IMAGE_FILE = "greedy_heatmap_gridview_soln3.png"
HEATMAP_TITLE = "Solution 3 - Greedy Solution for Campus Heatmap (Grid View)"

# Optimization Parameters (Used for matrix generation only)
CAMERA_RADIUS = 5
FOV_HALF_ANGLE = 45

# Grid Settings
TARGET_COLS = 80
TARGET_ROWS = 130
SUBTRACT_ONE = 0 
BATCH_SIZE = 1000 

print(f"--- PROCESS STARTED: LOADING {EXTERNAL_SOLUTION_FILE} ---")
start_time = time.time()

# ==============================
# 2. LOAD DATA (Standard Setup)
# ==============================
df_demand = pd.read_excel(INPUT_FILE, sheet_name=DEMAND_SHEET)
df_demand.columns = [c.strip().lower() for c in df_demand.columns]
df_demand = df_demand[df_demand['weight'] > 0].copy()
df_demand['x'] = df_demand['x'] - SUBTRACT_ONE
df_demand['y'] = df_demand['y'] - SUBTRACT_ONE

TOTAL_DEMAND_WEIGHT = df_demand['weight'].sum()

df_cam_locs = pd.read_excel(INPUT_FILE, sheet_name=CAMERA_SHEET)
df_cam_locs.columns = [c.strip().lower() for c in df_cam_locs.columns]
df_cam_locs['x'] = df_cam_locs['x'] - SUBTRACT_ONE
df_cam_locs['y'] = df_cam_locs['y'] - SUBTRACT_ONE

print(f"Demand Points: {len(df_demand)} | Total Weight: {TOTAL_DEMAND_WEIGHT:,.0f}")

# ==============================
# 3. GENERATE CANDIDATES (Must match original generation order!)
# ==============================
# We need to regenerate the exact same list of candidates to map IDs correctly
candidates_list = []
directions = [0, 90, 180, 270]

for idx, row in df_cam_locs.iterrows():
    for angle in directions:
        candidates_list.append({
            'loc_id': idx,
            'x': row['x'],
            'y': row['y'],
            'dir_angle': angle
        })

df_candidates = pd.DataFrame(candidates_list)
df_candidates['candidate_id'] = range(len(df_candidates))
num_cands = len(df_candidates)

# ==============================
# 4. COMPUTE COVERAGE MATRIX
# ==============================
# We need the matrix to calculate what the loaded cameras actually cover
print("Recomputing Coverage Matrix to verify solution...")

dem_x = df_demand['x'].values[:, np.newaxis]
dem_y = df_demand['y'].values[:, np.newaxis]
sparse_chunks = []

for start_idx in range(0, num_cands, BATCH_SIZE):
    end_idx = min(start_idx + BATCH_SIZE, num_cands)
    
    batch_df = df_candidates.iloc[start_idx:end_idx]
    cand_x = batch_df['x'].values[np.newaxis, :]
    cand_y = batch_df['y'].values[np.newaxis, :]
    cand_dirs = batch_df['dir_angle'].values[np.newaxis, :]
    
    dx = dem_x - cand_x
    dy = dem_y - cand_y
    distances = np.sqrt(dx**2 + dy**2)
    
    point_angles = np.degrees(np.arctan2(dx, -dy))
    point_angles = (point_angles + 360) % 360
    
    angle_diff = np.abs(point_angles - cand_dirs)
    angle_diff = np.minimum(angle_diff, 360 - angle_diff)
    
    mask = (distances <= CAMERA_RADIUS) & (angle_diff <= FOV_HALF_ANGLE) & (distances > 0)
    sparse_chunks.append(sparse.csc_matrix(mask))

coverage_matrix = sparse.hstack(sparse_chunks).tocsc()

# ==============================
# 5. LOAD EXTERNAL SOLUTION
# ==============================
print(f"Loading selected cameras from: {EXTERNAL_SOLUTION_FILE}")

try:
    # Read the file produced by your previous runs
    df_loaded_sol = pd.read_excel(EXTERNAL_SOLUTION_FILE, sheet_name='Selected_Cameras')
    
    # Extract the 'candidate_id' column
    # These IDs correspond to the rows in our df_candidates / columns in coverage_matrix
    selected_indices = df_loaded_sol['candidate_id'].values
    
    num_selected = len(selected_indices)
    print(f"Successfully loaded {num_selected} cameras.")
    
except Exception as e:
    print(f"ERROR: Could not read solution file. \n{e}")
    print("Ensure the file exists and has a sheet named 'Selected_Cameras' with a 'candidate_id' column.")
    exit()

# ==============================
# 6. CALCULATE STATISTICS
# ==============================
# Slice the matrix to get only the selected cameras
selected_matrix = coverage_matrix[:, selected_indices]

# Sum across rows to see how many cameras cover each demand point
coverage_counts_per_point = np.array(selected_matrix.sum(axis=1)).flatten()

# Calculate Covered Weight
# Demand points where coverage_counts > 0 are covered
is_covered = coverage_counts_per_point > 0
covered_weight = df_demand.loc[is_covered, 'weight'].sum()
coverage_pct = (covered_weight / TOTAL_DEMAND_WEIGHT) * 100

run_time = time.time() - start_time

print("-" * 40)
print(f"LOADED SOLUTION METRICS")
print(f"CAMERAS:  {num_selected}")
print(f"COVERAGE: {covered_weight:,.0f} / {TOTAL_DEMAND_WEIGHT:,.0f} ({coverage_pct:.2f}%)")
print("-" * 40)

# ==============================
# 7. VISUALIZATION (GRID VIEW)
# ==============================
dem_x = df_demand['x'].values
dem_y = df_demand['y'].values

final_heatmap_grid = np.zeros((TARGET_ROWS, TARGET_COLS))

for i, count in enumerate(coverage_counts_per_point):
    if count > 0:
        r, c = int(dem_y[i]), int(dem_x[i])
        if 0 <= r < TARGET_ROWS and 0 <= c < TARGET_COLS:
            final_heatmap_grid[r, c] = count

# Plot Setup
aspect_ratio = TARGET_ROWS / TARGET_COLS
base_width = 10 
fig_height = base_width * aspect_ratio

fig, ax = plt.subplots(figsize=(base_width, fig_height), dpi=120)

# Configure Axes
ax.set_xlim(0, TARGET_COLS)
ax.set_ylim(TARGET_ROWS, 0) # Inverted Y
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

ax.set_xticks(np.arange(0, TARGET_COLS + 1, 10))
ax.set_yticks(np.arange(0, TARGET_ROWS + 1, 10))
ax.set_xlabel("Column Index (X)")
ax.set_ylabel("Row Index (Y)")

# Grid Lines
ax.set_xticks(np.arange(0.5, TARGET_COLS, 1), minor=True)
ax.set_yticks(np.arange(0.5, TARGET_ROWS, 1), minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
ax.tick_params(which='minor', bottom=False, left=False)

# Colors
colors = ['#ffffff00', '#800080', '#00ced1', '#ffd700'] 
cmap = ListedColormap(colors)
bounds = [0, 1, 2, 3, 100]
norm = BoundaryNorm(bounds, cmap.N)

# Display Heatmap
ax.imshow(
    final_heatmap_grid,
    extent=[0, TARGET_COLS, TARGET_ROWS, 0],
    cmap=cmap,
    norm=norm,
    interpolation='nearest',
    zorder=1
)

# Legend
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#800080', markersize=10, label='1 Cam'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#00ced1', markersize=10, label='2 Cams'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#ffd700', markersize=10, label='3+ Cams')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

# Title
title_text = f"{HEATMAP_TITLE}\nCoverage: {coverage_pct:.1f}% | Cams: {num_selected}"
ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches='tight')
print(f"Heatmap saved: {OUTPUT_IMAGE_FILE}")



























import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import sparse
from matplotlib.lines import Line2D
import time
import random

# ==============================
# 1. SETTINGS
# ==============================
INPUT_FILE = "new.xlsx"
INITIAL_SOLUTION_FILE = "greedy_solution_final_soln1.xlsx" 

DEMAND_SHEET = "demand"
CAMERA_SHEET = "camera"

OUTPUT_IMAGE_FILE = "local_search_heatmap_soln1.png"
OUTPUT_PLOT_FILE = "local_search_plot_soln1.png"
OUTPUT_EXCEL_FILE = "local_search_soln1.xlsx"
HEATMAP_TITLE = "Solution 1: 1-Swap Local Search"

# Optimization Parameters
CAMERA_RADIUS = 5
FOV_HALF_ANGLE = 45

# Threshold to prevent swapping for negligible/zero gains
MIN_IMPROVEMENT_THRESHOLD = 0.01 

# Grid Settings
TARGET_COLS = 80
TARGET_ROWS = 130
SUBTRACT_ONE = 0 
BATCH_SIZE = 1000 

print("--- PROCESS STARTED ---")
start_time = time.time()

# ==============================
# 2. LOAD DATA
# ==============================
df_demand = pd.read_excel(INPUT_FILE, sheet_name=DEMAND_SHEET)
df_demand.columns = [c.strip().lower() for c in df_demand.columns]
df_demand = df_demand[df_demand['weight'] > 0].copy()
df_demand['x'] = df_demand['x'] - SUBTRACT_ONE
df_demand['y'] = df_demand['y'] - SUBTRACT_ONE

TOTAL_DEMAND_WEIGHT = df_demand['weight'].sum()
demand_weights = df_demand['weight'].values

df_cam_locs = pd.read_excel(INPUT_FILE, sheet_name=CAMERA_SHEET)
df_cam_locs.columns = [c.strip().lower() for c in df_cam_locs.columns]
df_cam_locs['x'] = df_cam_locs['x'] - SUBTRACT_ONE
df_cam_locs['y'] = df_cam_locs['y'] - SUBTRACT_ONE

print(f"Demand Points: {len(df_demand)} | Total Weight: {TOTAL_DEMAND_WEIGHT:,.4f}")

# ==============================
# 3. GENERATE CANDIDATES
# ==============================
candidates_list = []
directions = [0, 90, 180, 270]

for idx, row in df_cam_locs.iterrows():
    for angle in directions:
        candidates_list.append({
            'loc_id': idx,
            'x': row['x'],
            'y': row['y'],
            'dir_angle': angle
        })

df_candidates = pd.DataFrame(candidates_list)
df_candidates['candidate_id'] = range(len(df_candidates))
num_cands = len(df_candidates)
print(f"Total Candidates Generated: {num_cands}")

# ==============================
# 4. COMPUTE COVERAGE MATRIX
# ==============================
print("Computing Coverage Matrix...")
dem_x = df_demand['x'].values[:, np.newaxis]
dem_y = df_demand['y'].values[:, np.newaxis]
sparse_chunks = []

for start_idx in range(0, num_cands, BATCH_SIZE):
    end_idx = min(start_idx + BATCH_SIZE, num_cands)
    batch_df = df_candidates.iloc[start_idx:end_idx]
    cand_x = batch_df['x'].values[np.newaxis, :]
    cand_y = batch_df['y'].values[np.newaxis, :]
    cand_dirs = batch_df['dir_angle'].values[np.newaxis, :]
    
    dx = dem_x - cand_x
    dy = dem_y - cand_y
    distances = np.sqrt(dx**2 + dy**2)
    point_angles = np.degrees(np.arctan2(dx, -dy))
    point_angles = (point_angles + 360) % 360
    angle_diff = np.abs(point_angles - cand_dirs)
    angle_diff = np.minimum(angle_diff, 360 - angle_diff)
    
    mask = (distances <= CAMERA_RADIUS) & (angle_diff <= FOV_HALF_ANGLE) & (distances > 0)
    sparse_chunks.append(sparse.csc_matrix(mask))

coverage_matrix = sparse.hstack(sparse_chunks).tocsc()

# ==============================
# 5. LOAD INITIAL SOLUTION
# ==============================
print(f"--- Loading Initial Solution from {INITIAL_SOLUTION_FILE} ---")
try:
    df_initial = pd.read_excel(INITIAL_SOLUTION_FILE, sheet_name='Selected_Cameras')
    initial_indices = df_initial['candidate_id'].values.tolist()
    current_solution_set = set(initial_indices)
    print(f"Loaded {len(current_solution_set)} cameras from file.")
except Exception as e:
    print(f"ERROR: {e}")
    exit()

# Reconstruct State
print("Reconstructing coverage state...")
selected_matrix = coverage_matrix[:, initial_indices]
coverage_counts = np.array(selected_matrix.sum(axis=1)).flatten()

current_covered_mask = coverage_counts > 0
initial_covered_weight = demand_weights[current_covered_mask].sum()
initial_pct = (initial_covered_weight / TOTAL_DEMAND_WEIGHT) * 100

print(f"Initial Z: {initial_covered_weight:,.4f} ({initial_pct:.2f}%)")

# ==============================
# 6. 1-SWAP LOCAL SEARCH
# ==============================
print(f"\n--- Starting 1-Swap Local Search ---")

best_solution = set(current_solution_set)
best_coverage_val = initial_covered_weight
improved = True
iteration = 0

# Lists to store history for plotting
history_iterations = [0]
history_obj_values = [best_coverage_val]

while improved:
    improved = False
    iteration += 1
    # print(f"  > Iteration {iteration} scanning...")
    
    current_sel = list(best_solution)
    current_unsel = list(set(range(num_cands)) - best_solution)
    random.shuffle(current_unsel) 
    
    swap_found = False
    
    for cam_out in current_sel:
        # LOSS
        pts_covered_by_out = coverage_matrix[:, cam_out].indices
        uniquely_covered_mask = coverage_counts[pts_covered_by_out] == 1
        points_losing_coverage = pts_covered_by_out[uniquely_covered_mask]
        loss_val = demand_weights[points_losing_coverage].sum()
        
        for cam_in in current_unsel:
            # GAIN
            pts_covered_by_in = coverage_matrix[:, cam_in].indices
            currently_uncovered_mask = coverage_counts[pts_covered_by_in] == 0
            points_gaining_coverage = pts_covered_by_in[currently_uncovered_mask]
            gain_val = demand_weights[points_gaining_coverage].sum()
            
            # NET IMPROVEMENT
            net_improvement = gain_val - loss_val
            
            if net_improvement > MIN_IMPROVEMENT_THRESHOLD:
                
                best_solution.remove(cam_out)
                best_solution.add(cam_in)
                best_coverage_val += net_improvement
                
                coverage_counts[pts_covered_by_out] -= 1
                coverage_counts[pts_covered_by_in] += 1
                
                print(f"  It {iteration}: Out {cam_out} / In {cam_in} | Gain +{net_improvement:.4f} | Z: {best_coverage_val:,.4f}")
                
                history_iterations.append(iteration)
                history_obj_values.append(best_coverage_val)
                
                swap_found = True
                improved = True
                break 
        
        if swap_found:
            break
            
    if not swap_found:
        print("  No further improvements found. Local Optimum reached.")

# ==============================
# 7. FINAL METRICS
# ==============================
final_pct = (best_coverage_val / TOTAL_DEMAND_WEIGHT) * 100
end_time = time.time()
run_time_seconds = end_time - start_time

print("-" * 40)
print(f"INITIAL Z:      {initial_covered_weight:,.4f} ({initial_pct:.2f}%)")
print(f"FINAL Z:        {best_coverage_val:,.4f} ({final_pct:.2f}%)")
print(f"IMPROVEMENT:    +{best_coverage_val - initial_covered_weight:,.4f}")
print(f"TOTAL RUN TIME: {run_time_seconds:.2f} seconds")
print("-" * 40)

# ==============================
# 8. EXCEL SAVE
# ==============================
final_indices = list(best_solution)
df_results_cam = df_candidates.iloc[final_indices].copy()
df_results_cam = df_results_cam[['candidate_id', 'loc_id', 'x', 'y', 'dir_angle']]
df_results_cam.rename(columns={'dir_angle': 'direction_degrees'}, inplace=True)

df_results_dem = df_demand.copy()
df_results_dem['cameras_covering'] = coverage_counts 
df_results_dem['is_covered'] = np.where(df_results_dem['cameras_covering'] > 0, 'Yes', 'No')

with pd.ExcelWriter(OUTPUT_EXCEL_FILE) as writer:
    df_results_cam.to_excel(writer, sheet_name='Selected_Cameras', index=False)
    df_results_dem.to_excel(writer, sheet_name='Demand_Coverage', index=False)
print(f"Excel saved: {OUTPUT_EXCEL_FILE}")

# ==============================
# 9. HEATMAP VISUALIZATION
# ==============================
final_heatmap_grid = np.zeros((TARGET_ROWS, TARGET_COLS))
dem_x_flat = df_demand['x'].values
dem_y_flat = df_demand['y'].values

for i, count in enumerate(coverage_counts):
    if count > 0:
        r, c = int(dem_y_flat[i]), int(dem_x_flat[i])
        if 0 <= r < TARGET_ROWS and 0 <= c < TARGET_COLS:
            final_heatmap_grid[r, c] = count

aspect_ratio = TARGET_ROWS / TARGET_COLS
base_width = 10 
fig_height = base_width * aspect_ratio
fig, ax = plt.subplots(figsize=(base_width, fig_height), dpi=120)

ax.set_xlim(0, TARGET_COLS)
ax.set_ylim(TARGET_ROWS, 0) 
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_xticks(np.arange(0, TARGET_COLS + 1, 10))
ax.set_yticks(np.arange(0, TARGET_ROWS + 1, 10))
ax.set_xticks(np.arange(0.5, TARGET_COLS, 1), minor=True)
ax.set_yticks(np.arange(0.5, TARGET_ROWS, 1), minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
ax.tick_params(which='minor', bottom=False, left=False)

colors = ['#ffffff00', '#800080', '#00ced1', '#ffd700'] 
cmap = ListedColormap(colors)
bounds = [0, 1, 2, 3, 100]
norm = BoundaryNorm(bounds, cmap.N)
ax.imshow(final_heatmap_grid, extent=[0, TARGET_COLS, TARGET_ROWS, 0], cmap=cmap, norm=norm, interpolation='nearest', zorder=1)

legend_elements = [Line2D([0], [0], marker='s', color='w', markerfacecolor='#800080', markersize=10, label='1 Cam'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='#00ced1', markersize=10, label='2 Cams'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='#ffd700', markersize=10, label='3+ Cams')]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

# Update Title to include Coverage and Run Time
title_text = f"{HEATMAP_TITLE}\nCoverage: {final_pct:.2f}% | Run Time: {run_time_seconds:.1f}s"
ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches='tight')
plt.close()
print(f"Heatmap saved: {OUTPUT_IMAGE_FILE}")

# ==============================
# 10. PLOT OPTIMIZATION TRAJECTORY
# ==============================
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(history_iterations, history_obj_values, marker='o', linestyle='-', color='#e74c3c', markersize=4, label='Obj Function (Z)')

plt.annotate(f"Start: {history_obj_values[0]:,.0f}", (history_iterations[0], history_obj_values[0]), 
             textcoords="offset points", xytext=(10, 0), ha='left', fontsize=9)
plt.annotate(f"End: {history_obj_values[-1]:,.0f}", (history_iterations[-1], history_obj_values[-1]), 
             textcoords="offset points", xytext=(-10, 10), ha='right', fontsize=9)

# Update Title to include Coverage and Run Time
plot_title = f"Local Search Trajectory\nFinal Coverage: {final_pct:.2f}% | Time: {run_time_seconds:.1f}s"
plt.title(plot_title, fontsize=14, fontweight='bold')
plt.xlabel("Iteration Number")
plt.ylabel("Total Weighted Coverage (Z)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PLOT_FILE)
plt.close()

print(f"Trajectory plot saved: {OUTPUT_PLOT_FILE}")




















# ==============================
# 1. SETTINGS
# ==============================
INPUT_FILE = "new.xlsx"
INITIAL_SOLUTION_FILE = "greedy_solution_final_soln2.xlsx" 

DEMAND_SHEET = "demand"
CAMERA_SHEET = "camera"

OUTPUT_IMAGE_FILE = "local_search_heatmap_soln2.png"
OUTPUT_PLOT_FILE = "local_search_plot_soln2.png"
OUTPUT_EXCEL_FILE = "local_search_soln2.xlsx"
HEATMAP_TITLE = "Solution 2: 1-Swap Local Search"

# Optimization Parameters
CAMERA_RADIUS = 5
FOV_HALF_ANGLE = 45

# Threshold to prevent swapping for negligible/zero gains
MIN_IMPROVEMENT_THRESHOLD = 0.01 

# Grid Settings
TARGET_COLS = 80
TARGET_ROWS = 130
SUBTRACT_ONE = 0 
BATCH_SIZE = 1000 

print("--- PROCESS STARTED ---")
start_time = time.time()

# ==============================
# 2. LOAD DATA
# ==============================
df_demand = pd.read_excel(INPUT_FILE, sheet_name=DEMAND_SHEET)
df_demand.columns = [c.strip().lower() for c in df_demand.columns]
df_demand = df_demand[df_demand['weight'] > 0].copy()
df_demand['x'] = df_demand['x'] - SUBTRACT_ONE
df_demand['y'] = df_demand['y'] - SUBTRACT_ONE

TOTAL_DEMAND_WEIGHT = df_demand['weight'].sum()
demand_weights = df_demand['weight'].values

df_cam_locs = pd.read_excel(INPUT_FILE, sheet_name=CAMERA_SHEET)
df_cam_locs.columns = [c.strip().lower() for c in df_cam_locs.columns]
df_cam_locs['x'] = df_cam_locs['x'] - SUBTRACT_ONE
df_cam_locs['y'] = df_cam_locs['y'] - SUBTRACT_ONE

print(f"Demand Points: {len(df_demand)} | Total Weight: {TOTAL_DEMAND_WEIGHT:,.4f}")

# ==============================
# 3. GENERATE CANDIDATES
# ==============================
candidates_list = []
directions = [0, 90, 180, 270]

for idx, row in df_cam_locs.iterrows():
    for angle in directions:
        candidates_list.append({
            'loc_id': idx,
            'x': row['x'],
            'y': row['y'],
            'dir_angle': angle
        })

df_candidates = pd.DataFrame(candidates_list)
df_candidates['candidate_id'] = range(len(df_candidates))
num_cands = len(df_candidates)
print(f"Total Candidates Generated: {num_cands}")

# ==============================
# 4. COMPUTE COVERAGE MATRIX
# ==============================
print("Computing Coverage Matrix...")
dem_x = df_demand['x'].values[:, np.newaxis]
dem_y = df_demand['y'].values[:, np.newaxis]
sparse_chunks = []

for start_idx in range(0, num_cands, BATCH_SIZE):
    end_idx = min(start_idx + BATCH_SIZE, num_cands)
    batch_df = df_candidates.iloc[start_idx:end_idx]
    cand_x = batch_df['x'].values[np.newaxis, :]
    cand_y = batch_df['y'].values[np.newaxis, :]
    cand_dirs = batch_df['dir_angle'].values[np.newaxis, :]
    
    dx = dem_x - cand_x
    dy = dem_y - cand_y
    distances = np.sqrt(dx**2 + dy**2)
    point_angles = np.degrees(np.arctan2(dx, -dy))
    point_angles = (point_angles + 360) % 360
    angle_diff = np.abs(point_angles - cand_dirs)
    angle_diff = np.minimum(angle_diff, 360 - angle_diff)
    
    mask = (distances <= CAMERA_RADIUS) & (angle_diff <= FOV_HALF_ANGLE) & (distances > 0)
    sparse_chunks.append(sparse.csc_matrix(mask))

coverage_matrix = sparse.hstack(sparse_chunks).tocsc()

# ==============================
# 5. LOAD INITIAL SOLUTION
# ==============================
print(f"--- Loading Initial Solution from {INITIAL_SOLUTION_FILE} ---")
try:
    df_initial = pd.read_excel(INITIAL_SOLUTION_FILE, sheet_name='Selected_Cameras')
    initial_indices = df_initial['candidate_id'].values.tolist()
    current_solution_set = set(initial_indices)
    print(f"Loaded {len(current_solution_set)} cameras from file.")
except Exception as e:
    print(f"ERROR: {e}")
    exit()

# Reconstruct State
print("Reconstructing coverage state...")
selected_matrix = coverage_matrix[:, initial_indices]
coverage_counts = np.array(selected_matrix.sum(axis=1)).flatten()

current_covered_mask = coverage_counts > 0
initial_covered_weight = demand_weights[current_covered_mask].sum()
initial_pct = (initial_covered_weight / TOTAL_DEMAND_WEIGHT) * 100

print(f"Initial Z: {initial_covered_weight:,.4f} ({initial_pct:.2f}%)")

# ==============================
# 6. 1-SWAP LOCAL SEARCH
# ==============================
print(f"\n--- Starting 1-Swap Local Search ---")

best_solution = set(current_solution_set)
best_coverage_val = initial_covered_weight
improved = True
iteration = 0

# Lists to store history for plotting
history_iterations = [0]
history_obj_values = [best_coverage_val]

while improved:
    improved = False
    iteration += 1
    # print(f"  > Iteration {iteration} scanning...")
    
    current_sel = list(best_solution)
    current_unsel = list(set(range(num_cands)) - best_solution)
    random.shuffle(current_unsel) 
    
    swap_found = False
    
    for cam_out in current_sel:
        # LOSS
        pts_covered_by_out = coverage_matrix[:, cam_out].indices
        uniquely_covered_mask = coverage_counts[pts_covered_by_out] == 1
        points_losing_coverage = pts_covered_by_out[uniquely_covered_mask]
        loss_val = demand_weights[points_losing_coverage].sum()
        
        for cam_in in current_unsel:
            # GAIN
            pts_covered_by_in = coverage_matrix[:, cam_in].indices
            currently_uncovered_mask = coverage_counts[pts_covered_by_in] == 0
            points_gaining_coverage = pts_covered_by_in[currently_uncovered_mask]
            gain_val = demand_weights[points_gaining_coverage].sum()
            
            # NET IMPROVEMENT
            net_improvement = gain_val - loss_val
            
            if net_improvement > MIN_IMPROVEMENT_THRESHOLD:
                
                best_solution.remove(cam_out)
                best_solution.add(cam_in)
                best_coverage_val += net_improvement
                
                coverage_counts[pts_covered_by_out] -= 1
                coverage_counts[pts_covered_by_in] += 1
                
                print(f"  It {iteration}: Out {cam_out} / In {cam_in} | Gain +{net_improvement:.4f} | Z: {best_coverage_val:,.4f}")
                
                history_iterations.append(iteration)
                history_obj_values.append(best_coverage_val)
                
                swap_found = True
                improved = True
                break 
        
        if swap_found:
            break
            
    if not swap_found:
        print("  No further improvements found. Local Optimum reached.")

# ==============================
# 7. FINAL METRICS
# ==============================
final_pct = (best_coverage_val / TOTAL_DEMAND_WEIGHT) * 100
end_time = time.time()
run_time_seconds = end_time - start_time

print("-" * 40)
print(f"INITIAL Z:      {initial_covered_weight:,.4f} ({initial_pct:.2f}%)")
print(f"FINAL Z:        {best_coverage_val:,.4f} ({final_pct:.2f}%)")
print(f"IMPROVEMENT:    +{best_coverage_val - initial_covered_weight:,.4f}")
print(f"TOTAL RUN TIME: {run_time_seconds:.2f} seconds")
print("-" * 40)

# ==============================
# 8. EXCEL SAVE
# ==============================
final_indices = list(best_solution)
df_results_cam = df_candidates.iloc[final_indices].copy()
df_results_cam = df_results_cam[['candidate_id', 'loc_id', 'x', 'y', 'dir_angle']]
df_results_cam.rename(columns={'dir_angle': 'direction_degrees'}, inplace=True)

df_results_dem = df_demand.copy()
df_results_dem['cameras_covering'] = coverage_counts 
df_results_dem['is_covered'] = np.where(df_results_dem['cameras_covering'] > 0, 'Yes', 'No')

with pd.ExcelWriter(OUTPUT_EXCEL_FILE) as writer:
    df_results_cam.to_excel(writer, sheet_name='Selected_Cameras', index=False)
    df_results_dem.to_excel(writer, sheet_name='Demand_Coverage', index=False)
print(f"Excel saved: {OUTPUT_EXCEL_FILE}")

# ==============================
# 9. HEATMAP VISUALIZATION
# ==============================
final_heatmap_grid = np.zeros((TARGET_ROWS, TARGET_COLS))
dem_x_flat = df_demand['x'].values
dem_y_flat = df_demand['y'].values

for i, count in enumerate(coverage_counts):
    if count > 0:
        r, c = int(dem_y_flat[i]), int(dem_x_flat[i])
        if 0 <= r < TARGET_ROWS and 0 <= c < TARGET_COLS:
            final_heatmap_grid[r, c] = count

aspect_ratio = TARGET_ROWS / TARGET_COLS
base_width = 10 
fig_height = base_width * aspect_ratio
fig, ax = plt.subplots(figsize=(base_width, fig_height), dpi=120)

ax.set_xlim(0, TARGET_COLS)
ax.set_ylim(TARGET_ROWS, 0) 
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_xticks(np.arange(0, TARGET_COLS + 1, 10))
ax.set_yticks(np.arange(0, TARGET_ROWS + 1, 10))
ax.set_xticks(np.arange(0.5, TARGET_COLS, 1), minor=True)
ax.set_yticks(np.arange(0.5, TARGET_ROWS, 1), minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
ax.tick_params(which='minor', bottom=False, left=False)

colors = ['#ffffff00', '#800080', '#00ced1', '#ffd700'] 
cmap = ListedColormap(colors)
bounds = [0, 1, 2, 3, 100]
norm = BoundaryNorm(bounds, cmap.N)
ax.imshow(final_heatmap_grid, extent=[0, TARGET_COLS, TARGET_ROWS, 0], cmap=cmap, norm=norm, interpolation='nearest', zorder=1)

legend_elements = [Line2D([0], [0], marker='s', color='w', markerfacecolor='#800080', markersize=10, label='1 Cam'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='#00ced1', markersize=10, label='2 Cams'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='#ffd700', markersize=10, label='3+ Cams')]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

# Update Title to include Coverage and Run Time
title_text = f"{HEATMAP_TITLE}\nCoverage: {final_pct:.2f}% | Run Time: {run_time_seconds:.1f}s"
ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches='tight')
plt.close()
print(f"Heatmap saved: {OUTPUT_IMAGE_FILE}")

# ==============================
# 10. PLOT OPTIMIZATION TRAJECTORY
# ==============================
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(history_iterations, history_obj_values, marker='o', linestyle='-', color='#e74c3c', markersize=4, label='Obj Function (Z)')

plt.annotate(f"Start: {history_obj_values[0]:,.0f}", (history_iterations[0], history_obj_values[0]), 
             textcoords="offset points", xytext=(10, 0), ha='left', fontsize=9)
plt.annotate(f"End: {history_obj_values[-1]:,.0f}", (history_iterations[-1], history_obj_values[-1]), 
             textcoords="offset points", xytext=(-10, 10), ha='right', fontsize=9)

# Update Title to include Coverage and Run Time
plot_title = f"Local Search Trajectory\nFinal Coverage: {final_pct:.2f}% | Time: {run_time_seconds:.1f}s"
plt.title(plot_title, fontsize=14, fontweight='bold')
plt.xlabel("Iteration Number")
plt.ylabel("Total Weighted Coverage (Z)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PLOT_FILE)
plt.close()

print(f"Trajectory plot saved: {OUTPUT_PLOT_FILE}")




























# ==============================
# 1. SETTINGS
# ==============================
INPUT_FILE = "new.xlsx"
INITIAL_SOLUTION_FILE = "greedy_solution_final_soln3.xlsx" 

DEMAND_SHEET = "demand"
CAMERA_SHEET = "camera"

OUTPUT_IMAGE_FILE = "local_search_heatmap_soln3.png"
OUTPUT_PLOT_FILE = "local_search_plot_soln3.png"
OUTPUT_EXCEL_FILE = "local_search_soln3.xlsx"
HEATMAP_TITLE = "Solution 3: 1-Swap Local Search"

# Optimization Parameters
CAMERA_RADIUS = 5
FOV_HALF_ANGLE = 45

# Threshold to prevent swapping for negligible/zero gains
MIN_IMPROVEMENT_THRESHOLD = 0.01 

# Grid Settings
TARGET_COLS = 80
TARGET_ROWS = 130
SUBTRACT_ONE = 0 
BATCH_SIZE = 1000 

print("--- PROCESS STARTED ---")
start_time = time.time()

# ==============================
# 2. LOAD DATA
# ==============================
df_demand = pd.read_excel(INPUT_FILE, sheet_name=DEMAND_SHEET)
df_demand.columns = [c.strip().lower() for c in df_demand.columns]
df_demand = df_demand[df_demand['weight'] > 0].copy()
df_demand['x'] = df_demand['x'] - SUBTRACT_ONE
df_demand['y'] = df_demand['y'] - SUBTRACT_ONE

TOTAL_DEMAND_WEIGHT = df_demand['weight'].sum()
demand_weights = df_demand['weight'].values

df_cam_locs = pd.read_excel(INPUT_FILE, sheet_name=CAMERA_SHEET)
df_cam_locs.columns = [c.strip().lower() for c in df_cam_locs.columns]
df_cam_locs['x'] = df_cam_locs['x'] - SUBTRACT_ONE
df_cam_locs['y'] = df_cam_locs['y'] - SUBTRACT_ONE

print(f"Demand Points: {len(df_demand)} | Total Weight: {TOTAL_DEMAND_WEIGHT:,.4f}")

# ==============================
# 3. GENERATE CANDIDATES
# ==============================
candidates_list = []
directions = [0, 90, 180, 270]

for idx, row in df_cam_locs.iterrows():
    for angle in directions:
        candidates_list.append({
            'loc_id': idx,
            'x': row['x'],
            'y': row['y'],
            'dir_angle': angle
        })

df_candidates = pd.DataFrame(candidates_list)
df_candidates['candidate_id'] = range(len(df_candidates))
num_cands = len(df_candidates)
print(f"Total Candidates Generated: {num_cands}")

# ==============================
# 4. COMPUTE COVERAGE MATRIX
# ==============================
print("Computing Coverage Matrix...")
dem_x = df_demand['x'].values[:, np.newaxis]
dem_y = df_demand['y'].values[:, np.newaxis]
sparse_chunks = []

for start_idx in range(0, num_cands, BATCH_SIZE):
    end_idx = min(start_idx + BATCH_SIZE, num_cands)
    batch_df = df_candidates.iloc[start_idx:end_idx]
    cand_x = batch_df['x'].values[np.newaxis, :]
    cand_y = batch_df['y'].values[np.newaxis, :]
    cand_dirs = batch_df['dir_angle'].values[np.newaxis, :]
    
    dx = dem_x - cand_x
    dy = dem_y - cand_y
    distances = np.sqrt(dx**2 + dy**2)
    point_angles = np.degrees(np.arctan2(dx, -dy))
    point_angles = (point_angles + 360) % 360
    angle_diff = np.abs(point_angles - cand_dirs)
    angle_diff = np.minimum(angle_diff, 360 - angle_diff)
    
    mask = (distances <= CAMERA_RADIUS) & (angle_diff <= FOV_HALF_ANGLE) & (distances > 0)
    sparse_chunks.append(sparse.csc_matrix(mask))

coverage_matrix = sparse.hstack(sparse_chunks).tocsc()

# ==============================
# 5. LOAD INITIAL SOLUTION
# ==============================
print(f"--- Loading Initial Solution from {INITIAL_SOLUTION_FILE} ---")
try:
    df_initial = pd.read_excel(INITIAL_SOLUTION_FILE, sheet_name='Selected_Cameras')
    initial_indices = df_initial['candidate_id'].values.tolist()
    current_solution_set = set(initial_indices)
    print(f"Loaded {len(current_solution_set)} cameras from file.")
except Exception as e:
    print(f"ERROR: {e}")
    exit()

# Reconstruct State
print("Reconstructing coverage state...")
selected_matrix = coverage_matrix[:, initial_indices]
coverage_counts = np.array(selected_matrix.sum(axis=1)).flatten()

current_covered_mask = coverage_counts > 0
initial_covered_weight = demand_weights[current_covered_mask].sum()
initial_pct = (initial_covered_weight / TOTAL_DEMAND_WEIGHT) * 100

print(f"Initial Z: {initial_covered_weight:,.4f} ({initial_pct:.2f}%)")

# ==============================
# 6. 1-SWAP LOCAL SEARCH
# ==============================
print(f"\n--- Starting 1-Swap Local Search ---")

best_solution = set(current_solution_set)
best_coverage_val = initial_covered_weight
improved = True
iteration = 0

# Lists to store history for plotting
history_iterations = [0]
history_obj_values = [best_coverage_val]

while improved:
    improved = False
    iteration += 1
    # print(f"  > Iteration {iteration} scanning...")
    
    current_sel = list(best_solution)
    current_unsel = list(set(range(num_cands)) - best_solution)
    random.shuffle(current_unsel) 
    
    swap_found = False
    
    for cam_out in current_sel:
        # LOSS
        pts_covered_by_out = coverage_matrix[:, cam_out].indices
        uniquely_covered_mask = coverage_counts[pts_covered_by_out] == 1
        points_losing_coverage = pts_covered_by_out[uniquely_covered_mask]
        loss_val = demand_weights[points_losing_coverage].sum()
        
        for cam_in in current_unsel:
            # GAIN
            pts_covered_by_in = coverage_matrix[:, cam_in].indices
            currently_uncovered_mask = coverage_counts[pts_covered_by_in] == 0
            points_gaining_coverage = pts_covered_by_in[currently_uncovered_mask]
            gain_val = demand_weights[points_gaining_coverage].sum()
            
            # NET IMPROVEMENT
            net_improvement = gain_val - loss_val
            
            if net_improvement > MIN_IMPROVEMENT_THRESHOLD:
                
                best_solution.remove(cam_out)
                best_solution.add(cam_in)
                best_coverage_val += net_improvement
                
                coverage_counts[pts_covered_by_out] -= 1
                coverage_counts[pts_covered_by_in] += 1
                
                print(f"  It {iteration}: Out {cam_out} / In {cam_in} | Gain +{net_improvement:.4f} | Z: {best_coverage_val:,.4f}")
                
                history_iterations.append(iteration)
                history_obj_values.append(best_coverage_val)
                
                swap_found = True
                improved = True
                break 
        
        if swap_found:
            break
            
    if not swap_found:
        print("  No further improvements found. Local Optimum reached.")

# ==============================
# 7. FINAL METRICS
# ==============================
final_pct = (best_coverage_val / TOTAL_DEMAND_WEIGHT) * 100
end_time = time.time()
run_time_seconds = end_time - start_time

print("-" * 40)
print(f"INITIAL Z:      {initial_covered_weight:,.4f} ({initial_pct:.2f}%)")
print(f"FINAL Z:        {best_coverage_val:,.4f} ({final_pct:.2f}%)")
print(f"IMPROVEMENT:    +{best_coverage_val - initial_covered_weight:,.4f}")
print(f"TOTAL RUN TIME: {run_time_seconds:.2f} seconds")
print("-" * 40)

# ==============================
# 8. EXCEL SAVE
# ==============================
final_indices = list(best_solution)
df_results_cam = df_candidates.iloc[final_indices].copy()
df_results_cam = df_results_cam[['candidate_id', 'loc_id', 'x', 'y', 'dir_angle']]
df_results_cam.rename(columns={'dir_angle': 'direction_degrees'}, inplace=True)

df_results_dem = df_demand.copy()
df_results_dem['cameras_covering'] = coverage_counts 
df_results_dem['is_covered'] = np.where(df_results_dem['cameras_covering'] > 0, 'Yes', 'No')

with pd.ExcelWriter(OUTPUT_EXCEL_FILE) as writer:
    df_results_cam.to_excel(writer, sheet_name='Selected_Cameras', index=False)
    df_results_dem.to_excel(writer, sheet_name='Demand_Coverage', index=False)
print(f"Excel saved: {OUTPUT_EXCEL_FILE}")

# ==============================
# 9. HEATMAP VISUALIZATION
# ==============================
final_heatmap_grid = np.zeros((TARGET_ROWS, TARGET_COLS))
dem_x_flat = df_demand['x'].values
dem_y_flat = df_demand['y'].values

for i, count in enumerate(coverage_counts):
    if count > 0:
        r, c = int(dem_y_flat[i]), int(dem_x_flat[i])
        if 0 <= r < TARGET_ROWS and 0 <= c < TARGET_COLS:
            final_heatmap_grid[r, c] = count

aspect_ratio = TARGET_ROWS / TARGET_COLS
base_width = 10 
fig_height = base_width * aspect_ratio
fig, ax = plt.subplots(figsize=(base_width, fig_height), dpi=120)

ax.set_xlim(0, TARGET_COLS)
ax.set_ylim(TARGET_ROWS, 0) 
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_xticks(np.arange(0, TARGET_COLS + 1, 10))
ax.set_yticks(np.arange(0, TARGET_ROWS + 1, 10))
ax.set_xticks(np.arange(0.5, TARGET_COLS, 1), minor=True)
ax.set_yticks(np.arange(0.5, TARGET_ROWS, 1), minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
ax.tick_params(which='minor', bottom=False, left=False)

colors = ['#ffffff00', '#800080', '#00ced1', '#ffd700'] 
cmap = ListedColormap(colors)
bounds = [0, 1, 2, 3, 100]
norm = BoundaryNorm(bounds, cmap.N)
ax.imshow(final_heatmap_grid, extent=[0, TARGET_COLS, TARGET_ROWS, 0], cmap=cmap, norm=norm, interpolation='nearest', zorder=1)

legend_elements = [Line2D([0], [0], marker='s', color='w', markerfacecolor='#800080', markersize=10, label='1 Cam'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='#00ced1', markersize=10, label='2 Cams'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='#ffd700', markersize=10, label='3+ Cams')]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

# Update Title to include Coverage and Run Time
title_text = f"{HEATMAP_TITLE}\nCoverage: {final_pct:.2f}% | Run Time: {run_time_seconds:.1f}s"
ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches='tight')
plt.close()
print(f"Heatmap saved: {OUTPUT_IMAGE_FILE}")

# ==============================
# 10. PLOT OPTIMIZATION TRAJECTORY
# ==============================
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(history_iterations, history_obj_values, marker='o', linestyle='-', color='#e74c3c', markersize=4, label='Obj Function (Z)')

plt.annotate(f"Start: {history_obj_values[0]:,.0f}", (history_iterations[0], history_obj_values[0]), 
             textcoords="offset points", xytext=(10, 0), ha='left', fontsize=9)
plt.annotate(f"End: {history_obj_values[-1]:,.0f}", (history_iterations[-1], history_obj_values[-1]), 
             textcoords="offset points", xytext=(-10, 10), ha='right', fontsize=9)

# Update Title to include Coverage and Run Time
plot_title = f"Local Search Trajectory\nFinal Coverage: {final_pct:.2f}% | Time: {run_time_seconds:.1f}s"
plt.title(plot_title, fontsize=14, fontweight='bold')
plt.xlabel("Iteration Number")
plt.ylabel("Total Weighted Coverage (Z)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PLOT_FILE)
plt.close()

print(f"Trajectory plot saved: {OUTPUT_PLOT_FILE}")