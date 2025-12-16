import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import sparse
from matplotlib.lines import Line2D
import time

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