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
INPUT_FILE = "Final_Dataset_MCLP.xlsx"
DEMAND_SHEET = "demand"
CAMERA_SHEET = "camera"

# --- OUTPUT FILES ---
OUTPUT_IMAGE_FILE = "greedy_heatmap_soln1.png"
OUTPUT_CURVE_FILE = "greedy_growth_curve_soln1.png"
OUTPUT_EXCEL_FILE = "greedy_solution_final_soln1.xlsx"
HEATMAP_TITLE = "Solution 1: Randomized Greedy (100 Cameras)"

# Optimization Parameters
CAMERA_RADIUS = 5
FOV_HALF_ANGLE = 45
MAX_ITERATIONS = 10000  # High limit to ensure we reach 100%

# Randomization (RCL) settings
# RCL_SIZE = 50 (High Exploration)
RCL_SIZE = 50           
RANDOM_SEED = None      # Set to integer (e.g. 42) for reproducible results

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
# 5. RANDOMIZED GREEDY OPTIMIZATION (Until 100%)
# ==============================
print(f"Optimizing until 100% coverage...")

current_weights = df_demand['weight'].values.copy()
selected_indices = []
history = []

for i in range(MAX_ITERATIONS):
    # A. Calculate gains
    gains = coverage_matrix.T @ current_weights
    
    valid_gains_mask = gains > 0
    valid_indices = np.where(valid_gains_mask)[0]
    
    # STOPPING CONDITION 1: No more gain possible
    if len(valid_indices) == 0:
        print(f"  -> Optimization stopped at Camera {i}: No further gain possible.")
        break
        
    # B. RCL Selection (Top RCL_SIZE Best)
    current_rcl_size = min(RCL_SIZE, len(valid_indices))
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
    
    # REPORTING: Every 10 steps or exactly at 100
    count = i + 1
    if count == 1 or count % 10 == 0:
        print(f"Cam {count:<5} | Coverage: {pct:.2f}%")
        
    # STOPPING CONDITION 2: 100% Coverage Reached
    if pct >= 100.0:
        print(f"  -> Optimization stopped at Camera {count}: 100% Coverage reached!")
        break

# ==============================
# 6. SCENARIO REPORTING
# ==============================
end_time = time.time()
run_time = end_time - start_time
final_pct = history[-1]['coverage_pct']
total_cams = len(selected_indices)

print("\n" + "="*50)
print(f" FINAL REPORT")
print("="*50)
print(f"Total Cameras Needed for {final_pct:.2f}%: {total_cams}")

# Check status at exactly 100 cameras
pct_100 = 0.0
if len(history) >= 100:
    pct_100 = history[99]['coverage_pct']
    print(f"Status at 100 Cameras:     {pct_100:.2f}% coverage")
else:
    print(f"Status at 100 Cameras:     N/A (Reached 100% before 100 cameras)")
    pct_100 = final_pct

print("-" * 50)
print(f"Time Elapsed: {run_time:.2f}s")
print("="*50 + "\n")

# ==============================
# 7. SAVE FULL RESULTS TO EXCEL
# ==============================
print(f"Saving to {OUTPUT_EXCEL_FILE}...")

# 1. Camera Solution (Full list)
df_results_cam = df_candidates.iloc[selected_indices].copy()
df_results_cam['selection_order'] = range(1, len(df_results_cam) + 1)
df_results_cam = df_results_cam[['selection_order', 'candidate_id', 'loc_id', 'x', 'y', 'dir_angle']]
df_results_cam.rename(columns={'dir_angle': 'direction_degrees'}, inplace=True)

# 2. Demand Coverage Status (Full Coverage)
selected_matrix_full = coverage_matrix[:, selected_indices]
coverage_counts_full = np.array(selected_matrix_full.sum(axis=1)).flatten()

df_results_dem = df_demand.copy()
df_results_dem['cameras_covering'] = coverage_counts_full
df_results_dem['is_covered'] = np.where(df_results_dem['cameras_covering'] > 0, 'Yes', 'No')

# 3. Growth History
df_history = pd.DataFrame(history)

with pd.ExcelWriter(OUTPUT_EXCEL_FILE) as writer:
    df_results_cam.to_excel(writer, sheet_name='Selected_Cameras', index=False)
    df_results_dem.to_excel(writer, sheet_name='Demand_Coverage', index=False)
    df_history.to_excel(writer, sheet_name='Growth_History', index=False)

print("Excel file created successfully.")

# ==============================
# 8. VISUALIZATION: GROWTH CURVE (FULL)
# ==============================
plt.figure(figsize=(10, 6), dpi=100)
x_vals = [h['camera_count'] for h in history]
y_vals = [h['coverage_pct'] for h in history]

plt.plot(x_vals, y_vals, color='#2980b9', linewidth=2, label='Coverage %')

# Mark the 100th camera point if it exists
if len(history) >= 100:
    val_100 = history[99]['coverage_pct']
    plt.plot(100, val_100, 'o', color='orange', label='100 Cameras')
    plt.annotate(f"100 Cams: {val_100:.1f}%", (100, val_100), textcoords="offset points", xytext=(0, -20), ha='center', fontsize=9)

# Mark the Final point
plt.plot(x_vals[-1], y_vals[-1], 'o', color='green', label='Final')
plt.annotate(f"Final: {y_vals[-1]:.1f}%", (x_vals[-1], y_vals[-1]), textcoords="offset points", xytext=(0, 10), ha='right', fontsize=9)

plt.title(f"Coverage Growth to 100%\nTotal Cams: {total_cams}", fontsize=14, fontweight='bold')
plt.xlabel("Number of Cameras")
plt.ylabel("Coverage Percentage (%)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_CURVE_FILE)
print(f"Curve saved: {OUTPUT_CURVE_FILE}")

# ==============================
# 9. VISUALIZATION: HEATMAP (ONLY FIRST 100 CAMERAS)
# ==============================
# Filter to get only the first 100 indices (or less if total < 100)
limit = 100
indices_100 = selected_indices[:limit]

# Re-compute coverage counts for just these 100 cameras
selected_matrix_100 = coverage_matrix[:, indices_100]
coverage_counts_100 = np.array(selected_matrix_100.sum(axis=1)).flatten()

# Generate Grid for Heatmap
final_heatmap_grid = np.zeros((TARGET_ROWS, TARGET_COLS))
dem_x_flat = df_demand['x'].values
dem_y_flat = df_demand['y'].values

for i, count in enumerate(coverage_counts_100):
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
ax.grid(which='major', color='gray', linestyle=':', linewidth=0.5)

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

# Use the specific coverage percentage for 100 cameras in the title
ax.set_title(f"{HEATMAP_TITLE}\nCoverage at 100 Cams: {pct_100:.1f}%", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches='tight')
print(f"Heatmap (100 Cams) saved: {OUTPUT_IMAGE_FILE}")






























# ==============================
# 1. SETTINGS
# ==============================
INPUT_FILE = "Final_Dataset_MCLP.xlsx"
DEMAND_SHEET = "demand"
CAMERA_SHEET = "camera"

# --- OUTPUT FILES ---
OUTPUT_IMAGE_FILE = "greedy_heatmap_soln2.png"
OUTPUT_CURVE_FILE = "greedy_growth_curve_soln2.png"
OUTPUT_EXCEL_FILE = "greedy_solution_final_soln2.xlsx"
HEATMAP_TITLE = "Solution 2: Randomized Greedy (100 Cameras)"

# Optimization Parameters
CAMERA_RADIUS = 5
FOV_HALF_ANGLE = 45
MAX_ITERATIONS = 10000  # High limit to ensure we reach 100%

# Randomization (RCL) settings
# RCL_SIZE = 50 (High Exploration)
RCL_SIZE = 50           
RANDOM_SEED = None      # Set to integer (e.g. 42) for reproducible results

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
# 5. RANDOMIZED GREEDY OPTIMIZATION (Until 100%)
# ==============================
print(f"Optimizing until 100% coverage...")

current_weights = df_demand['weight'].values.copy()
selected_indices = []
history = []

for i in range(MAX_ITERATIONS):
    # A. Calculate gains
    gains = coverage_matrix.T @ current_weights
    
    valid_gains_mask = gains > 0
    valid_indices = np.where(valid_gains_mask)[0]
    
    # STOPPING CONDITION 1: No more gain possible
    if len(valid_indices) == 0:
        print(f"  -> Optimization stopped at Camera {i}: No further gain possible.")
        break
        
    # B. RCL Selection (Top RCL_SIZE Best)
    current_rcl_size = min(RCL_SIZE, len(valid_indices))
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
    
    # REPORTING: Every 10 steps or exactly at 100
    count = i + 1
    if count == 1 or count % 10 == 0:
        print(f"Cam {count:<5} | Coverage: {pct:.2f}%")
        
    # STOPPING CONDITION 2: 100% Coverage Reached
    if pct >= 100.0:
        print(f"  -> Optimization stopped at Camera {count}: 100% Coverage reached!")
        break

# ==============================
# 6. SCENARIO REPORTING
# ==============================
end_time = time.time()
run_time = end_time - start_time
final_pct = history[-1]['coverage_pct']
total_cams = len(selected_indices)

print("\n" + "="*50)
print(f" FINAL REPORT")
print("="*50)
print(f"Total Cameras Needed for {final_pct:.2f}%: {total_cams}")

# Check status at exactly 100 cameras
pct_100 = 0.0
if len(history) >= 100:
    pct_100 = history[99]['coverage_pct']
    print(f"Status at 100 Cameras:     {pct_100:.2f}% coverage")
else:
    print(f"Status at 100 Cameras:     N/A (Reached 100% before 100 cameras)")
    pct_100 = final_pct

print("-" * 50)
print(f"Time Elapsed: {run_time:.2f}s")
print("="*50 + "\n")

# ==============================
# 7. SAVE FULL RESULTS TO EXCEL
# ==============================
print(f"Saving to {OUTPUT_EXCEL_FILE}...")

# 1. Camera Solution (Full list)
df_results_cam = df_candidates.iloc[selected_indices].copy()
df_results_cam['selection_order'] = range(1, len(df_results_cam) + 1)
df_results_cam = df_results_cam[['selection_order', 'candidate_id', 'loc_id', 'x', 'y', 'dir_angle']]
df_results_cam.rename(columns={'dir_angle': 'direction_degrees'}, inplace=True)

# 2. Demand Coverage Status (Full Coverage)
selected_matrix_full = coverage_matrix[:, selected_indices]
coverage_counts_full = np.array(selected_matrix_full.sum(axis=1)).flatten()

df_results_dem = df_demand.copy()
df_results_dem['cameras_covering'] = coverage_counts_full
df_results_dem['is_covered'] = np.where(df_results_dem['cameras_covering'] > 0, 'Yes', 'No')

# 3. Growth History
df_history = pd.DataFrame(history)

with pd.ExcelWriter(OUTPUT_EXCEL_FILE) as writer:
    df_results_cam.to_excel(writer, sheet_name='Selected_Cameras', index=False)
    df_results_dem.to_excel(writer, sheet_name='Demand_Coverage', index=False)
    df_history.to_excel(writer, sheet_name='Growth_History', index=False)

print("Excel file created successfully.")

# ==============================
# 8. VISUALIZATION: GROWTH CURVE (FULL)
# ==============================
plt.figure(figsize=(10, 6), dpi=100)
x_vals = [h['camera_count'] for h in history]
y_vals = [h['coverage_pct'] for h in history]

plt.plot(x_vals, y_vals, color='#2980b9', linewidth=2, label='Coverage %')

# Mark the 100th camera point if it exists
if len(history) >= 100:
    val_100 = history[99]['coverage_pct']
    plt.plot(100, val_100, 'o', color='orange', label='100 Cameras')
    plt.annotate(f"100 Cams: {val_100:.1f}%", (100, val_100), textcoords="offset points", xytext=(0, -20), ha='center', fontsize=9)

# Mark the Final point
plt.plot(x_vals[-1], y_vals[-1], 'o', color='green', label='Final')
plt.annotate(f"Final: {y_vals[-1]:.1f}%", (x_vals[-1], y_vals[-1]), textcoords="offset points", xytext=(0, 10), ha='right', fontsize=9)

plt.title(f"Coverage Growth to 100%\nTotal Cams: {total_cams}", fontsize=14, fontweight='bold')
plt.xlabel("Number of Cameras")
plt.ylabel("Coverage Percentage (%)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_CURVE_FILE)
print(f"Curve saved: {OUTPUT_CURVE_FILE}")

# ==============================
# 9. VISUALIZATION: HEATMAP (ONLY FIRST 100 CAMERAS)
# ==============================
# Filter to get only the first 100 indices (or less if total < 100)
limit = 100
indices_100 = selected_indices[:limit]

# Re-compute coverage counts for just these 100 cameras
selected_matrix_100 = coverage_matrix[:, indices_100]
coverage_counts_100 = np.array(selected_matrix_100.sum(axis=1)).flatten()

# Generate Grid for Heatmap
final_heatmap_grid = np.zeros((TARGET_ROWS, TARGET_COLS))
dem_x_flat = df_demand['x'].values
dem_y_flat = df_demand['y'].values

for i, count in enumerate(coverage_counts_100):
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
ax.grid(which='major', color='gray', linestyle=':', linewidth=0.5)

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

# Use the specific coverage percentage for 100 cameras in the title
ax.set_title(f"{HEATMAP_TITLE}\nCoverage at 100 Cams: {pct_100:.1f}%", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches='tight')
print(f"Heatmap (100 Cams) saved: {OUTPUT_IMAGE_FILE}")

















































# ==============================
# 1. SETTINGS
# ==============================
INPUT_FILE = "Final_Dataset_MCLP.xlsx"
DEMAND_SHEET = "demand"
CAMERA_SHEET = "camera"

# --- OUTPUT FILES ---
OUTPUT_IMAGE_FILE = "greedy_heatmap_soln3.png"
OUTPUT_CURVE_FILE = "greedy_growth_curve_soln3.png"
OUTPUT_EXCEL_FILE = "greedy_solution_final_soln3.xlsx"
HEATMAP_TITLE = "Solution 3: Randomized Greedy (100 Cameras)"

# Optimization Parameters
CAMERA_RADIUS = 5
FOV_HALF_ANGLE = 45
MAX_ITERATIONS = 10000  # High limit to ensure we reach 100%

# Randomization (RCL) settings
# RCL_SIZE = 50 (High Exploration)
RCL_SIZE = 50           
RANDOM_SEED = None      # Set to integer (e.g. 42) for reproducible results

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
# 5. RANDOMIZED GREEDY OPTIMIZATION (Until 100%)
# ==============================
print(f"Optimizing until 100% coverage...")

current_weights = df_demand['weight'].values.copy()
selected_indices = []
history = []

for i in range(MAX_ITERATIONS):
    # A. Calculate gains
    gains = coverage_matrix.T @ current_weights
    
    valid_gains_mask = gains > 0
    valid_indices = np.where(valid_gains_mask)[0]
    
    # STOPPING CONDITION 1: No more gain possible
    if len(valid_indices) == 0:
        print(f"  -> Optimization stopped at Camera {i}: No further gain possible.")
        break
        
    # B. RCL Selection (Top RCL_SIZE Best)
    current_rcl_size = min(RCL_SIZE, len(valid_indices))
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
    
    # REPORTING: Every 10 steps or exactly at 100
    count = i + 1
    if count == 1 or count % 10 == 0:
        print(f"Cam {count:<5} | Coverage: {pct:.2f}%")
        
    # STOPPING CONDITION 2: 100% Coverage Reached
    if pct >= 100.0:
        print(f"  -> Optimization stopped at Camera {count}: 100% Coverage reached!")
        break

# ==============================
# 6. SCENARIO REPORTING
# ==============================
end_time = time.time()
run_time = end_time - start_time
final_pct = history[-1]['coverage_pct']
total_cams = len(selected_indices)

print("\n" + "="*50)
print(f" FINAL REPORT")
print("="*50)
print(f"Total Cameras Needed for {final_pct:.2f}%: {total_cams}")

# Check status at exactly 100 cameras
pct_100 = 0.0
if len(history) >= 100:
    pct_100 = history[99]['coverage_pct']
    print(f"Status at 100 Cameras:     {pct_100:.2f}% coverage")
else:
    print(f"Status at 100 Cameras:     N/A (Reached 100% before 100 cameras)")
    pct_100 = final_pct

print("-" * 50)
print(f"Time Elapsed: {run_time:.2f}s")
print("="*50 + "\n")

# ==============================
# 7. SAVE FULL RESULTS TO EXCEL
# ==============================
print(f"Saving to {OUTPUT_EXCEL_FILE}...")

# 1. Camera Solution (Full list)
df_results_cam = df_candidates.iloc[selected_indices].copy()
df_results_cam['selection_order'] = range(1, len(df_results_cam) + 1)
df_results_cam = df_results_cam[['selection_order', 'candidate_id', 'loc_id', 'x', 'y', 'dir_angle']]
df_results_cam.rename(columns={'dir_angle': 'direction_degrees'}, inplace=True)

# 2. Demand Coverage Status (Full Coverage)
selected_matrix_full = coverage_matrix[:, selected_indices]
coverage_counts_full = np.array(selected_matrix_full.sum(axis=1)).flatten()

df_results_dem = df_demand.copy()
df_results_dem['cameras_covering'] = coverage_counts_full
df_results_dem['is_covered'] = np.where(df_results_dem['cameras_covering'] > 0, 'Yes', 'No')

# 3. Growth History
df_history = pd.DataFrame(history)

with pd.ExcelWriter(OUTPUT_EXCEL_FILE) as writer:
    df_results_cam.to_excel(writer, sheet_name='Selected_Cameras', index=False)
    df_results_dem.to_excel(writer, sheet_name='Demand_Coverage', index=False)
    df_history.to_excel(writer, sheet_name='Growth_History', index=False)

print("Excel file created successfully.")

# ==============================
# 8. VISUALIZATION: GROWTH CURVE (FULL)
# ==============================
plt.figure(figsize=(10, 6), dpi=100)
x_vals = [h['camera_count'] for h in history]
y_vals = [h['coverage_pct'] for h in history]

plt.plot(x_vals, y_vals, color='#2980b9', linewidth=2, label='Coverage %')

# Mark the 100th camera point if it exists
if len(history) >= 100:
    val_100 = history[99]['coverage_pct']
    plt.plot(100, val_100, 'o', color='orange', label='100 Cameras')
    plt.annotate(f"100 Cams: {val_100:.1f}%", (100, val_100), textcoords="offset points", xytext=(0, -20), ha='center', fontsize=9)

# Mark the Final point
plt.plot(x_vals[-1], y_vals[-1], 'o', color='green', label='Final')
plt.annotate(f"Final: {y_vals[-1]:.1f}%", (x_vals[-1], y_vals[-1]), textcoords="offset points", xytext=(0, 10), ha='right', fontsize=9)

plt.title(f"Coverage Growth to 100%\nTotal Cams: {total_cams}", fontsize=14, fontweight='bold')
plt.xlabel("Number of Cameras")
plt.ylabel("Coverage Percentage (%)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_CURVE_FILE)
print(f"Curve saved: {OUTPUT_CURVE_FILE}")

# ==============================
# 9. VISUALIZATION: HEATMAP (ONLY FIRST 100 CAMERAS)
# ==============================
# Filter to get only the first 100 indices (or less if total < 100)
limit = 100
indices_100 = selected_indices[:limit]

# Re-compute coverage counts for just these 100 cameras
selected_matrix_100 = coverage_matrix[:, indices_100]
coverage_counts_100 = np.array(selected_matrix_100.sum(axis=1)).flatten()

# Generate Grid for Heatmap
final_heatmap_grid = np.zeros((TARGET_ROWS, TARGET_COLS))
dem_x_flat = df_demand['x'].values
dem_y_flat = df_demand['y'].values

for i, count in enumerate(coverage_counts_100):
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
ax.grid(which='major', color='gray', linestyle=':', linewidth=0.5)

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

# Use the specific coverage percentage for 100 cameras in the title
ax.set_title(f"{HEATMAP_TITLE}\nCoverage at 100 Cams: {pct_100:.1f}%", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches='tight')
print(f"Heatmap (100 Cams) saved: {OUTPUT_IMAGE_FILE}")