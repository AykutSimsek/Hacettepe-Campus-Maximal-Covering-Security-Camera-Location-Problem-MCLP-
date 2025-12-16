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