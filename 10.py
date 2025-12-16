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