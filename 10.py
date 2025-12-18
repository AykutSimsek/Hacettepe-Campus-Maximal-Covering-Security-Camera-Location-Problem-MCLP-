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
SEED_FILE = "local_search_soln1.xlsx"  # CHANGE THIS for soln2 / soln3

DEMAND_SHEET = "demand"
CAMERA_SHEET = "camera"

# Outputs
OUTPUT_IMAGE_FILE = "pso_heatmap_soln1.png"
OUTPUT_PLOT_FILE = "pso_trajectory_plot_soln1.png"
OUTPUT_EXCEL_FILE = "pso_solution_soln1.xlsx"
HEATMAP_TITLE = "PSO Solution 1: 100 Cameras"

# Constraints
CAMERA_RADIUS = 5
FOV_HALF_ANGLE = 45
NUM_CAMERAS_TO_SELECT = 100

# Grid
TARGET_COLS = 80
TARGET_ROWS = 130
SUBTRACT_ONE = 0 
BATCH_SIZE = 1000 

# --- RUIN & RECREATE SETTINGS ---
SWARM_SIZE = 30           
MAX_ITERATIONS = 50       
W = 0.5                   
C1 = 1.5                  
C2 = 1.5   
RUIN_PERCENT = 0.10       # Re-optimize 10% (10 cameras) every time
MUTATION_PROB = 0.3       # 30% chance a particle undergoes Ruin & Recreate

print("--- RUIN & RECREATE PSO STARTED ---")
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
print("Coverage Matrix Computed.")

# ==============================
# 5. INITIALIZE SWARM
# ==============================
print(f"\n--- Initializing Swarm ---")

X = np.random.uniform(0.0, 0.4, (SWARM_SIZE, num_cands))

# Load Seed
seed_indices = []
try:
    print(f"Loading seed from: {SEED_FILE}")
    df_seed = pd.read_excel(SEED_FILE, sheet_name='Selected_Cameras')
    seed_indices = df_seed['candidate_id'].values[:NUM_CAMERAS_TO_SELECT].tolist()
    
    # Init Particle 0 exact
    X[0, seed_indices] = 1.0
    
    # Init others with variation
    for i in range(1, SWARM_SIZE):
        X[i, seed_indices] = np.random.uniform(0.8, 1.0, len(seed_indices))
        # Randomly remove 10% to let Ruin & Recreate fill them
        drop_indices = random.sample(seed_indices, int(NUM_CAMERAS_TO_SELECT * 0.1))
        X[i, drop_indices] = np.random.uniform(0.0, 0.2, len(drop_indices))
        
    print(f" >> Seed loaded and variations created.")
except Exception as e:
    print(f" >> ERROR loading seed: {e}. Using random start.")

V = np.random.uniform(-0.1, 0.1, (SWARM_SIZE, num_cands))

P_BEST_POS = X.copy()
P_BEST_SCORES = np.zeros(SWARM_SIZE)
G_BEST_POS = np.zeros(num_cands)
G_BEST_SCORE = -1.0
G_BEST_INDICES = []

def evaluate_particle(priority_vector):
    top_k_indices = np.argpartition(priority_vector, -NUM_CAMERAS_TO_SELECT)[-NUM_CAMERAS_TO_SELECT:]
    selected_matrix = coverage_matrix[:, top_k_indices]
    coverage_counts = np.array(selected_matrix.sum(axis=1)).flatten()
    covered_mask = coverage_counts > 0
    total_coverage = demand_weights[covered_mask].sum()
    return total_coverage, top_k_indices, coverage_counts

# Initial Eval
print("Evaluating initial swarm...")
for i in range(SWARM_SIZE):
    score, indices, _ = evaluate_particle(X[i])
    P_BEST_SCORES[i] = score
    if score > G_BEST_SCORE:
        G_BEST_SCORE = score
        G_BEST_POS = X[i].copy()
        G_BEST_INDICES = indices

print(f"Initial Global Best Z: {G_BEST_SCORE:,.4f}")

# ==============================
# 6. RUIN & RECREATE LOGIC
# ==============================

def ruin_and_recreate(current_indices, counts):
    """
    1. RUIN: Drop cameras that provide the least UNIQUE coverage.
    2. RECREATE: Greedily add cameras that cover the most UNCOVERED weight.
    """
    current_set = set(current_indices)
    
    # --- STEP 1: SMART RUIN ---
    # Find cameras to remove (those with lowest unique contribution)
    candidates_to_remove = []
    
    for cam_idx in current_indices:
        pts = coverage_matrix[:, cam_idx].indices
        # Unique contribution: Weight of points where count == 1 (only this camera)
        unique_mask = (counts[pts] == 1)
        unique_val = demand_weights[pts[unique_mask]].sum()
        candidates_to_remove.append((cam_idx, unique_val))
    
    # Sort ascending (lowest contribution first)
    candidates_to_remove.sort(key=lambda x: x[1])
    
    # Drop the bottom N (Ruin Percent)
    num_to_drop = int(NUM_CAMERAS_TO_SELECT * RUIN_PERCENT)
    dropped_cams = [x[0] for x in candidates_to_remove[:num_to_drop]]
    
    for cam in dropped_cams:
        current_set.remove(cam)
        
    # Recalculate coverage mask after removal
    temp_indices = list(current_set)
    sel_mat = coverage_matrix[:, temp_indices]
    temp_counts = np.array(sel_mat.sum(axis=1)).flatten()
    current_uncovered_mask = (temp_counts == 0)
    current_uncovered_weight = demand_weights.copy()
    current_uncovered_weight[~current_uncovered_mask] = 0 # Zero out already covered
    
    # --- STEP 2: GREEDY RECREATE ---
    # We need to pick 'num_to_drop' new cameras
    added_cams = []
    
    # To speed this up, we only check a subset of candidates or use matrix math
    # Here we do a fast matrix multiplication to find best gains
    # Gain = coverage_matrix.T @ current_uncovered_weight
    
    for _ in range(num_to_drop):
        # Calculate gains for ALL candidates against CURRENT uncovered weight
        # This is the "Greedy" step
        gains = coverage_matrix.T @ current_uncovered_weight
        
        # We must ignore candidates already in the set
        # Set their gain to -1
        gains[temp_indices] = -1
        gains[added_cams] = -1
        
        # Pick best
        best_cam = np.argmax(gains)
        
        if gains[best_cam] <= 0:
            break # No more gains possible
            
        added_cams.append(best_cam)
        
        # Update weights (remove covered points)
        new_covered_pts = coverage_matrix[:, best_cam].indices
        current_uncovered_weight[new_covered_pts] = 0
        
    return dropped_cams, added_cams

# ==============================
# 7. OPTIMIZATION LOOP
# ==============================
print(f"\n--- Starting Loop ({MAX_ITERATIONS} Iterations) ---")

history_iterations = [0]
history_obj_values = [G_BEST_SCORE]

for iteration in range(1, MAX_ITERATIONS + 1):
    
    for i in range(SWARM_SIZE):
        # Standard PSO Movement
        r1 = np.random.rand(num_cands)
        r2 = np.random.rand(num_cands)
        V[i] = W * V[i] + C1 * r1 * (P_BEST_POS[i] - X[i]) + C2 * r2 * (G_BEST_POS - X[i])
        X[i] = X[i] + V[i]
        
        # --- RUIN & RECREATE MUTATION ---
        # If we hit probability, we IGNORE velocity and do a smart repair
        if random.random() < MUTATION_PROB:
            # 1. Evaluate current pos to get counts
            _, curr_indices, counts = evaluate_particle(X[i])
            
            # 2. Run heuristic
            dropped, added = ruin_and_recreate(curr_indices, counts)
            
            # 3. Apply to Priority Vector
            if len(dropped) > 0:
                X[i, dropped] = 0.0  # Force out
                X[i, added] = 1.0    # Force in
        
        # Clamp
        X[i] = np.clip(X[i], 0.0, 1.0)
        
        # Eval
        score, indices, _ = evaluate_particle(X[i])
        
        if score > P_BEST_SCORES[i]:
            P_BEST_SCORES[i] = score
            P_BEST_POS[i] = X[i].copy()
            
        if score > G_BEST_SCORE:
            diff = score - G_BEST_SCORE
            if diff > 0.0001:
                print(f"  > IMPROVEMENT at Iter {iteration} (P{i}): +{diff:.4f} | Z: {score:,.4f}")
                G_BEST_SCORE = score
                G_BEST_POS = X[i].copy()
                G_BEST_INDICES = indices

    history_iterations.append(iteration)
    history_obj_values.append(G_BEST_SCORE)
    
    if iteration % 5 == 0:
        print(f"  Iter {iteration}/{MAX_ITERATIONS} | Best Z: {G_BEST_SCORE:,.4f}")

# ==============================
# 8. EXCEL SAVE (SAME AS BEFORE)
# ==============================
final_pct = (G_BEST_SCORE / TOTAL_DEMAND_WEIGHT) * 100
end_time = time.time()
run_time = end_time - start_time

print("-" * 40)
print(f"FINAL Z (PSO):      {G_BEST_SCORE:,.4f} ({final_pct:.2f}%)")
print(f"TOTAL RUN TIME:     {run_time:.2f} seconds")
print("-" * 40)

final_indices = G_BEST_INDICES
selected_matrix_final = coverage_matrix[:, final_indices]
final_coverage_counts = np.array(selected_matrix_final.sum(axis=1)).flatten()

df_results_cam = df_candidates.iloc[final_indices].copy()
df_results_cam = df_results_cam[['candidate_id', 'loc_id', 'x', 'y', 'dir_angle']]
df_results_cam.rename(columns={'dir_angle': 'direction_degrees'}, inplace=True)

df_results_dem = df_demand.copy()
df_results_dem['cameras_covering'] = final_coverage_counts 
df_results_dem['is_covered'] = np.where(df_results_dem['cameras_covering'] > 0, 'Yes', 'No')

with pd.ExcelWriter(OUTPUT_EXCEL_FILE) as writer:
    df_results_cam.to_excel(writer, sheet_name='Selected_Cameras', index=False)
    df_results_dem.to_excel(writer, sheet_name='Demand_Coverage', index=False)
print(f"Excel saved: {OUTPUT_EXCEL_FILE}")

# ==============================
# 9. HEATMAP (SAME AS BEFORE)
# ==============================
final_heatmap_grid = np.zeros((TARGET_ROWS, TARGET_COLS))
dem_x_flat = df_demand['x'].values
dem_y_flat = df_demand['y'].values

for i, count in enumerate(final_coverage_counts):
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

title_text = f"{HEATMAP_TITLE}\nCoverage: {final_pct:.2f}% | Run Time: {run_time:.1f}s"
ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches='tight')
plt.close()
print(f"Heatmap saved: {OUTPUT_IMAGE_FILE}")

# ==============================
# 10. PLOT (SAME AS BEFORE)
# ==============================
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(history_iterations, history_obj_values, marker='o', linestyle='-', color='#e74c3c', markersize=4, label='Global Best (Z)')

if history_obj_values:
    plt.annotate(f"Start: {history_obj_values[0]:,.0f}", (history_iterations[0], history_obj_values[0]), 
                 textcoords="offset points", xytext=(10, -10), ha='left', fontsize=9)
    plt.annotate(f"End: {history_obj_values[-1]:,.0f}", (history_iterations[-1], history_obj_values[-1]), 
                 textcoords="offset points", xytext=(-10, 10), ha='right', fontsize=9)

plot_title = f"PSO Convergence (100 Cams)\nFinal Coverage: {final_pct:.2f}% | Time: {run_time:.1f}s"
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
INPUT_FILE = "Final_Dataset_MCLP.xlsx"
SEED_FILE = "local_search_soln2.xlsx"  # CHANGE THIS for soln2 / soln3

DEMAND_SHEET = "demand"
CAMERA_SHEET = "camera"

# Outputs
OUTPUT_IMAGE_FILE = "pso_heatmap_soln2.png"
OUTPUT_PLOT_FILE = "pso_trajectory_plot_soln2.png"
OUTPUT_EXCEL_FILE = "pso_solution_soln2.xlsx"
HEATMAP_TITLE = "PSO Solution 2: 100 Cameras"

# Constraints
CAMERA_RADIUS = 5
FOV_HALF_ANGLE = 45
NUM_CAMERAS_TO_SELECT = 100

# Grid
TARGET_COLS = 80
TARGET_ROWS = 130
SUBTRACT_ONE = 0 
BATCH_SIZE = 1000 

# --- RUIN & RECREATE SETTINGS ---
SWARM_SIZE = 30           
MAX_ITERATIONS = 50       
W = 0.5                   
C1 = 1.5                  
C2 = 1.5   
RUIN_PERCENT = 0.10       # Re-optimize 10% (10 cameras) every time
MUTATION_PROB = 0.3       # 30% chance a particle undergoes Ruin & Recreate

print("--- RUIN & RECREATE PSO STARTED ---")
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
print("Coverage Matrix Computed.")

# ==============================
# 5. INITIALIZE SWARM
# ==============================
print(f"\n--- Initializing Swarm ---")

X = np.random.uniform(0.0, 0.4, (SWARM_SIZE, num_cands))

# Load Seed
seed_indices = []
try:
    print(f"Loading seed from: {SEED_FILE}")
    df_seed = pd.read_excel(SEED_FILE, sheet_name='Selected_Cameras')
    seed_indices = df_seed['candidate_id'].values[:NUM_CAMERAS_TO_SELECT].tolist()
    
    # Init Particle 0 exact
    X[0, seed_indices] = 1.0
    
    # Init others with variation
    for i in range(1, SWARM_SIZE):
        X[i, seed_indices] = np.random.uniform(0.8, 1.0, len(seed_indices))
        # Randomly remove 10% to let Ruin & Recreate fill them
        drop_indices = random.sample(seed_indices, int(NUM_CAMERAS_TO_SELECT * 0.1))
        X[i, drop_indices] = np.random.uniform(0.0, 0.2, len(drop_indices))
        
    print(f" >> Seed loaded and variations created.")
except Exception as e:
    print(f" >> ERROR loading seed: {e}. Using random start.")

V = np.random.uniform(-0.1, 0.1, (SWARM_SIZE, num_cands))

P_BEST_POS = X.copy()
P_BEST_SCORES = np.zeros(SWARM_SIZE)
G_BEST_POS = np.zeros(num_cands)
G_BEST_SCORE = -1.0
G_BEST_INDICES = []

def evaluate_particle(priority_vector):
    top_k_indices = np.argpartition(priority_vector, -NUM_CAMERAS_TO_SELECT)[-NUM_CAMERAS_TO_SELECT:]
    selected_matrix = coverage_matrix[:, top_k_indices]
    coverage_counts = np.array(selected_matrix.sum(axis=1)).flatten()
    covered_mask = coverage_counts > 0
    total_coverage = demand_weights[covered_mask].sum()
    return total_coverage, top_k_indices, coverage_counts

# Initial Eval
print("Evaluating initial swarm...")
for i in range(SWARM_SIZE):
    score, indices, _ = evaluate_particle(X[i])
    P_BEST_SCORES[i] = score
    if score > G_BEST_SCORE:
        G_BEST_SCORE = score
        G_BEST_POS = X[i].copy()
        G_BEST_INDICES = indices

print(f"Initial Global Best Z: {G_BEST_SCORE:,.4f}")

# ==============================
# 6. RUIN & RECREATE LOGIC
# ==============================

def ruin_and_recreate(current_indices, counts):
    """
    1. RUIN: Drop cameras that provide the least UNIQUE coverage.
    2. RECREATE: Greedily add cameras that cover the most UNCOVERED weight.
    """
    current_set = set(current_indices)
    
    # --- STEP 1: SMART RUIN ---
    # Find cameras to remove (those with lowest unique contribution)
    candidates_to_remove = []
    
    for cam_idx in current_indices:
        pts = coverage_matrix[:, cam_idx].indices
        # Unique contribution: Weight of points where count == 1 (only this camera)
        unique_mask = (counts[pts] == 1)
        unique_val = demand_weights[pts[unique_mask]].sum()
        candidates_to_remove.append((cam_idx, unique_val))
    
    # Sort ascending (lowest contribution first)
    candidates_to_remove.sort(key=lambda x: x[1])
    
    # Drop the bottom N (Ruin Percent)
    num_to_drop = int(NUM_CAMERAS_TO_SELECT * RUIN_PERCENT)
    dropped_cams = [x[0] for x in candidates_to_remove[:num_to_drop]]
    
    for cam in dropped_cams:
        current_set.remove(cam)
        
    # Recalculate coverage mask after removal
    temp_indices = list(current_set)
    sel_mat = coverage_matrix[:, temp_indices]
    temp_counts = np.array(sel_mat.sum(axis=1)).flatten()
    current_uncovered_mask = (temp_counts == 0)
    current_uncovered_weight = demand_weights.copy()
    current_uncovered_weight[~current_uncovered_mask] = 0 # Zero out already covered
    
    # --- STEP 2: GREEDY RECREATE ---
    # We need to pick 'num_to_drop' new cameras
    added_cams = []
    
    # To speed this up, we only check a subset of candidates or use matrix math
    # Here we do a fast matrix multiplication to find best gains
    # Gain = coverage_matrix.T @ current_uncovered_weight
    
    for _ in range(num_to_drop):
        # Calculate gains for ALL candidates against CURRENT uncovered weight
        # This is the "Greedy" step
        gains = coverage_matrix.T @ current_uncovered_weight
        
        # We must ignore candidates already in the set
        # Set their gain to -1
        gains[temp_indices] = -1
        gains[added_cams] = -1
        
        # Pick best
        best_cam = np.argmax(gains)
        
        if gains[best_cam] <= 0:
            break # No more gains possible
            
        added_cams.append(best_cam)
        
        # Update weights (remove covered points)
        new_covered_pts = coverage_matrix[:, best_cam].indices
        current_uncovered_weight[new_covered_pts] = 0
        
    return dropped_cams, added_cams

# ==============================
# 7. OPTIMIZATION LOOP
# ==============================
print(f"\n--- Starting Loop ({MAX_ITERATIONS} Iterations) ---")

history_iterations = [0]
history_obj_values = [G_BEST_SCORE]

for iteration in range(1, MAX_ITERATIONS + 1):
    
    for i in range(SWARM_SIZE):
        # Standard PSO Movement
        r1 = np.random.rand(num_cands)
        r2 = np.random.rand(num_cands)
        V[i] = W * V[i] + C1 * r1 * (P_BEST_POS[i] - X[i]) + C2 * r2 * (G_BEST_POS - X[i])
        X[i] = X[i] + V[i]
        
        # --- RUIN & RECREATE MUTATION ---
        # If we hit probability, we IGNORE velocity and do a smart repair
        if random.random() < MUTATION_PROB:
            # 1. Evaluate current pos to get counts
            _, curr_indices, counts = evaluate_particle(X[i])
            
            # 2. Run heuristic
            dropped, added = ruin_and_recreate(curr_indices, counts)
            
            # 3. Apply to Priority Vector
            if len(dropped) > 0:
                X[i, dropped] = 0.0  # Force out
                X[i, added] = 1.0    # Force in
        
        # Clamp
        X[i] = np.clip(X[i], 0.0, 1.0)
        
        # Eval
        score, indices, _ = evaluate_particle(X[i])
        
        if score > P_BEST_SCORES[i]:
            P_BEST_SCORES[i] = score
            P_BEST_POS[i] = X[i].copy()
            
        if score > G_BEST_SCORE:
            diff = score - G_BEST_SCORE
            if diff > 0.0001:
                print(f"  > IMPROVEMENT at Iter {iteration} (P{i}): +{diff:.4f} | Z: {score:,.4f}")
                G_BEST_SCORE = score
                G_BEST_POS = X[i].copy()
                G_BEST_INDICES = indices

    history_iterations.append(iteration)
    history_obj_values.append(G_BEST_SCORE)
    
    if iteration % 5 == 0:
        print(f"  Iter {iteration}/{MAX_ITERATIONS} | Best Z: {G_BEST_SCORE:,.4f}")

# ==============================
# 8. EXCEL SAVE (SAME AS BEFORE)
# ==============================
final_pct = (G_BEST_SCORE / TOTAL_DEMAND_WEIGHT) * 100
end_time = time.time()
run_time = end_time - start_time

print("-" * 40)
print(f"FINAL Z (PSO):      {G_BEST_SCORE:,.4f} ({final_pct:.2f}%)")
print(f"TOTAL RUN TIME:     {run_time:.2f} seconds")
print("-" * 40)

final_indices = G_BEST_INDICES
selected_matrix_final = coverage_matrix[:, final_indices]
final_coverage_counts = np.array(selected_matrix_final.sum(axis=1)).flatten()

df_results_cam = df_candidates.iloc[final_indices].copy()
df_results_cam = df_results_cam[['candidate_id', 'loc_id', 'x', 'y', 'dir_angle']]
df_results_cam.rename(columns={'dir_angle': 'direction_degrees'}, inplace=True)

df_results_dem = df_demand.copy()
df_results_dem['cameras_covering'] = final_coverage_counts 
df_results_dem['is_covered'] = np.where(df_results_dem['cameras_covering'] > 0, 'Yes', 'No')

with pd.ExcelWriter(OUTPUT_EXCEL_FILE) as writer:
    df_results_cam.to_excel(writer, sheet_name='Selected_Cameras', index=False)
    df_results_dem.to_excel(writer, sheet_name='Demand_Coverage', index=False)
print(f"Excel saved: {OUTPUT_EXCEL_FILE}")

# ==============================
# 9. HEATMAP (SAME AS BEFORE)
# ==============================
final_heatmap_grid = np.zeros((TARGET_ROWS, TARGET_COLS))
dem_x_flat = df_demand['x'].values
dem_y_flat = df_demand['y'].values

for i, count in enumerate(final_coverage_counts):
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

title_text = f"{HEATMAP_TITLE}\nCoverage: {final_pct:.2f}% | Run Time: {run_time:.1f}s"
ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches='tight')
plt.close()
print(f"Heatmap saved: {OUTPUT_IMAGE_FILE}")

# ==============================
# 10. PLOT (SAME AS BEFORE)
# ==============================
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(history_iterations, history_obj_values, marker='o', linestyle='-', color='#e74c3c', markersize=4, label='Global Best (Z)')

if history_obj_values:
    plt.annotate(f"Start: {history_obj_values[0]:,.0f}", (history_iterations[0], history_obj_values[0]), 
                 textcoords="offset points", xytext=(10, -10), ha='left', fontsize=9)
    plt.annotate(f"End: {history_obj_values[-1]:,.0f}", (history_iterations[-1], history_obj_values[-1]), 
                 textcoords="offset points", xytext=(-10, 10), ha='right', fontsize=9)

plot_title = f"PSO Convergence (100 Cams)\nFinal Coverage: {final_pct:.2f}% | Time: {run_time:.1f}s"
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
INPUT_FILE = "Final_Dataset_MCLP.xlsx"
SEED_FILE = "local_search_soln3.xlsx"  # CHANGE THIS for soln2 / soln3

DEMAND_SHEET = "demand"
CAMERA_SHEET = "camera"

# Outputs
OUTPUT_IMAGE_FILE = "pso_heatmap_soln3.png"
OUTPUT_PLOT_FILE = "pso_trajectory_plot_soln3.png"
OUTPUT_EXCEL_FILE = "pso_solution_soln3.xlsx"
HEATMAP_TITLE = "PSO Solution 3: 100 Cameras"

# Constraints
CAMERA_RADIUS = 5
FOV_HALF_ANGLE = 45
NUM_CAMERAS_TO_SELECT = 100

# Grid
TARGET_COLS = 80
TARGET_ROWS = 130
SUBTRACT_ONE = 0 
BATCH_SIZE = 1000 

# --- RUIN & RECREATE SETTINGS ---
SWARM_SIZE = 30           
MAX_ITERATIONS = 50       
W = 0.5                   
C1 = 1.5                  
C2 = 1.5   
RUIN_PERCENT = 0.10       # Re-optimize 10% (10 cameras) every time
MUTATION_PROB = 0.3       # 30% chance a particle undergoes Ruin & Recreate

print("--- RUIN & RECREATE PSO STARTED ---")
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
print("Coverage Matrix Computed.")

# ==============================
# 5. INITIALIZE SWARM
# ==============================
print(f"\n--- Initializing Swarm ---")

X = np.random.uniform(0.0, 0.4, (SWARM_SIZE, num_cands))

# Load Seed
seed_indices = []
try:
    print(f"Loading seed from: {SEED_FILE}")
    df_seed = pd.read_excel(SEED_FILE, sheet_name='Selected_Cameras')
    seed_indices = df_seed['candidate_id'].values[:NUM_CAMERAS_TO_SELECT].tolist()
    
    # Init Particle 0 exact
    X[0, seed_indices] = 1.0
    
    # Init others with variation
    for i in range(1, SWARM_SIZE):
        X[i, seed_indices] = np.random.uniform(0.8, 1.0, len(seed_indices))
        # Randomly remove 10% to let Ruin & Recreate fill them
        drop_indices = random.sample(seed_indices, int(NUM_CAMERAS_TO_SELECT * 0.1))
        X[i, drop_indices] = np.random.uniform(0.0, 0.2, len(drop_indices))
        
    print(f" >> Seed loaded and variations created.")
except Exception as e:
    print(f" >> ERROR loading seed: {e}. Using random start.")

V = np.random.uniform(-0.1, 0.1, (SWARM_SIZE, num_cands))

P_BEST_POS = X.copy()
P_BEST_SCORES = np.zeros(SWARM_SIZE)
G_BEST_POS = np.zeros(num_cands)
G_BEST_SCORE = -1.0
G_BEST_INDICES = []

def evaluate_particle(priority_vector):
    top_k_indices = np.argpartition(priority_vector, -NUM_CAMERAS_TO_SELECT)[-NUM_CAMERAS_TO_SELECT:]
    selected_matrix = coverage_matrix[:, top_k_indices]
    coverage_counts = np.array(selected_matrix.sum(axis=1)).flatten()
    covered_mask = coverage_counts > 0
    total_coverage = demand_weights[covered_mask].sum()
    return total_coverage, top_k_indices, coverage_counts

# Initial Eval
print("Evaluating initial swarm...")
for i in range(SWARM_SIZE):
    score, indices, _ = evaluate_particle(X[i])
    P_BEST_SCORES[i] = score
    if score > G_BEST_SCORE:
        G_BEST_SCORE = score
        G_BEST_POS = X[i].copy()
        G_BEST_INDICES = indices

print(f"Initial Global Best Z: {G_BEST_SCORE:,.4f}")

# ==============================
# 6. RUIN & RECREATE LOGIC
# ==============================

def ruin_and_recreate(current_indices, counts):
    """
    1. RUIN: Drop cameras that provide the least UNIQUE coverage.
    2. RECREATE: Greedily add cameras that cover the most UNCOVERED weight.
    """
    current_set = set(current_indices)
    
    # --- STEP 1: SMART RUIN ---
    # Find cameras to remove (those with lowest unique contribution)
    candidates_to_remove = []
    
    for cam_idx in current_indices:
        pts = coverage_matrix[:, cam_idx].indices
        # Unique contribution: Weight of points where count == 1 (only this camera)
        unique_mask = (counts[pts] == 1)
        unique_val = demand_weights[pts[unique_mask]].sum()
        candidates_to_remove.append((cam_idx, unique_val))
    
    # Sort ascending (lowest contribution first)
    candidates_to_remove.sort(key=lambda x: x[1])
    
    # Drop the bottom N (Ruin Percent)
    num_to_drop = int(NUM_CAMERAS_TO_SELECT * RUIN_PERCENT)
    dropped_cams = [x[0] for x in candidates_to_remove[:num_to_drop]]
    
    for cam in dropped_cams:
        current_set.remove(cam)
        
    # Recalculate coverage mask after removal
    temp_indices = list(current_set)
    sel_mat = coverage_matrix[:, temp_indices]
    temp_counts = np.array(sel_mat.sum(axis=1)).flatten()
    current_uncovered_mask = (temp_counts == 0)
    current_uncovered_weight = demand_weights.copy()
    current_uncovered_weight[~current_uncovered_mask] = 0 # Zero out already covered
    
    # --- STEP 2: GREEDY RECREATE ---
    # We need to pick 'num_to_drop' new cameras
    added_cams = []
    
    # To speed this up, we only check a subset of candidates or use matrix math
    # Here we do a fast matrix multiplication to find best gains
    # Gain = coverage_matrix.T @ current_uncovered_weight
    
    for _ in range(num_to_drop):
        # Calculate gains for ALL candidates against CURRENT uncovered weight
        # This is the "Greedy" step
        gains = coverage_matrix.T @ current_uncovered_weight
        
        # We must ignore candidates already in the set
        # Set their gain to -1
        gains[temp_indices] = -1
        gains[added_cams] = -1
        
        # Pick best
        best_cam = np.argmax(gains)
        
        if gains[best_cam] <= 0:
            break # No more gains possible
            
        added_cams.append(best_cam)
        
        # Update weights (remove covered points)
        new_covered_pts = coverage_matrix[:, best_cam].indices
        current_uncovered_weight[new_covered_pts] = 0
        
    return dropped_cams, added_cams

# ==============================
# 7. OPTIMIZATION LOOP
# ==============================
print(f"\n--- Starting Loop ({MAX_ITERATIONS} Iterations) ---")

history_iterations = [0]
history_obj_values = [G_BEST_SCORE]

for iteration in range(1, MAX_ITERATIONS + 1):
    
    for i in range(SWARM_SIZE):
        # Standard PSO Movement
        r1 = np.random.rand(num_cands)
        r2 = np.random.rand(num_cands)
        V[i] = W * V[i] + C1 * r1 * (P_BEST_POS[i] - X[i]) + C2 * r2 * (G_BEST_POS - X[i])
        X[i] = X[i] + V[i]
        
        # --- RUIN & RECREATE MUTATION ---
        # If we hit probability, we IGNORE velocity and do a smart repair
        if random.random() < MUTATION_PROB:
            # 1. Evaluate current pos to get counts
            _, curr_indices, counts = evaluate_particle(X[i])
            
            # 2. Run heuristic
            dropped, added = ruin_and_recreate(curr_indices, counts)
            
            # 3. Apply to Priority Vector
            if len(dropped) > 0:
                X[i, dropped] = 0.0  # Force out
                X[i, added] = 1.0    # Force in
        
        # Clamp
        X[i] = np.clip(X[i], 0.0, 1.0)
        
        # Eval
        score, indices, _ = evaluate_particle(X[i])
        
        if score > P_BEST_SCORES[i]:
            P_BEST_SCORES[i] = score
            P_BEST_POS[i] = X[i].copy()
            
        if score > G_BEST_SCORE:
            diff = score - G_BEST_SCORE
            if diff > 0.0001:
                print(f"  > IMPROVEMENT at Iter {iteration} (P{i}): +{diff:.4f} | Z: {score:,.4f}")
                G_BEST_SCORE = score
                G_BEST_POS = X[i].copy()
                G_BEST_INDICES = indices

    history_iterations.append(iteration)
    history_obj_values.append(G_BEST_SCORE)
    
    if iteration % 5 == 0:
        print(f"  Iter {iteration}/{MAX_ITERATIONS} | Best Z: {G_BEST_SCORE:,.4f}")

# ==============================
# 8. EXCEL SAVE (SAME AS BEFORE)
# ==============================
final_pct = (G_BEST_SCORE / TOTAL_DEMAND_WEIGHT) * 100
end_time = time.time()
run_time = end_time - start_time

print("-" * 40)
print(f"FINAL Z (PSO):      {G_BEST_SCORE:,.4f} ({final_pct:.2f}%)")
print(f"TOTAL RUN TIME:     {run_time:.2f} seconds")
print("-" * 40)

final_indices = G_BEST_INDICES
selected_matrix_final = coverage_matrix[:, final_indices]
final_coverage_counts = np.array(selected_matrix_final.sum(axis=1)).flatten()

df_results_cam = df_candidates.iloc[final_indices].copy()
df_results_cam = df_results_cam[['candidate_id', 'loc_id', 'x', 'y', 'dir_angle']]
df_results_cam.rename(columns={'dir_angle': 'direction_degrees'}, inplace=True)

df_results_dem = df_demand.copy()
df_results_dem['cameras_covering'] = final_coverage_counts 
df_results_dem['is_covered'] = np.where(df_results_dem['cameras_covering'] > 0, 'Yes', 'No')

with pd.ExcelWriter(OUTPUT_EXCEL_FILE) as writer:
    df_results_cam.to_excel(writer, sheet_name='Selected_Cameras', index=False)
    df_results_dem.to_excel(writer, sheet_name='Demand_Coverage', index=False)
print(f"Excel saved: {OUTPUT_EXCEL_FILE}")

# ==============================
# 9. HEATMAP (SAME AS BEFORE)
# ==============================
final_heatmap_grid = np.zeros((TARGET_ROWS, TARGET_COLS))
dem_x_flat = df_demand['x'].values
dem_y_flat = df_demand['y'].values

for i, count in enumerate(final_coverage_counts):
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

title_text = f"{HEATMAP_TITLE}\nCoverage: {final_pct:.2f}% | Run Time: {run_time:.1f}s"
ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches='tight')
plt.close()
print(f"Heatmap saved: {OUTPUT_IMAGE_FILE}")

# ==============================
# 10. PLOT (SAME AS BEFORE)
# ==============================
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(history_iterations, history_obj_values, marker='o', linestyle='-', color='#e74c3c', markersize=4, label='Global Best (Z)')

if history_obj_values:
    plt.annotate(f"Start: {history_obj_values[0]:,.0f}", (history_iterations[0], history_obj_values[0]), 
                 textcoords="offset points", xytext=(10, -10), ha='left', fontsize=9)
    plt.annotate(f"End: {history_obj_values[-1]:,.0f}", (history_iterations[-1], history_obj_values[-1]), 
                 textcoords="offset points", xytext=(-10, 10), ha='right', fontsize=9)

plot_title = f"PSO Convergence (100 Cams)\nFinal Coverage: {final_pct:.2f}% | Time: {run_time:.1f}s"
plt.title(plot_title, fontsize=14, fontweight='bold')
plt.xlabel("Iteration Number")
plt.ylabel("Total Weighted Coverage (Z)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PLOT_FILE)
plt.close()

print(f"Trajectory plot saved: {OUTPUT_PLOT_FILE}")