import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
import time
import gurobipy as gp
from gurobipy import GRB

# ==============================
# 1. SETTINGS
# ==============================
INPUT_FILE = "Final_Dataset_MCLP.xlsx"
DEMAND_SHEET = "demand"
CAMERA_SHEET = "camera"

OUTPUT_IMAGE_FILE = "gurobi_optimal_heatmap.png"
OUTPUT_EXCEL_FILE = "gurobi_optimal_solution.xlsx"
HEATMAP_TITLE = "Global Optimal Solution (Gurobi)"

# --- Model Parameters ---
# "p" in Equation (7): The maximum number of cameras allowed
P_MAX_CAMERAS = 100 

# Constraints
CAMERA_RADIUS = 5
FOV_HALF_ANGLE = 45

# Grid Settings
TARGET_COLS = 80
TARGET_ROWS = 130
SUBTRACT_ONE = 0 
BATCH_SIZE = 1000 

print("--- GUROBI OPTIMIZATION STARTED ---")
start_time = time.time()

# ==============================
# 2. LOAD DATA (Nodes n ∈ N)
# ==============================
df_demand = pd.read_excel(INPUT_FILE, sheet_name=DEMAND_SHEET)
df_demand.columns = [c.strip().lower() for c in df_demand.columns]
df_demand = df_demand[df_demand['weight'] > 0].copy()
df_demand['x'] = df_demand['x'] - SUBTRACT_ONE
df_demand['y'] = df_demand['y'] - SUBTRACT_ONE

# n: Index for demand nodes
df_demand['n'] = range(len(df_demand)) 
N_set = df_demand['n'].values
weights_w = df_demand['weight'].values  # w_n in Eq (6)

TOTAL_DEMAND_WEIGHT = df_demand['weight'].sum()

print(f"Set N (Demand Nodes): {len(N_set)} | Total Weight W: {TOTAL_DEMAND_WEIGHT:,.4f}")

# ==============================
# 3. GENERATE CANDIDATES (x_{cd})
# ==============================
# Load locations (c ∈ C)
df_cam_locs = pd.read_excel(INPUT_FILE, sheet_name=CAMERA_SHEET)
df_cam_locs.columns = [c.strip().lower() for c in df_cam_locs.columns]
df_cam_locs['x'] = df_cam_locs['x'] - SUBTRACT_ONE
df_cam_locs['y'] = df_cam_locs['y'] - SUBTRACT_ONE

# Generate all pairs (c, d) where c=location, d=direction
candidates_list = []
directions_D = [0, 90, 180, 270] # Set D

for c_idx, row in df_cam_locs.iterrows():
    for d_angle in directions_D:
        candidates_list.append({
            'loc_id': c_idx,         # c
            'dir_angle': d_angle,    # d
            'x': row['x'],
            'y': row['y']
        })

df_candidates = pd.DataFrame(candidates_list)
# Assign unique ID for the flattened variable x_{cd}
df_candidates['cd_id'] = range(len(df_candidates)) 
num_candidates = len(df_candidates)

print(f"Set C (Locations): {len(df_cam_locs)}")
print(f"Set D (Directions): {len(directions_D)}")
print(f"Total Decision Variables x_cd: {num_candidates}")

# ==============================
# 4. COMPUTE COVERAGE PARAMETERS (a_{ncd})
# ==============================
print("Computing coverage parameter a_{ncd}...")

# Mapping: n -> list of cd_ids that cover it
# This corresponds to the non-zero entries of a_{ncd} for each n
# Used to build Constraint (8) efficiently
nodes_covered_by = {n: [] for n in N_set}

dem_x = df_demand['x'].values[:, np.newaxis]
dem_y = df_demand['y'].values[:, np.newaxis]

for start_idx in range(0, num_candidates, BATCH_SIZE):
    end_idx = min(start_idx + BATCH_SIZE, num_candidates)
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
    
    # Coverage logic
    mask = (distances <= CAMERA_RADIUS) & (angle_diff <= FOV_HALF_ANGLE) & (distances > 0)
    
    # Populate the sparse mapping
    covered_indices = np.argwhere(mask)
    
    for n_idx, local_cd_idx in covered_indices:
        global_cd_id = start_idx + local_cd_idx
        nodes_covered_by[n_idx].append(global_cd_id)

print("Parameter a_{ncd} computed.")

# ==============================
# 5. BUILD GUROBI MODEL
# ==============================
try:
    # Create a new model
    m = gp.Model("MCLP_Camera_Optimization")
    
    # --- VARIABLES ---
    # x[j] corresponds to x_{cd} in Eq (9)
    # y[n] corresponds to y_n in Eq (9)
    print("Adding variables to model...")
    x = m.addVars(num_candidates, vtype=GRB.BINARY, name="x")
    y = m.addVars(len(N_set), vtype=GRB.BINARY, name="y")
    
    # --- CONSTRAINTS ---
    
    # Eq (7): Sum(x_cd) <= p
    # "at most the specified value p"
    print(f"Adding Constraint Eq(7): Budget p <= {P_MAX_CAMERAS}...")
    m.addConstr(x.sum() <= P_MAX_CAMERAS, name="Budget")
    
    # Eq (8): y_n <= Sum(a_{ncd} * x_{cd})
    # Implemented as: y[n] <= sum(x[j] for j covering n)
    print("Adding Constraint Eq(8): Coverage logic...")
    for n in N_set:
        covering_candidates = nodes_covered_by[n]
        if not covering_candidates:
            # Cannot be covered
            m.addConstr(y[n] == 0, name=f"Cover_{n}")
        else:
            # Use quicksum for efficiency
            m.addConstr(y[n] <= gp.quicksum(x[j] for j in covering_candidates), name=f"Cover_{n}")
            
    # --- OBJECTIVE ---
    # Eq (6): Maximize Z = Sum(w_n * y_n)
    print("Setting Objective Eq(6)...")
    # Use linexpr for fast objective setting
    obj_expr = gp.quicksum(weights_w[n] * y[n] for n in N_set)
    m.setObjective(obj_expr, GRB.MAXIMIZE)
    
    # --- PARAMETERS ---
    # Set a time limit (e.g., 300 seconds) if needed, or run to optimality
    m.setParam('TimeLimit', 600)  # 10 minutes limit
    m.setParam('MIPGap', 0.001)   # Stop if gap < 0.1%
    
    # ==============================
    # 6. SOLVE
    # ==============================
    print(f"\nSolving model with {num_candidates + len(N_set)} variables...")
    m.optimize()
    
    # ==============================
    # 7. EXTRACT RESULTS
    # ==============================
    if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
        print(f"\n Optimal Solution Found!")
        final_z = m.objVal
        final_pct = (final_z / TOTAL_DEMAND_WEIGHT) * 100
        print(f"Objective Value (Z): {final_z:,.4f}")
        print(f"Coverage Percentage: {final_pct:.2f}%")
        
        # Get selected camera indices
        selected_indices = [j for j in range(num_candidates) if x[j].X > 0.5]
        print(f"Selected {len(selected_indices)} cameras.")
        
        # Re-calculate coverage for visualization
        final_coverage_counts = np.zeros(len(df_demand))
        for n in N_set:
            if y[n].X > 0.5:
                # Count specifically how many cover it (for heatmap intensity)
                count = 0
                for cd_id in nodes_covered_by[n]:
                    if cd_id in selected_indices:
                        count += 1
                final_coverage_counts[n] = count
                
        # Save Results
        run_time_seconds = time.time() - start_time
        print(f"Total Run Time: {run_time_seconds:.2f} seconds")
        
        # Prepare Excel
        df_results_cam = df_candidates.iloc[selected_indices].copy()
        df_results_cam = df_results_cam[['cd_id', 'loc_id', 'x', 'y', 'dir_angle']]
        df_results_cam.rename(columns={'dir_angle': 'direction_degrees'}, inplace=True)

        df_results_dem = df_demand.copy()
        df_results_dem['cameras_covering'] = final_coverage_counts 
        df_results_dem['is_covered'] = np.where(df_results_dem['cameras_covering'] > 0, 'Yes', 'No')

        with pd.ExcelWriter(OUTPUT_EXCEL_FILE) as writer:
            df_results_cam.to_excel(writer, sheet_name='Selected_Cameras', index=False)
            df_results_dem.to_excel(writer, sheet_name='Demand_Coverage', index=False)
        print(f"Excel saved: {OUTPUT_EXCEL_FILE}")
        
        # Plot Heatmap
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

        title_text = f"{HEATMAP_TITLE}\nCoverage: {final_pct:.2f}% | Solved in: {run_time_seconds:.1f}s"
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved: {OUTPUT_IMAGE_FILE}")
        
    else:
        print("Optimization was stopped with status " + str(m.status))

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')