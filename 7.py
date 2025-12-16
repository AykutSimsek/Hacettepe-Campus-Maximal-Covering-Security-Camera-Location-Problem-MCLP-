import pandas as pd
import numpy as np
import math

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
