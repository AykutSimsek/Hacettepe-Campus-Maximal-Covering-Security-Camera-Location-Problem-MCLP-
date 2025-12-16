import pandas as pd
import numpy as np

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