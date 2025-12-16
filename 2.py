import pandas as pd
import numpy as np

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
