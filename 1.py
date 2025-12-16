import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

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
