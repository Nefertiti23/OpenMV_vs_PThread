import numpy as np
import os
from PIL import Image

# the output directory
os.makedirs("./images/archive", exist_ok=True)

# Load your .npy file (update the filename to match your downloaded file)
# Typically colorization datasets have 'L' (lightness/grayscale) and 'ab' channels or RGB arrays
print("Loading .npy file...")
data = np.load("./images/archive/ab/ab/ab3.npy") # Change to your actual .npy filename

print(f"Dataset shape: {data.shape}")

# Loop through and save each image index
for i in range(len(data)):
    img_array = data[i]

    # Normalize or scale if necessary depending on dataset format (e.g., 0-1 float to 0-255 uint8)
    if img_array.dtype != np.uint8:
        img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)

    img = Image.fromarray(img_array)

    if img.mode in ("LA", "RGBA"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")

    img.save(f"./images/archive/img3_{i:05d}.jpg")

print("Successfully unpacked all images into ./images/")