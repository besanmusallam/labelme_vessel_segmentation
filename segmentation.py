import json
import numpy as np
import cv2

# Load the annotation JSON file
json_path = "./images/Im(5)_300.json"  
output_path = "./masks/Im(5)_300.png"

with open(json_path) as f:
    data = json.load(f)

# Get image dimensions from JSON
image_shape = (data["imageHeight"], data["imageWidth"])

# Extract polygons
outer_polygons = []
inner_polygons = []

for shape in data["shapes"]:
    label = shape.get("label", "").lower()
    points = np.array(shape["points"], dtype=np.int32)

    # Assuming "outer" and "inner" are keywords in labels for differentiation
    if "out" in label:
        outer_polygons.append(points)
    elif "ingo" in label:
        inner_polygons.append(points)

# Ensure at least one outer polygon exists
if len(outer_polygons) == 0:
    raise ValueError("No outer polygons found!")

# Create an empty mask
mask = np.zeros(image_shape, dtype=np.uint8)

# Fill outer polygons with class 1 (initially setting entire region as 1)
cv2.fillPoly(mask, outer_polygons, 1)

# Fill inner polygons with class 0 (to remove the center)
cv2.fillPoly(mask, inner_polygons, 0)
cv2.imwrite(output_path, mask * 255)  # Save as binary mask

print(f"Mask saved to {output_path}")
