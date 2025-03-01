import json
import numpy as np
import cv2

# Load the annotation JSON file
json_path = "./Image__36.json"  # Change this to your actual JSON file
with open(json_path) as f:
    data = json.load(f)

# Get image dimensions from JSON
image_shape = (data["imageHeight"], data["imageWidth"])

# Extract polygons
polygons = []
for shape in data["shapes"]:
    polygons.append(np.array(shape["points"], dtype=np.int32))

# Ensure there are exactly two polygons (inner & outer)
if len(polygons) != 2:
    raise ValueError("Expected exactly two polygons (inner and outer). Found: {}".format(len(polygons)))

# Create an empty mask
mask = np.zeros(image_shape, dtype=np.uint8)

# Fill outer polygon with class 1
cv2.fillPoly(mask, [polygons[1]], 1)

# Fill inner polygon with class 0 (removing the center)
cv2.fillPoly(mask, [polygons[0]], 0)

# Save the mask as an image (without displaying it)
output_path = "./segmentation_mask.png"
cv2.imwrite(output_path, mask * 255)  # Save as binary mask

print(f"Mask saved to {output_path}")
