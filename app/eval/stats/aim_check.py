from pysaliency import AIM
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Replace with actual path to AIM model folder
model = AIM(location="app/models/pysal")

# Load test image (any RGB image works)
img = cv2.imread("data/COCO/val2017/000000000139.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Compute saliency
sal_map = model.saliency_map(img_rgb)

# Show results
plt.imshow(sal_map, cmap='gray')
plt.title("AIM Output")
plt.colorbar()
plt.show()
