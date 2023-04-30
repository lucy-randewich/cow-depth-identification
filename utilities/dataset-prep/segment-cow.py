import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from pathlib import Path
import skimage

def colour_vis(img):
    stretch = skimage.exposure.rescale_intensity(img, in_range='image', out_range=(0,255)).astype(np.uint8)
    stretch = cv2.merge([stretch,stretch,stretch])
    color1 = (0, 0, 255)     #red
    color2 = (0, 165, 255)   #orange
    color3 = (0, 255, 255)   #yellow
    color4 = (255, 255, 0)   #cyan
    color5 = (255, 0, 0)     #blue
    color6 = (128, 64, 64)   #violet
    colorArr = np.array([[color1, color2, color3, color4, color5, color6]], dtype=np.uint8)
    lut = cv2.resize(colorArr, (256,1), interpolation = cv2.INTER_LINEAR)
    result = cv2.LUT(stretch, lut)
    return result

# Read in depth image
img = cv2.imread("depth.png", cv2.IMREAD_ANYDEPTH)

# Remove values outside of cow range
lower = 2000
upper = 3400
upper_mask = img < upper
new = img.copy()
new[~upper_mask] = 0
lower_mask = img > lower
new[~lower_mask] = 0

# Remove fence and other background objects
background = cv2.imread("background.png", cv2.IMREAD_ANYDEPTH)
upper_mask = background < 3800
new_background = background.copy()
new_background[~upper_mask] = 0
lower_mask = background > 1000
new_background[~lower_mask] = 0
background_mask = new_background > 0
just_cows = new.copy()
just_cows[background_mask] = 0

# Do some filtering to get just the cow blob(s)
from scipy.ndimage import median_filter
smooth_img = median_filter(just_cows, size=4)
smooth_img2_mask = smooth_img > 0
smooth_img2 = smooth_img.copy()
smooth_img2[smooth_img2_mask] = 2000

# Find cow blob regions
from skimage.measure import label, regionprops
label_image = label(smooth_img2)
boximg = img.copy()
num_cows = 0
cows = []
for region in regionprops(label_image):
    if region.area >= 8000 and region.area <= 22000:
        num_cows += 1
        boximg = cv2.rectangle(boximg, (region.bbox[3], region.bbox[0]), (region.bbox[1], region.bbox[2]), (0, 0, 0), 2)
        # Use mask to grab and save correct part of original depth image
        cropped_image = img[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]]
        cows.append(cropped_image)


# Save image(s)
print(f"Number of cows found in image: {num_cows}")

cv2.imwrite('depth.png', img)
cv2.imwrite('depth-colour.png', colour_vis(img))
cv2.imwrite('cow-boxes.png', colour_vis(boximg))
cv2.imwrite('/segmentation.png', colour_vis(smooth_img2))

for i, cow in enumerate(cows):
    cv2.imwrite('/cow{i}.png', cow)
    cv2.imwrite('/cow{i}-colour.png', colour_vis(cow))