# Code adapted from stackoverflow answer by user fmw42: https://stackoverflow.com/questions/67678048/whats-the-proper-way-to-colorize-a-16-bit-depth-image (Accessed 30/04/2023)

import numpy as np
import skimage.exposure
import cv2

# Read in depth image
img = cv2.imread("cow.png", cv2.IMREAD_ANYDEPTH)

# stretch to full dynamic range
stretch = skimage.exposure.rescale_intensity(img, in_range='image', out_range=(0,255)).astype(np.uint8)

# convert to 3 channels
stretch = cv2.merge([stretch,stretch,stretch])

# define colors
color1 = (0, 0, 255)     #red
color2 = (0, 165, 255)   #orange
color3 = (0, 255, 255)   #yellow
color4 = (255, 255, 0)   #cyan
color5 = (255, 0, 0)     #blue
color6 = (128, 64, 64)   #violet
colorArr = np.array([[color1, color2, color3, color4, color5, color6]], dtype=np.uint8)

# resize and apply lut
lut = cv2.resize(colorArr, (256,1), interpolation = cv2.INTER_LINEAR)
result = cv2.LUT(stretch, lut)

# create gradient image
grad = np.linspace(0, 255, 512, dtype=np.uint8)
grad = np.tile(grad, (20,1))
grad = cv2.merge([grad,grad,grad])

# apply lut to gradient for viewing
grad_colored = cv2.LUT(grad, lut)

# save result
cv2.imwrite('rgb.png', result)

# display result
cv2.imshow('RESULT', result)
cv2.waitKey()
cv2.destroyAllWindows()