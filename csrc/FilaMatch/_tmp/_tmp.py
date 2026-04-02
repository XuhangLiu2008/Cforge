import cv2
import numpy as np

import matplotlib.pyplot as plt

input_image_path = "csrc/FilaMatch/circular_reflect_1.png"
output_image_path = "csrc/FilaMatch/_tmp_output_1.png"

img = cv2.imread(input_image_path)
if img is None:
    raise FileNotFoundError(f"无法读取图像: {input_image_path}")

gauss_kernel_size = 121
laplace_kernel_size = 121

gauss_kernel = np.zeros((laplace_kernel_size, laplace_kernel_size)) + 1

laplace_kernel = np.zeros((laplace_kernel_size, laplace_kernel_size)) - 1
laplace_kernel[laplace_kernel_size // 2, laplace_kernel_size // 2] = laplace_kernel_size**2 - 1

img = cv2.filter2D(img, cv2.CV_32F, gauss_kernel)
img = cv2.filter2D(img, cv2.CV_32F, laplace_kernel)

img = img[10 : -10, 10 : -10]

img = np.abs(img)
img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
img = np.uint8(img)

cv2.imwrite(output_image_path, img)