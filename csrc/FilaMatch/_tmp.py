import cv2
import numpy as np

import matplotlib.pyplot as plt

input_image_path = "csrc/FilaMatch/circular_reflect_1.png"
output_image_path = "csrc/FilaMatch/_tmp_output_1.png"

img = cv2.imread(input_image_path)
if img is None:
    raise FileNotFoundError(f"无法读取图像: {input_image_path}")

res_list = []

min_kernel_size = 41
max_kernel_size = 43

for kernel_size in range(min_kernel_size, max_kernel_size, 2):

    kernel = np.zeros((kernel_size, kernel_size)) - 1
    kernel[kernel_size // 2, kernel_size // 2] = kernel_size**2 - 1

    result = cv2.filter2D(img, cv2.CV_32F, kernel)

    result = result[kernel_size // 2 : -kernel_size // 2, kernel_size // 2 : -kernel_size // 2]

    result = np.abs(result)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    result = np.uint8(result)

    res_list.append(result)

img = res_list[0]
cv2.imwrite(output_image_path, img)

# num_col = 5

# num_row = len(res_list) // num_col

# plt.figure(figsize=(num_col*1.5, num_row*1.5))

# for i in range(len(res_list)):
#     plt.subplot(num_row, num_col, i + 1)
#     plt.imshow(res_list[i], cmap="gray")
#     plt.axis("off")

# plt.tight_layout(pad=0, w_pad=0, h_pad=0)
# plt.show()