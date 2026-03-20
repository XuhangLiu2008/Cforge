import cv2
import numpy as np
from csrc.SamplingNew.sampling import sampling
ImagePath = "csrc/SamplingNew/Images/af39ab4f7a6b9f95d79b29a72cce839c.jpg"
fil_img, center_x, center_y, radius_min, radius_max = sampling.prepare(ImagePath)
StartAngle = np.pi / 2 - 1.0406

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(fil_img, cv2.COLOR_BGR2RGB))
plt.scatter([center_x], [center_y], c='r')

for OrderNumber in range(16):
    Shift = np.pi / 32
    Start_Angle = StartAngle - (OrderNumber * (np.pi / 8)) - Shift
    End_Angle = StartAngle - ((OrderNumber + 1) *(np.pi / 8)) + Shift
    for radius in range(radius_min, radius_max + 1, 50):
        for angle in np.arange(End_Angle, Start_Angle, np.pi / (radius / 8)):
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            plt.scatter([x], [y], c='w', s=1)
plt.savefig('vis.png')
