import cv2
import numpy as np
from csrc.SamplingNew.sampling import sampling

ImagePath = "csrc/SamplingNew/Images/af39ab4f7a6b9f95d79b29a72cce839c.jpg"

fil_img, center_x, center_y, radius_min, radius_max = sampling.prepare(ImagePath)
StartAngle = np.pi / 2 - 1.0406

print(f"StartAngle: {StartAngle}")
print(f"Center: {center_x}, {center_y}")
print(f"Radius range: {radius_min} - {radius_max}")

for OrderNumber in [0, 8]:
    Shift = np.pi / 32
    Start_Angle = StartAngle - (OrderNumber * (np.pi / 8)) - Shift
    End_Angle = StartAngle - ((OrderNumber + 1) *(np.pi / 8)) + Shift
    print(f"Order: {OrderNumber}, Start Angle: {Start_Angle}, End Angle: {End_Angle}")
    
    samples_r, samples_g, samples_b = [], [], []
    for radius in range(radius_min, radius_max + 1, 1):
        for angle in np.arange(End_Angle, Start_Angle, np.pi / (radius / 8)):
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            b, g, r = fil_img[int(y), int(x)]
            samples_r.append(r)
            samples_g.append(g)
            samples_b.append(b)
    print(f"Order {OrderNumber} direct pixel color: {int(np.mean(samples_r))}, {int(np.mean(samples_g))}, {int(np.mean(samples_b))}")
