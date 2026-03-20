import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import scipy

KMean_counter = 0

class sampling:
    def prepare(img_path):
        fil_img = cv2.imread(img_path)
        if fil_img is None:
            print("Error: Could not load image.")
        else:
            print("Image loaded successfully.")
            print(f"Image shape: {fil_img.shape}")
        
        if fil_img.shape[2] != 3:
            print("Error: Not a RGB image.")

        img_x = fil_img.shape[1]
        img_y = fil_img.shape[0]

        center_x = img_x // 2
        center_y = img_y // 2
        radius = (img_x + img_y) // 4
        radius_min = int(radius * (1/4))
        radius_max = int(radius * (3/4))

        return fil_img, center_x, center_y, radius_min, radius_max
    
    def SamplePoint(fil_img, r, angle, center_x, center_y):
        x = center_x + r * np.cos(angle)
        y = center_y + r * np.sin(angle)

        # When the points are integers, or not, use powered averages
        x_proportion_upper = x - int(x)
        y_proportion_upper = y - int(y)
        x_proportion_lower = 1 - x_proportion_upper
        y_proportion_lower = 1 - y_proportion_upper

        b1, g1, r1 = fil_img[int(y), int(x)]
        b2, g2, r2 = fil_img[int(y), int(x) + 1]
        b3, g3, r3 = fil_img[int(y) + 1, int(x)]
        b4, g4, r4 = fil_img[int(y) + 1, int(x) + 1]

        b_x_upper, g_x_upper, r_x_upper = (
            b1 * x_proportion_lower + b2 * x_proportion_upper,
            g1 * x_proportion_lower + g2 * x_proportion_upper,
            r1 * x_proportion_lower + r2 * x_proportion_upper,
        )
        b_x_lower, g_x_lower, r_x_lower = (
            b3 * x_proportion_lower + b4 * x_proportion_upper,
            g3 * x_proportion_lower + g4 * x_proportion_upper,
            r3 * x_proportion_lower + r4 * x_proportion_upper,
        )

        b, g, r = (
            b_x_upper * y_proportion_lower + b_x_lower * y_proportion_upper,
            g_x_upper * y_proportion_lower + g_x_lower * y_proportion_upper,
            r_x_upper * y_proportion_lower + r_x_lower * y_proportion_upper,
        )
        return int(r), int(g), int(b)
        # tmp = fil_img[int(y), int(x)]
        # return tmp[0], tmp[1], tmp[2]

    def SampleOneMaterial(fil_img: np.ndarray, OrderNumber: int, StartAngle: float, radius_min: int, radius_max: int, center_x: int, center_y: int):
        Shift = np.pi / 32
        Start_Angle = StartAngle - (OrderNumber * (np.pi / 8)) - Shift
        End_Angle = StartAngle - ((OrderNumber + 1) *(np.pi / 8)) + Shift

        # print(Start_Angle, End_Angle)

        OneThicknessSamples_r = []
        OneThicknessSamples_g = []
        OneThicknessSamples_b = []
        for radius in range(radius_min, radius_max + 1, 1):
            # print(radius)
            for angle in np.arange(End_Angle, Start_Angle, np.pi / (radius / 8)):
                (r, g, b) = sampling.SamplePoint(fil_img, radius, angle, center_x, center_y)
                OneThicknessSamples_r.extend([r])
                OneThicknessSamples_g.extend([g])
                OneThicknessSamples_b.extend([b])

        categorized_r = sampling.categorize(OneThicknessSamples_r)
        categorized_g = sampling.categorize(OneThicknessSamples_g)
        categorized_b = sampling.categorize(OneThicknessSamples_b)

        print(categorized_r, categorized_g, categorized_b)
        return categorized_r, categorized_g, categorized_b
        
    def categorize(data):
        return int(sum(data) / len(data)) if data else 0

    # def gaussian_fit_score(ImageData):
    #     data = np.array(ImageData).reshape(-1, 1)

    #     gm = GaussianMixture(
    #         n_components=1,
    #         covariance_type="full",
    #         random_state=0,
    #         init_params="random",  # <- no internal KMeans
    #     )
    #     gm.fit(data)

    #     # higher (less negative) means closer to normal
    #     score = gm.score(data)
    #     mean = gm.means_.ravel()[0]
    #     return score, mean

    # def categorize(data):
    #     global KMean_counter
    #     arr = np.array(data, dtype=np.float64).reshape(-1, 1)
    #     '''
    #     if arr.size == 0:
    #         return arr.flatten()
    #     if np.unique(arr).size < 2:
    #         return arr.flatten()
    #     '''
    #     normal_score, normal_mean = sampling.gaussian_fit_score(arr)
    #     if normal_score > 0.0:
    #         print(arr.flatten())
    #         return normal_mean

    #     KMean_counter += 1

    #     kmeans = KMeans(
    #         n_clusters=2,
    #         init="random",   # <- avoids the k-means++ potential / matmul
    #         n_init=10,
    #         random_state=0,
    #     )
    #     kmeans.fit(arr)
    #     centers = kmeans.cluster_centers_.flatten()
    #     low_label = np.argmin(centers)     # cluster whose mean is smallest
    #     lows = arr[kmeans.labels_ == low_label]
    #     low = lows[0].item()        # or lows[0].item()
    #     return low
    
    def compute(self, ImagePath, StartAngle):
        fil_img, center_x, center_y, radius_min, radius_max = sampling.prepare(ImagePath)
        SampledArray = []
        for OrderNumber in range(16):
            r, g, b = sampling.SampleOneMaterial(fil_img, OrderNumber, StartAngle, radius_min, radius_max, center_x, center_y)
            # print(StartAngle)
            SampledArray.append((((OrderNumber + 1) / 10), (r, g, b)))
        return SampledArray


def display_sample(SampledArray: list):

    # Expect 16 samples arranged conceptually as a 4x4 grid
    grid_size = 4

    fig, ax = plt.subplots()

    for idx, item in enumerate(SampledArray):
        _, rgb = item
        r, g, b = rgb

        row = idx // grid_size
        col = idx % grid_size

        # Normalize RGB to [0,1] for matplotlib
        color = (r / 255.0, g / 255.0, b / 255.0)

        rect = plt.Rectangle((col, grid_size - 1 - row), 1, 1, color=color)
        ax.add_patch(rect)

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.axis('off')

if __name__ == "__main__":
    ImagePath = "csrc/SamplingNew/Images/af39ab4f7a6b9f95d79b29a72cce839c.jpg"

    SampleInstance = sampling()

    StartAngle = np.pi / 2 - 1.0406
    array = SampleInstance.compute(ImagePath, StartAngle)

    # print(f"KMeans was used {KMean_counter} times.")
    print(array)

    display_sample(array)