import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

KMean_counter = 0

def sampling(image_path):

    fil_img = cv2.imread(image_path)
    if fil_img is None:
        print("Error: Could not load image.")
    else:
        print("Image loaded successfully.")
        print(f"Image shape: {fil_img.shape}")
    
    if fil_img.shape[2] != 3:
        print("Error: Not a RGB image.")

    img_x = fil_img.shape[0]
    img_y = fil_img.shape[1]

    # step_x = round(img_x / 8)
    # step_y = round(img_y / 8)
    # shift_x = round(step_x / 4)
    # shift_y = round(step_y / 4)
    # print(f"step_x: {step_x}, step_y: {step_y}, shift_x: {shift_x}, shift_y: {shift_y}")

    center_x = img_x // 2
    center_y = img_y // 2
    radius = (img_x + img_y) // 4
    radius_min = radius // (1/4)
    radius_max = radius // (3/4)

    array = []

    # KMean_counter = 0

    def SampleRad(fil_img, r, angle):
        x = center_x + r * np.cos(angle)
        y = center_y + r * np.sin(angle)

        # When the points are integers, or not, use powered averages
        x_proportion_upper = x - int(x)
        y_proportion_upper = y - int(y)
        x_proportion_lower = 1 - x_proportion_upper
        y_proportion_lower = 1 - y_proportion_upper

        b1, g1, r1 = fil_img[int(x), int(y)]
        b2, g2, r2 = fil_img[int(x) + 1, int(y)]
        b3, g3, r3 = fil_img[int(x), int(y) + 1]
        b4, g4, r4 = fil_img[int(x) + 1, int(y) + 1]

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
            b_x_lower * y_proportion_lower + b_x_upper * y_proportion_upper,
            g_x_lower * y_proportion_lower + g_x_upper * y_proportion_upper,
            r_x_lower * y_proportion_lower + r_x_upper * y_proportion_upper,
        )
        return int(r), int(g), int(b)

    def SampleOne():
        pass

    def gaussian_fit_score(data):
        data = np.array(data).reshape(-1, 1)

        gm = GaussianMixture(
            n_components=1,
            covariance_type="full",
            random_state=0,
            init_params="random",  # <- no internal KMeans
        )
        gm.fit(data)

        # higher (less negative) means closer to normal
        score = gm.score(data)
        mean = gm.means_.ravel()[0]
        return score, mean

    def categorize(data):
        global KMean_counter
        arr = np.array(data, dtype=np.float64).reshape(-1, 1)
        '''
        if arr.size == 0:
            return arr.flatten()
        if np.unique(arr).size < 2:
            return arr.flatten()
        '''
        normal_score, normal_mean = gaussian_fit_score(arr)
        if normal_score > 0.0:
            print(arr.flatten())
            return normal_mean

        KMean_counter += 1

        kmeans = KMeans(
            n_clusters=2,
            init="random",   # <- avoids the k-means++ potential / matmul
            n_init=10,
            random_state=0,
        )
        kmeans.fit(arr)
        centers = kmeans.cluster_centers_.flatten()
        low_label = np.argmin(centers)     # cluster whose mean is smallest
        lows = arr[kmeans.labels_ == low_label]
        low = lows[0].item()        # or lows[0].item()
        return low

    return array


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
    image_path = "csrc/SamplingNew/Images/Mar112_orange_lin_up.jpeg"

    array = sampling(image_path)
    StartAngle = - np.pi

    # print(f"KMeans was used {KMean_counter} times.")
    print(array)

    display_sample(array)