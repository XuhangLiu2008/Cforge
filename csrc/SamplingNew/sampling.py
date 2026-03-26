import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import scipy

KMean_counter = 0

class sampling:

    @staticmethod
    def prepare_array(fil_img : np.ndarray):

        img_x = fil_img.shape[1]
        img_y = fil_img.shape[0]

        center_x = img_x // 2
        center_y = img_y // 2
        radius = (img_x + img_y) // 4
        radius_min = int(radius * (1/2))
        radius_max = int(radius * (3/4))

        return center_x, center_y, radius_min, radius_max

    @staticmethod
    def prepare(img_path : str | np.ndarray):

        if type(img_path) == str:
            fil_img = cv2.imread(img_path)
            if fil_img is None:
                print("Error: Could not load image.")
            else:
                print("Image loaded successfully.")
                print(f"Image shape: {fil_img.shape}")
            
            if fil_img.shape[2] != 3:
                print("Error: Not a RGB image.")
        else:
            fil_img = img_path

        return fil_img, *sampling.prepare_array(fil_img)
    
    @staticmethod
    def SamplePoint(fil_img, r, angle, center_x, center_y):
        x = center_x + r * np.cos(angle)
        y = center_y - r * np.sin(angle) # indice ordered from top to bottom

        # When the points are integers, or not, use powered averages
        x_proportion_upper = x - int(x)
        y_proportion_upper = y - int(y)
        x_proportion_lower = 1 - x_proportion_upper
        y_proportion_lower = 1 - y_proportion_upper

        b1, g1, r1 = fil_img[int(y), int(x)]
        b2, g2, r2 = fil_img[int(y), int(x) + 1]
        b3, g3, r3 = fil_img[int(y) + 1, int(x)]
        b4, g4, r4 = fil_img[int(y) + 1, int(x) + 1]

        # fil_img[int(y), int(x)] = np.array([0, 255, 0])
        # fil_img[int(y), int(x) + 1] = np.array([0, 255, 0])
        # fil_img[int(y) + 1, int(x)] = np.array([0, 255, 0])
        # fil_img[int(y) + 1, int(x) + 1] = np.array([0, 255, 0])
        # # for debug

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

    @staticmethod
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

        return OneThicknessSamples_r, OneThicknessSamples_g, OneThicknessSamples_b
        
    # def categorize(data):
    #     return int(sum(data) / len(data)) if data else 0

    @staticmethod
    def gaussian_fit_score(ImageData):
        data = np.array(ImageData).reshape(-1, 1)

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

    @staticmethod
    def categorize(data):
        global KMean_counter
        arr = np.array(data, dtype=np.float64).reshape(-1, 1)
        '''
        if arr.size == 0:
            return arr.flatten()
        if np.unique(arr).size < 2:
            return arr.flatten()
        '''
        normal_score, normal_mean = sampling.gaussian_fit_score(arr)
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
    
    @staticmethod
    def compute(ImagePath : str | np.ndarray, StartAngle, categorize=True):

        fil_img, center_x, center_y, radius_min, radius_max = sampling.prepare(ImagePath)
        SampledArray = []

        for OrderNumber in range(16):

            OneThicknessSamples_rgb = sampling.SampleOneMaterial(fil_img, OrderNumber, StartAngle, radius_min, radius_max, center_x, center_y)

            OneThicknessSamples_r, OneThicknessSamples_g, OneThicknessSamples_b = OneThicknessSamples_rgb

            # cv2.imshow("image", fil_img)
            # cv2.waitKey(0)
            # print(StartAngle)
            
            if not categorize:
                avg = lambda x: int(sum(x) / len(x)) if x else 0
                categorized_r = avg(OneThicknessSamples_r)
                categorized_g = avg(OneThicknessSamples_g)
                categorized_b = avg(OneThicknessSamples_b)
            else:
                categorized_r = sampling.categorize(OneThicknessSamples_r)
                categorized_g = sampling.categorize(OneThicknessSamples_g)
                categorized_b = sampling.categorize(OneThicknessSamples_b)

            print(categorized_r, categorized_g, categorized_b)

            SampledArray.append((((OrderNumber + 1) / 10), (categorized_r, categorized_g, categorized_b)))
        return SampledArray

    def __init__(self, t_ImagePath : str, src_ImagePath : str, r_ImagePath : str, 
                 luminance : np.ndarray = None, 
                 t_StartAngle : float = None, r_StartAngle : float = None):
        
        self.t_ImagePath = t_ImagePath
        self.src_ImagePath = src_ImagePath
        self.r_ImagePath = r_ImagePath

        self.luminance = luminance if luminance is not None else np.ones(3)

        self.t_StartAngle = t_StartAngle if t_StartAngle is not None else 0
        self.r_StartAngle = r_StartAngle if r_StartAngle is not None else 0

        self.r_input_luminance = None # should be float
        self.t_input_luminance = None

        self.r_output_luminance = None # should be np.ndarray
        self.t_output_luminance = None

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

    StartAngle = np.pi / 2 - 1.0406
    array = sampling.compute(ImagePath, StartAngle)

    # print(f"KMeans was used {KMean_counter} times.")
    print(array)

    display_sample(array)