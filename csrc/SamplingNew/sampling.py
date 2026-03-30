import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

import scipy as sp

import visual_manage
from copy import deepcopy

KMean_counter = 0

class sampling:

    @staticmethod 
    def _flex_read(image_path : str | np.ndarray):
        # read image
        if type(image_path) == str:
            img = cv2.imread(image_path)
            if img is None:
                print("Error: Could not load image.")
            else:
                print("Image loaded successfully.")
                print(f"Image shape: {img.shape}")
            
            if img.shape[2] != 3:
                print("Error: Not a RGB image.")
        else:
            img = image_path
        return img

    class geo_info:
        def __init__(self, center_x, center_y, radius, start_angle):
            self.center_x = center_x
            self.center_y = center_y
            self.radius = radius
            self.start_angle = start_angle

    @staticmethod
    def inverse_gamma_correction(img, gamma=2.2):
        # 将标量或数组转换为浮点
        value = np.asarray(value, dtype=np.float32)
        # 根据 sRGB 规范，对于小于 0.04045 的数值使用线性部分
        mask = value <= 0.04045
        result = np.zeros_like(value)
        result[mask] = value[mask] / 12.92
        result[~mask] = ((value[~mask] + 0.055) / 1.055) ** 2.4
        return result

    GeoSampleNumAngles = 720
    GeoSampleNumRadii = 500
    CentreSampleDiskSize = 10

    CentreEmptyRealR = 10.0 #(mm)
    SampleDiskRealSize = 30.0 #(mm)

    @staticmethod
    def prepare_geometric(image_path : str | np.ndarray, shown=True) -> geo_info:

        def sample_ring(gray, center, radius, angles):
            intensities = []

            for theta in angles:
                px = int(center[0] + radius * np.cos(theta))
                py = int(center[1] + radius * np.sin(theta))

                if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
                    intensities.append(gray[py, px])
                else:
                    intensities.append(0)

            return np.array(intensities)

        def detect_circle(gray):
            
            blur = cv2.GaussianBlur(gray, (9, 9), 1.5)

            circles = cv2.HoughCircles(
                blur,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=100,
                param1=100,
                param2=30,
                minRadius=50,
                maxRadius=0
            )

            if circles is None:
                raise RuntimeError("No circle detected. Adjust parameters.")

            x, y, _ = np.uint16(np.around(circles))[0][0]

            angles = np.linspace(0, 2*np.pi, sampling.GeoSampleNumAngles, endpoint=False)

            h, w = gray.shape
            radii = np.linspace(0, min(h, w) // 2, sampling.GeoSampleNumRadii, endpoint=False)

            means = []

            for radius in radii:
                intensities = sample_ring(gray, (x, y), radius, angles)
                means.append(np.mean(intensities))

            means = np.array(means)
            gradients = np.gradient(sp.ndimage.gaussian_filter(means, sigma=2))

            small_R = 10

            if np.average(means[:small_R]) < 0.5 * 255: 
                # it's a dark circle -> r image
                # maximum gradient corresponds to the boundary of the centre empty circle
                r_empty_region = radii[np.argmax(gradients)]
                r_sample_disk = int(sampling.SampleDiskRealSize / sampling.CentreEmptyRealR * r_empty_region)

            else:
                # it's a light circle -> t image
                # minimum gradient corresponds to the boundary of the centre empty circle
                r_empty_region = radii[np.argmin(gradients)]
                r_sample_disk = int(sampling.SampleDiskRealSize / sampling.CentreEmptyRealR * r_empty_region)

            return (x, y), r_sample_disk

        def compute_gradient(intensities):
            smooth = cv2.GaussianBlur(
                intensities.reshape(-1, 1), (9, 1), 0
            ).flatten()

            gradient = np.abs(np.gradient(smooth))
            return gradient, smooth

        def multi_ring_gradient(gray, center, r, num_angles=720,
                                r_min_ratio=0.55, r_max_ratio=0.75, step=2,
                                method="median"):

            angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)

            r_min = int(r * r_min_ratio)
            r_max = int(r * r_max_ratio)
            radii = np.arange(r_min, r_max, step)

            all_gradients = []

            for radius in radii:
                intensities = sample_ring(gray, center, radius, angles)
                grad, _ = compute_gradient(intensities)
                all_gradients.append(grad)

            all_gradients = np.array(all_gradients)

            if method == "mean":
                agg_gradient = np.mean(all_gradients, axis=0)
            else:
                agg_gradient = np.median(all_gradients, axis=0)

            return angles, agg_gradient, radii

        def detect_boundary(angles, gradient):
            idx = np.argmax(gradient)
            angle = angles[idx]
            return idx, angle

        def angle_to_coord(center, radius, angle):
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            return (x, y)

        @visual_manage.visualmethod("Boundary Detection Visualization")
        def visualize_result(img, center, r, boundary_angle,
                            fig = None, ax = None, r_display=None):

            output = deepcopy(img)

            if r_display is None:
                r_display = int(r * 0.65)

            # draw outer circle
            cv2.circle(output, center, r, (0,255,0), 2)

            # draw center
            cv2.circle(output, center, 3, (255,0,0), -1)

            # draw boundary line
            end_x = int(center[0] + r * np.cos(boundary_angle))
            end_y = int(center[1] + r * np.sin(boundary_angle))
            cv2.line(output, center, (end_x, end_y), (0,0,255), 2)

            ax.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            ax.set_title("Boundary Detection")
            ax.axis('off')

        @visual_manage.visualmethod("Gradient Profile")
        def plot_gradient(angles, gradient, boundary_angle, fig=None, ax=None):

            if fig is None or ax is None:
                fig, ax = plt.subplots(figsize=(10, 4))

            ax.plot(np.degrees(angles), gradient)
            ax.axvline(np.degrees(boundary_angle), color='r', alpha=0.5)
            ax.text(np.degrees(boundary_angle)+5, np.max(gradient)*0.8, f"{np.degrees(boundary_angle):.2f}°", color='r')
            ax.set_xlabel("Angle (degrees)")
            ax.set_ylabel("Gradient")
            ax.set_title("Aggregated Gradient Profile")
            ax.grid(True)

        img = sampling._flex_read(image_path)
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect center
        center, r = detect_circle(gray)

        print("Center:", center, "Radius:", r)

        # compute multi-ring gradient
        angles, gradient, radii = multi_ring_gradient(
            gray, center, r,
            method="median"   # change to "mean" if needed
        )

        # detect boundary
        idx, boundary_angle = detect_boundary(angles, gradient)

        print("Boundary angle (deg):", np.degrees(boundary_angle))

        # convert to coordinate
        display_radius = int(np.mean(radii))
        coord = angle_to_coord(center, display_radius, boundary_angle)

        print("Boundary coordinate:", coord)

        if shown:
            # visualize
            visualize_result(img, center, r, boundary_angle)

            # plot
            plot_gradient(angles, gradient, boundary_angle)

        return sampling.geo_info(center[0], center[1], r, 2 * np.pi - boundary_angle)

    BoundaryDismissAngle = np.pi / 32
    MinRadiusRatio = 0.5
    MaxRadiusRatio = 0.75

    LayerThickness = 0.1
    
    @staticmethod
    def sampleOneImage(ImagePath : str | np.ndarray, geo_info : geo_info = None, categorize=True, shown=False):

        def prepare_array(fil_img : np.ndarray):

            center_x, center_y, radius, start_angle = sampling.prepare_geometric(fil_img, shown=False)

            return center_x, center_y, radius * sampling.MinRadiusRatio, radius * sampling.MaxRadiusRatio, start_angle

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

            return fil_img, *prepare_array(fil_img)
        
        def displaySamplePoint(img_copy, r, angle, center_x, center_y):
            x = center_x + r * np.cos(angle)
            y = center_y - r * np.sin(angle) # indice ordered from top to bottom

            cv2.circle(img_copy, (int(x), int(y)), 1, (0, 255, 0), -1)

            return img_copy

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
            # tmp = fil_img[int(y), int(x)]
            # return tmp[0], tmp[1], tmp[2]

        def SampleOneMaterial(fil_img: np.ndarray, OrderNumber: int, StartAngle: float, radius_min: int, radius_max: int, center_x: int, center_y: int, img_copy = None):

            Start_Angle = StartAngle - (OrderNumber * (np.pi / 8)) - sampling.BoundaryDismissAngle 
            End_Angle = StartAngle - ((OrderNumber + 1) *(np.pi / 8)) + sampling.BoundaryDismissAngle 

            # print(Start_Angle, End_Angle)

            OneThicknessSamples_r = []
            OneThicknessSamples_g = []
            OneThicknessSamples_b = []
            for radius in range(radius_min, radius_max + 1, 1):
                # print(radius)
                for angle in np.arange(End_Angle, Start_Angle, np.pi / (radius / 2)):
                    (r, g, b) = SamplePoint(fil_img, radius, angle, center_x, center_y)
                    if img_copy is not None:
                        img_copy = displaySamplePoint(img_copy, radius, angle, center_x, center_y)
                    OneThicknessSamples_r.extend([r])
                    OneThicknessSamples_g.extend([g])
                    OneThicknessSamples_b.extend([b])

            return OneThicknessSamples_r, OneThicknessSamples_g, OneThicknessSamples_b
            
        # def categorize(data):
        #     return int(sum(data) / len(data)) if data else 0

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

        @visual_manage.visualmethod("Sample Points Visualization")
        def visualize_sample(img_copy, fig=None, ax=None):
            ax.imshow(cv2.cvtColor(img_copy.astype(np.uint8), cv2.COLOR_BGR2RGB))
            ax.set_title("Sample Points Visualization")
            ax.axis('off')

        if geo_info is None:
            fil_img, center_x, center_y, radius_min, radius_max, start_angle = prepare(ImagePath)
        else:
            fil_img = ImagePath if type(ImagePath) != str else cv2.imread(ImagePath)
            center_x = geo_info.center_x
            center_y = geo_info.center_y
            radius_min = int(geo_info.radius * sampling.MinRadiusRatio)
            radius_max = int(geo_info.radius * sampling.MaxRadiusRatio)
            start_angle = geo_info.start_angle

        SampledArray = []

        img_copy = deepcopy(fil_img)

        for OrderNumber in range(16):

            OneThicknessSamples_rgb = SampleOneMaterial(fil_img, OrderNumber, start_angle, radius_min, radius_max, center_x, center_y, img_copy if shown else None)

            OneThicknessSamples_r, OneThicknessSamples_g, OneThicknessSamples_b = OneThicknessSamples_rgb
            
            if not categorize:
                avg = lambda x: int(sum(x) / len(x)) if x else 0
                categorized_r = avg(OneThicknessSamples_r)
                categorized_g = avg(OneThicknessSamples_g)
                categorized_b = avg(OneThicknessSamples_b)
            else:
                categorized_r = categorize(OneThicknessSamples_r)
                categorized_g = categorize(OneThicknessSamples_g)
                categorized_b = categorize(OneThicknessSamples_b)

            print(categorized_r, categorized_g, categorized_b)

            SampledArray.append((round((OrderNumber + 1) * sampling.LayerThickness, 2), (categorized_r, categorized_g, categorized_b)))
        
        if shown:
            visualize_sample(img_copy)

        return SampledArray

    d_real = 52.7
    r_real = 30.0

    @staticmethod
    def transmittance(t_img : str | np.ndarray, geo_info = None, shown = False):

        if geo_info is None:
            geo_info = sampling.prepare_geometric(t_img, shown=shown)

        def prepare_t_relative_luminance(t_img : np.ndarray, geo_info: sampling.geo_info = None):
            
            if geo_info is None:
                _, _, r_image, _ = sampling.prepare_geometric(t_img, shown=False)
            else:
                r_image = geo_info.radius

            h, w = t_img.shape[:2]
            center = (w / 2, h / 2)

            # Vectorized distance calculation using meshgrid (100-1000x faster)
            y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

            dis_to_center = np.hypot(x - center[0], y - center[1]).astype(np.float32)

            real_dis_to_center = dis_to_center * (sampling.r_real / r_image)

            V = 1 / (1 + (real_dis_to_center / sampling.d_real) ** 2 ) ** 2
            return t_img / V[..., np.newaxis]

        def centre_white_sample(t_img : np.ndarray, geo_info: sampling.geo_info, region_size = 5):

            center_x, center_y = int(geo_info.center_x), int(geo_info.center_y)
            return np.average(t_img[center_y - region_size : center_y + region_size, 
                                    center_x - region_size : center_x + region_size])

        if type(t_img) == str:
            t_img = cv2.imread(t_img)
            if t_img is None:
                print("Error: Could not load image.")
            else:
                print("Image loaded successfully.")
                print(f"Image shape: {t_img.shape}")
            
            if t_img.shape[2] != 3:
                print("Error: Not a RGB image.")

        # t_relative_luminance = prepare_t_relative_luminance(t_img, geo_info)

        incident_white = centre_white_sample(t_img, geo_info)

        t_relative_luminance = t_img / incident_white * 255.0

        samples = sampling.sampleOneImage(t_relative_luminance, geo_info, shown = shown, categorize = False)

        # cv2.imshow("Relative Luminance", t_relative_luminance)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return [(samples[i][0], np.array(samples[i][1]))for i in range(len(samples))]


    def __init__(self, 
                 t_ImagePath : str | np.ndarray, 
                 r_ImagePath : str | np.ndarray, 
                 src_ImagePath : str | np.ndarray, 
                 r_luminance : float, src_luminance : float):
        
        flex_read = lambda path: cv2.imread(path) if type(path) == str else path
        
        self.t_img = flex_read(t_ImagePath) / 255.0
        self.r_img = flex_read(r_ImagePath) / 255.0
        self.src_img = flex_read(src_ImagePath) / 255.0

        self.r_luminance = r_luminance
        self.src_luminance = src_luminance

        self.t_geo_info = sampling.prepare_geometric(self.t_img, shown=False)

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
    ImagePath = "csrc/SamplingNew/Images/DNGimages/t.png"

    array = sampling.transmittance(ImagePath, shown=True)

    # print(f"KMeans was used {KMean_counter} times.")
    print(array)

    display_sample(array)

    plt.show()