import cv2
import numpy as np
import matplotlib.pyplot as plt
import visual_manage

def find_boundary(image_path, shown=True):

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

        x, y, r = np.uint16(np.around(circles))[0][0]
        return (x, y), r

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

        output = img.copy()

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

    img = cv2.imread(image_path)
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

    return {
        "center": center,
        "radius": r,
        "boundary_angle_rad": boundary_angle,
        "boundary_angle_deg": np.degrees(boundary_angle),
        "boundary_coord": coord
    }


# =========================
# Run
# =========================
if __name__ == "__main__":
    result = find_boundary("csrc/SamplingNew/Images/af39ab4f7a6b9f95d79b29a72cce839c.jpg")
    print(result)

    plt.show()