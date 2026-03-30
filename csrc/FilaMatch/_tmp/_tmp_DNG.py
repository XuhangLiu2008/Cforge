import rawpy
import numpy as np
import cv2

def dng_to_linear_rgb(dng_path, dark_dng_path):
    # === 读取 RAW ===
    with rawpy.imread(dng_path) as raw:
        raw_img = raw.raw_image_visible.astype(np.float32)

    # === 读取暗帧 ===
    with rawpy.imread(dark_dng_path) as dark_raw:
        dark_img = dark_raw.raw_image_visible.astype(np.float32)

    # === 暗帧校正（逐像素）===
    corrected = raw_img - dark_img
    corrected = np.clip(corrected, 0, None)

    # === 写回 raw buffer（关键技巧）===
    with rawpy.imread(dng_path) as raw:
        raw.raw_image_visible[:] = corrected

        rgb = raw.postprocess(
            use_camera_wb=False,
            use_auto_wb=False,
            no_auto_bright=True,
            gamma=(1, 1),           # 线性
            output_bps=16
        )

    # === 转 float [0,1] ===
    rgb = rgb.astype(np.float32)

    return rgb


if __name__ == "__main__":
    rgb = dng_to_linear_rgb("csrc/SamplingNew/Images/DNGimages/r.dng", "csrc/SamplingNew/Images/DNGimages/darkframe.dng")

    bgr = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imshow("Linear RGB (no WB)", bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()