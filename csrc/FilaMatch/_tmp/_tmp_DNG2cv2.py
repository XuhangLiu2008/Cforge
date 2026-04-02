import cv2
import numpy as np
import rawpy

def bayer_to_rgb(bayer_data, pattern, output_dtype=np.uint8, scale=True):
    """
    将 Bayer 格式的单通道数据转换为 RGB 图像
    
    参数:
        bayer_data: 单通道 numpy 数组，uint16 或 float32
        pattern: Bayer 模式，如 'RGGB', 'BGGR', 'GRBG', 'GBRG'
        output_dtype: 输出图像的数据类型（默认 np.uint8）
        scale: 是否将数据缩放到输出类型的有效范围（默认 True）
    
    返回:
        rgb_image: BGR 格式的三通道图像（OpenCV 默认 BGR）
    """
    # 1. 确保数据为 float32，便于缩放
    if bayer_data.dtype == np.uint16:
        bayer_float = bayer_data.astype(np.float32)
    else:
        bayer_float = bayer_data.astype(np.float32)
    
    # 2. 可选：缩放至 0-255（如果原始数据范围是 0-65535）
    if scale:
        max_val = np.max(bayer_float)
        # 假设原始数据最大可能为 65535（16-bit），也可以使用实际最大值
        if max_val > 255:
            bayer_float = bayer_float / 65535.0 * 255.0
        # 如果需要输出 float32 且范围 0-1，可以再调整
        if output_dtype == np.uint8:
            bayer_float = np.clip(bayer_float, 0, 255).astype(np.uint8)
        else:
            bayer_float = bayer_float / 255.0  # 转为 0-1 float
    
    # 3. 将 Bayer 模式映射为 OpenCV 的常量
    pattern_map = {
        'RGGB': cv2.COLOR_BayerRG2BGR,
        'BGGR': cv2.COLOR_BayerBG2BGR,
        'GRBG': cv2.COLOR_BayerGR2BGR,
        'GBRG': cv2.COLOR_BayerGB2BGR,
    }
    if pattern not in pattern_map:
        raise ValueError(f"不支持的 Bayer 模式: {pattern}")
    
    # 4. 去马赛克
    bgr = cv2.cvtColor(bayer_float, pattern_map[pattern])
    
    return bgr

# 使用示例
if __name__ == "__main__":
    dng_file = "your_image.dng"
    
    # 1. 使用之前定义的黑电平修正函数
    from your_previous_code import subtract_black_level  # 假设你已实现
    
    corrected_data, _ = subtract_black_level(dng_file)
    # corrected_data 是 uint16 的单通道 Bayer 数据
    
    # 2. 读取原始 RAW 对象以获取 Bayer 模式
    raw = rawpy.imread(dng_file)
    pattern = raw.raw_pattern  # 例如 [[0,1],[1,2]]，需要转为字符串
    # 将 pattern 数组转为字符串，如 'RGGB'
    pattern_str = ''.join([['R','G','G','B'][idx] for idx in pattern.flatten()])
    
    # 3. 转换为 BGR 图像
    bgr_image = bayer_to_rgb(corrected_data, pattern_str)
    
    # 4. 使用 OpenCV 显示或保存
    cv2.imshow("Result", bgr_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存为文件
    cv2.imwrite("output.jpg", bgr_image)