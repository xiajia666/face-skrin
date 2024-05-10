import cv2
import numpy as np

def hsv_threshold(image, h_min, s_min, v_min, h_max, s_max, v_max, in_range_color, out_of_range_color):
    # 将BGR图像转换为HSV图像
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 创建掩膜，将在范围内的像素设置为1，不在范围内的像素设置为0
    mask = cv2.inRange(hsv_image, (h_min, s_min, v_min), (h_max, s_max, v_max))

    # 将掩膜像素点的HSV值设定为指定颜色
    result = hsv_image.copy()
    result[np.where(mask == 255)] = in_range_color
    result[np.where(mask == 0)] = out_of_range_color

    return result

def main():
    # 加载RGB图像
    rgb_image = cv2.imread("./input.jpg")

    # 设置阈值范围
    h_min, s_min, v_min = 0, 30, 100
    h_max, s_max, v_max = 30, 165, 195

    # 定义颜色
    in_range_color = np.array([0, 255, 255])  # 在范围内的像素点的颜色
    out_of_range_color = np.array([0, 0, 255])  # 不在范围内的像素点的颜色

    # 进行HSV图像的阈值分割并设定颜色
    result = hsv_threshold(rgb_image, h_min, s_min, v_min, h_max, s_max, v_max, in_range_color, out_of_range_color)

    # 将结果转换为BGR格式以便显示
    result_bgr = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

    # 将掩膜图像与原图融合
    blended_image = cv2.addWeighted(rgb_image, 0.3, result_bgr, 0.7, 0)

    # 将图像缩小60%
    scaled_result = cv2.resize(blended_image, None, fx=0.4, fy=0.4)

    # 显示结果图像
    cv2.imshow("Result Image", scaled_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()