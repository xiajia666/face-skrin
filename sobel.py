import cv2
import numpy as np

# 加载图像
image = cv2.imread('ok1.jpg')

# 将图像转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用Canny边缘检测算法检测边缘
edges = cv2.Canny(gray_image, 50, 10)

# 找到轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在图像上绘制轮廓
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# 显示结果图像
cv2.imshow('Detected Wrinkles', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
