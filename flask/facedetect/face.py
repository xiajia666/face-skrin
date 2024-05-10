import cv2
import dlib
import time
# 读取图片
img_path = "./input.jpg"
img = cv2.imread(img_path)
n = 2
img = cv2.resize(img, (0, 0), fx=1/n, fy=1/n, interpolation=cv2.INTER_NEAREST)
# 转换为灰阶图片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 正向人脸检测器将图像
detector = dlib.get_frontal_face_detector()
# 使用训练好的68个特征点模型
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
# 使用检测器来检测图像中的人脸
faces = detector(gray, 1)
# 打印结果
print("人脸数: ", len(faces))
for i, face in enumerate(faces):
	print("第", i+1, "个人脸的矩形框坐标：\n","left:", face.left(), "right:", face.right(), "top:", face.top(), "bottom:", face.bottom())
	# 获取人脸特征点
	shape = predictor(img, face)
	print("第", i+1, '个人脸特征点:')
	print(shape.parts())


win = dlib.image_window()

# win.clear_overlay()
# win.set_image(img)
# 使用predictor来计算面部轮廓
print(i)
shape = predictor(img, faces[i])

# 绘制面部轮廓
win.add_overlay(shape)
for i in range(2, 17):
    cv2.line(img, (shape.part(i-1).x, shape.part(i-1).y), (shape.part(i).x, shape.part(i).y), (0, 255, 0), 1)
for i in range(18, 22):
    cv2.line(img, (shape.part(i-1).x, shape.part(i-1).y), (shape.part(i).x, shape.part(i).y), (0, 255, 0), 1)
for i in range(23, 27):
    cv2.line(img, (shape.part(i-1).x, shape.part(i-1).y), (shape.part(i).x, shape.part(i).y), (0, 255, 0), 1)
for i in range(28, 36):
    cv2.line(img, (shape.part(i-1).x, shape.part(i-1).y), (shape.part(i).x, shape.part(i).y), (0, 255, 0), 1)



output_img_path = "output.jpg"
cv2.imwrite(output_img_path, img)



