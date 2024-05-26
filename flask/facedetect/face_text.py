import cv2
import numpy as np
import collections
import dlib
from flask import Flask, send_file
from PIL import Image

# 创建十二级肤色的字典
class colorList:
    def getColorList(self):
        dict = collections.defaultdict(list)

        # c冷1白
        lower_coldwhite1 = np.array([9, 30, 170])  # bgr值 203 219 243
        upper_coldwhite1 = np.array([11, 90, 255])
        color_list = []
        color_list.append(lower_coldwhite1)
        color_list.append(upper_coldwhite1)
        dict['c冷1白'] = color_list

        # w暖1白
        lower_warmwhite1 = np.array([13, 30, 200])  # bgr值 185 213 248
        upper_warmwhite1 = np.array([15, 90, 255])
        color_list = []
        color_list.append(lower_warmwhite1)
        color_list.append(upper_warmwhite1)
        dict['w暖1白'] = color_list

        # c冷2白
        lower_coldwhite2 = np.array([11, 30, 200])  # bgr值 186 209 241
        upper_coldwhite2 = np.array([15, 90, 250])
        color_list = []
        color_list.append(lower_coldwhite2)
        color_list.append(upper_coldwhite2)
        dict['c冷2白'] = color_list

        # w暖2白
        lower_warmwhite2 = np.array([14, 48, 206])  # bgr值 167 202 236
        upper_warmwhite2 = np.array([16, 105, 255])
        color_list = []
        color_list.append(lower_warmwhite2)
        color_list.append(upper_warmwhite2)
        dict['w暖2白'] = color_list

        # cn冷3白
        lower_coldwhite3 = np.array([10, 52, 200])  # bgr值 157 186 231
        upper_coldwhite3 = np.array([12, 112, 255])
        color_list = []
        color_list.append(lower_coldwhite3)
        color_list.append(upper_coldwhite3)
        dict['cn冷3白'] = color_list

        # wn暖3白
        lower_warmwhite3 = np.array([13, 70, 200])  # bgr值 144 186 233
        upper_warmwhite3 = np.array([15, 130, 255])
        color_list = []
        color_list.append(lower_warmwhite3)
        color_list.append(upper_warmwhite3)
        dict['wn暖3白'] = color_list

        # n中性3白
        lower_middlewhite3 = np.array([14, 37, 150])  # bgr值 149 179 214
        upper_middlewhite3 = np.array([15, 97, 245])
        color_list = []
        color_list.append(lower_middlewhite3)
        color_list.append(upper_middlewhite3)
        dict['n中性3白'] = color_list

        # cn冷3白
        lower_cncoldwhite3 = np.array([8, 55, 148])  # bgr值 137 159 207
        upper_cncoldwhite3 = np.array([10, 115, 216])
        color_list = []
        color_list.append(lower_cncoldwhite3)
        color_list.append(upper_cncoldwhite3)
        dict['cn冷3白'] = color_list

        # w健康麦
        lower_whealthywheat = np.array([14, 94, 170])
        upper_whealthywheat = np.array([15, 156, 238])
        color_list = []
        color_list.append(lower_whealthywheat)
        color_list.append(upper_whealthywheat)
        dict['w健康麦'] = color_list

        # c健康麦
        lower_chealthywheat = np.array([9, 92, 180])  # bgr值 107 142 208
        upper_chealthywheat = np.array([11, 155, 240])
        color_list = []
        color_list.append(lower_chealthywheat)
        color_list.append(upper_chealthywheat)
        dict['c健康麦'] = color_list

        # w暖偏黑
        lower_warmblack = np.array([12, 124, 140])  # bgr值 67 112 173
        upper_warmblack = np.array([14, 176, 200])
        color_list = []
        color_list.append(lower_warmblack)
        color_list.append(upper_warmblack)
        dict['w暖偏黑'] = color_list

        # c冷偏黑
        lower_coldblack = np.array([9, 86, 126])  # bgr值 85 107 157
        upper_coldblack = np.array([10, 146, 186])
        color_list = []
        color_list.append(lower_coldblack)
        color_list.append(upper_coldblack)
        dict['c冷偏黑'] = color_list

        return dict

    def get_color(self, imgPath,face_detect):
        # img = cv2.imread(imgPath)  # 读取图像
        resize_img = cv2.resize(imgPath, dsize=(400, 400))
        gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)  # 转化成灰度
        # 加载人脸检测器

        # 使用人脸检测器检测人脸
        faces = face_detect.detectMultiScale(gray)
        # hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)#将图像转换为HSV
        # resize_img = cv.resize(img, dsize=(400, 400))
        for x, y, w, h in faces:
            # 提取人脸区域
            roi = imgPath[y:y + h, x:x + w]

            # 在人脸区域检测肤色
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 48, 80], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv_roi, lower_skin, upper_skin)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.rectangle(resize_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # cv.imshow('resize_img', roi)
        hsv = cv2.cvtColor(imgPath, cv2.COLOR_BGR2HSV)  # 将图像转换为HSV
        maxsum = 0
        color = None
        color_dict = colorList().getColorList()

        for d in color_dict:  # 对应上面创立的颜色字典,比较面积的大小
            mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
            binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
            binary = cv2.dilate(binary, None, iterations=2)
            img, cnts = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sum = 0
            for c in img:
                sum += cv2.contourArea(c)
            if sum > maxsum:
                maxsum = sum
                color = d
        return color


class contour:
    def contourImg(self, imgPath, predictor_path):
        # img = cv2.imread(imgPath)
        n = 2
        resPoints = []
        img = cv2.resize(imgPath, (0, 0), fx=1 / n, fy=1 / n, interpolation=cv2.INTER_NEAREST)
        # 转换为灰阶图片
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 正向人脸检测器将图像
        detector = dlib.get_frontal_face_detector()
        # 使用训练好的68个特征点模型
        predictor = dlib.shape_predictor(predictor_path)
        # 使用检测器来检测图像中的人脸
        faces = detector(gray, 1)
        # 打印结果
        if len(faces) == 0:
            print("未检测到人脸")
            resContourPoints = resPoints[2:17] + resPoints[18:22] + resPoints[23:27] + resPoints[28:36]
            return resPoints, resContourPoints, img
        else:
            shape = predictor(img, faces[0])
            for i in range(len(shape.parts())):
                (resPoints.append([shape.part(i).x, shape.part(i).y]))
            win = dlib.image_window()

            win.clear_overlay()
            win.set_image(img)
            # # 使用predictor来计算面部轮廓
            # shape = predictor(img, faces[i])
            #
            # # 绘制面部轮廓
            # win.add_overlay(shape)
            for i in range(2, 17):
                cv2.line(img, (shape.part(i - 1).x, shape.part(i - 1).y), (shape.part(i).x, shape.part(i).y), (0, 255, 0),
                         10)
            for i in range(18, 22):
                cv2.line(img, (shape.part(i - 1).x, shape.part(i - 1).y), (shape.part(i).x, shape.part(i).y), (0, 255, 0),
                         10)
            for i in range(23, 27):
                cv2.line(img, (shape.part(i - 1).x, shape.part(i - 1).y), (shape.part(i).x, shape.part(i).y), (0, 255, 0),
                         10)
            for i in range(28, 36):
                cv2.line(img, (shape.part(i - 1).x, shape.part(i - 1).y), (shape.part(i).x, shape.part(i).y), (0, 255, 0),
                         10)
            # return res, img
            image = Image.fromarray(img)
            resContourPoints = resPoints[2:17] + resPoints[18:22] + resPoints[23:27] + resPoints[28:36]
            return resPoints, resContourPoints, image


class sensitiveSkin:
    def hsv_threshold(self, image, h_min, s_min, v_min, h_max, s_max, v_max, in_range_color, out_of_range_color):
        # 将BGR图像转换为HSV图像
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 创建掩膜，将在范围内的像素设置为1，不在范围内的像素设置为0
        mask = cv2.inRange(hsv_image, (h_min, s_min, v_min), (h_max, s_max, v_max))

        # 将掩膜像素点的HSV值设定为指定颜色
        result = hsv_image.copy()
        result[np.where(mask == 255)] = in_range_color
        result[np.where(mask == 0)] = out_of_range_color

        return result

    def sensitiveSkinImg(self, imgPath):
        # 加载RGB图像
        rgb_image = imgPath

        # 设置阈值范围
        h_min, s_min, v_min = 0, 30, 100
        h_max, s_max, v_max = 30, 165, 195

        # 定义颜色
        in_range_color = np.array([0, 255, 255])  # 在范围内的像素点的颜色
        out_of_range_color = np.array([0, 0, 255])  # 不在范围内的像素点的颜色

        # 进行HSV图像的阈值分割并设定颜色
        result = self.hsv_threshold(rgb_image, h_min, s_min, v_min, h_max, s_max, v_max, in_range_color,
                                    out_of_range_color)

        # 将结果转换为BGR格式以便显示
        result_bgr = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

        # 将掩膜图像与原图融合
        blended_image = cv2.addWeighted(rgb_image, 0.3, result_bgr, 0.7, 0)

        # 将图像缩小60%
        scaled_result = cv2.resize(blended_image, None, fx=0.4, fy=0.4)
        imageSensitiveSkin = Image.fromarray(scaled_result)

        # 显示结果图像
        return imageSensitiveSkin
