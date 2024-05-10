import cv2 as cv
import numpy as np
import collections
# import dlib
#创建十二级肤色的字典
class colorList:
    def getColorList(self):
        dict = collections.defaultdict(list)

        # c冷1白
        lower_coldwhite1 = np.array([9, 33, 170])#bgr值 203 219 243
        upper_coldwhite1 = np.array([11, 53, 255])
        color_list = []
        color_list.append(lower_coldwhite1)
        color_list.append(upper_coldwhite1)
        dict['c冷1白'] = color_list

        # w暖1白
        lower_warmwhite1 = np.array([13, 56, 238])#bgr值 185 213 248
        upper_warmwhite1 = np.array([15, 76, 258])
        color_list = []
        color_list.append(lower_warmwhite1)
        color_list.append(upper_warmwhite1)
        dict['w暖1白'] = color_list

        # c冷2白
        lower_coldwhite2 = np.array([10,47, 200])#bgr值 186 209 241
        upper_coldwhite2 = np.array([13, 70, 250])
        color_list = []
        color_list.append(lower_coldwhite2)
        color_list.append(upper_coldwhite2)
        dict['c冷2白'] = color_list

        # w暖2白
        lower_warmwhite2 = np.array([14, 63, 226])#bgr值 167 202 236
        upper_warmwhite2 = np.array([16, 85, 245])
        color_list = []
        color_list.append(lower_warmwhite2)
        color_list.append(upper_warmwhite2)
        dict['w暖2白'] = color_list

        # cn冷3白
        lower_coldwhite3 = np.array([10, 72, 221])#bgr值 157 186 231
        upper_coldwhite3 = np.array([12, 92, 241])
        color_list = []
        color_list.append(lower_coldwhite3)
        color_list.append(upper_coldwhite3)
        dict['cn冷3白'] = color_list

        # wn暖3白
        lower_warmwhite3 = np.array([13, 84, 221])#bgr值 144 186 233
        upper_warmwhite3 = np.array([15, 109, 242])
        color_list = []
        color_list.append(lower_warmwhite3)
        color_list.append(upper_warmwhite3)
        dict['wn暖3白'] = color_list

        # n中性3白
        lower_middlewhite3 = np.array([14, 71,150])#bgr值 149 179 214
        upper_middlewhite3 = np.array([16, 91, 224])
        color_list = []
        color_list.append(lower_middlewhite3)
        color_list.append(upper_middlewhite3)
        dict['n中性3白'] = color_list

        # cn冷3白
        lower_cncoldwhite3 = np.array([8, 75, 148])#bgr值 137 159 207
        upper_cncoldwhite3 = np.array([10,95, 216])
        color_list = []
        color_list.append(lower_cncoldwhite3)
        color_list.append(upper_cncoldwhite3)
        dict['cn冷3白'] = color_list

        # w健康麦
        lower_whealthywheat = np.array([14,120,200])
        upper_whealthywheat = np.array([15, 130, 218])
        color_list = []
        color_list.append(lower_whealthywheat)
        color_list.append(upper_whealthywheat)
        dict['w健康麦'] = color_list

        # c健康麦
        lower_chealthywheat = np.array([9, 112, 200])#bgr值 107 142 208
        upper_chealthywheat = np.array([11, 125, 210])
        color_list = []
        color_list.append(lower_chealthywheat)
        color_list.append(upper_chealthywheat)
        dict['c健康麦'] = color_list

        # w暖偏黑
        lower_warmblack = np.array([12, 140,166])#bgr值 67 112 173
        upper_warmblack = np.array([14, 156, 182])
        color_list = []
        color_list.append(lower_warmblack)
        color_list.append(upper_warmblack)
        dict['w暖偏黑'] = color_list

        # c冷偏黑
        lower_coldblack = np.array([9, 116, 151])#bgr值 85 107 157
        upper_coldblack = np.array([10, 120, 170])
        color_list = []
        color_list.append(lower_coldblack)
        color_list.append(upper_coldblack)
        dict['c冷偏黑'] = color_list

        return dict

    def get_color(self):
        img = cv.imread("qq.jpg")#读取图像
        hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)#将图像转换为HSV
        resize_img = cv.resize(img, dsize=(400, 400))
        # cv.imshow('resize_img', resize_img)
        maxsum = 0
        color = None
        color_dict = colorList().getColorList()
        # while True:
        #     if ord('q') == cv.waitKey(0):
        #         break


        for d in color_dict:  # 对应上面创立的颜色字典,比较面积的大小
            mask = cv.inRange(hsv, color_dict[d][0], color_dict[d][1])
            binary = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)[1]
            binary = cv.dilate(binary, None, iterations=2)
            img, cnts = cv.findContours(binary.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            sum = 0
            for c in img:
                sum += cv.contourArea(c)
            if sum > maxsum:
                maxsum = sum
                color = d
        return color


if __name__ == '__main__':
    print(colorList().get_color())
