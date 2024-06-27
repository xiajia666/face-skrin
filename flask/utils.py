import numpy as np
from PIL import Image
from numpy import shape
from PIL import Image, ImageDraw, ImageEnhance

import json

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2.utils.visualizer import ColorMode
import torch
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy


def tensor_to_json(data):
    # 将张量移动到 CPU，并转换为 NumPy 数组
    tensor = data.cpu().detach().numpy()
    tensor_numpy = numpy.asarray(tensor)

    # 转换为 Python 列表
    tensor_list = tensor_numpy.tolist()

    # 转换为 JSON 字符串
    tensor_json = json.dumps(tensor_list)
    return tensor_json


# 切换为图形界面显示的终端TkAgg
matplotlib.use('TkAgg')


def plot_samples(dataset_name, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for s in random.sample(dataset_custom, n):
        img = cv2.imread(s["file_name"])
        v = Visualizer(img[:, :, ::-1], metadata=dataset_custom_metadata, scale=0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(15, 20))
        plt.imshow(v.get_image())
        plt.show()


def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device,
                  output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)

    cfg.DATALOADER.NUM_WORKERS = 0

    cfg.SOLVER.IMS_PER_BATCH = 1
    # cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.BASE_LR = 0.001

    cfg.SOLVER.MAX_ITER = 4000
    cfg.SOLVER.STEPS = []

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir
    return cfg


def on_Image(image_path, predictor):
    # 1.口角纹：Mouth and corner lines  2.法令纹：Nasal pattern 3.鱼尾纹：Crow’s feet
    # 4.眼袋：Bags under the eyes # 5.抬头纹：forehead lines 6.印第安纹：Indian pattern
    # 7.黑眼圈：Dark circles 8.眼周细纹：Fine lines around the eyes # 9.泪沟：Tear trough
    class_names = ["Bags under the eyes", "Crow feet", "Dark circles", "Fine lines around the eyes",
                   "Indian pattern", "Mouth and corner lines", "Nasal pattern", "Tear trough", "forehead lines"]

    outputs = predictor(image_path)
    img_np_copy = image_path.copy()

    pred_boxes = tensor_to_json(getattr(outputs["instances"].pred_boxes, "tensor"))
    pred_classes = tensor_to_json(outputs["instances"].pred_classes)
    pred_scores = tensor_to_json(outputs["instances"].scores)
    # print(pred_scores, pred_classes, pred_scores)
    pred_classes_array = outputs["instances"].pred_classes.cpu().numpy()
    pred_boxes_array = getattr(outputs["instances"].pred_boxes, "tensor").cpu().numpy()

    # instance_mode:
    IMAGE = 0
    """
    Picks a random color for every instance and overlay segmentations with low opacity.
    """
    SEGMENTATION = 1
    """
    Let instances of the same category have similar colors
    (from metadata.thing_colors), and overlay them with
    high opacity. This provides more attention on the quality of segmentation.
    """
    IMAGE_BW = 2
    """
    Same as IMAGE, but convert all areas without masks to gray-scale.
    Only available for drawing per-instance mask predictions.
    """

    v = Visualizer(image_path[:, :, ::-1], metadata={'thing_classes': class_names},
                   scale=0.5, instance_mode=ColorMode.SEGMENTATION)
    # print(type(v))
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # gray = cv2.cvtColor(v.get_image(), cv2.COLOR_BGR2GRAY)
    img_np = cv2.cvtColor(v.get_image(), cv2.COLOR_BGR2RGB)

    # -------------------------对指定类别的皱纹进行特殊处理，加入索贝算子突出皱纹的纹路---------------------------------------------------------
    loc = []  # 存放特定皱纹的矩形框坐标
    for box, class_name, scores in zip(getattr(outputs["instances"].pred_boxes, "tensor"),
                                       outputs["instances"].pred_classes,
                                       outputs["instances"].scores):
        if class_name == 3:  # 3表示眼袋
            loc.append(box)
    two_dim_coords = []
    for i in loc:
        a = [int(i[0]), int(i[1])]
        b = [int(i[2]), int(i[3])]
        two_dim_coords.append([a, b])

    # 对矩形框内的像素进行处理
    for i in two_dim_coords:
        # print(i[0][0],i[1][0], i[0][1],i[1][1])
        x1 = i[0][0]
        x2 = i[1][0]
        y1 = i[0][1]
        y2 = i[1][1]

        # print(i)
        # for x in range(i[0][0], i[1][0]):
        #     for y in range(i[0][1], i[1][1]) :
        #         # 在这里对像素进行处理，这里示例为将像素设为纯黑色
        #         image[x, y] = (0, 0, 0)
        gray = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray[int(x1 / 2):int(x2 / 2), int(y1 / 2):int(y2 / 2)], 30, 100)  # 参数可以根据具体情况调整
        coordinates = np.where(edges == 255)
        aa = img_np.shape[1] - coordinates[0] - x1 / 2
        bb = img_np.shape[0] - coordinates[1] - y1 / 2

        for x, y in zip(aa, bb):
            # print(x,y)
            img_np[int(-y), int(-x)] = (255, 0, 0)

    height, width = img_np.shape[:2]
    new_size = (width * 2, height * 2)
    # 像素提高两倍，还原为输入图像相同的规格
    img_np = cv2.resize(img_np, new_size, interpolation=cv2.INTER_NEAREST)

    # ----------------- 启动蒙版处理-----------------
    if len(outputs["instances"].pred_classes) != 0:
        # 创建一个全透明的图像
        transparent_image = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
        transparent_image[:, :, :] = 255  # 设置为全白，即完全不透明

        # 根据框的位置来绘制蒙版
        for i in getattr(outputs["instances"].pred_boxes, "tensor").cpu().numpy():
            for x1, y1, x2, y2 in [i]:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # 将框内的区域设置为完全不透明
                transparent_image[y1:y2, x1:x2, :] = 255
                mask = np.zeros_like(transparent_image)
                mask[y1:y2, x1:x2, :] = 1
                # 设置框外的区域透明度为90%
                transparent_image = transparent_image * (
                            1 - mask * 0.9) + mask * transparent_image  # mask系数越大，透明度越小，0.9越小，越透明
                transparent_image = transparent_image.astype(np.uint8)
        # 将蒙版应用到图像
        img_np = cv2.addWeighted(img_np, 1, transparent_image, 0.6, 0)

    #--------------------------- 单独画框 ----------------------------
    img_another = boxesToImage(pred_boxes_array, img_np_copy)

    # image = Image.fromarray(img_np)
    image = Image.fromarray(img_another)
    return pred_boxes, pred_boxes_array, pred_classes, pred_classes_array, pred_scores, image


def on_Video(videoPath, predictor):
    class_names = ["five", "four", "one", "three", "two"]
    cap = cv2.VideoCapture(videoPath)
    if (cap.isOpened() == False):
        print("Error opening file...")
        return

    (success, image) = cap.read()
    while success:
        predictions = predictor(image)
        v = Visualizer(image[:, :, ::-1], metadata={'thing_classes': class_names}, scale=0.5,
                       instance_mode=ColorMode.SEGMENTATION)
        output = v.draw_instance_predictions(predictions["instances"].to("cpu"))

        # cv2.imread("Reuslt", output.get_image()[:,:,::-1])
        # cv2.namedWindow("result", 0)
        # cv2.resizeWindow("result", 1200, 600)

        #调用电脑摄像头进行检测
        cv2.namedWindow("result", cv2.WINDOW_FREERATIO)  # 设置输出框的大小，参数WINDOW_FREERATIO表示自适应大小
        cv2.imshow("result", output.get_image()[:, :, ::-1])

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        (success, image) = cap.read()


# -------------------  计算皱纹是左边还是右边-------------------
# -------------------框坐标是对角两个点的坐标，以图像左上角为坐标远点 -----------------------
def calcLeftRight(imageSize, pred_boxes, pred_classes):
    leftOrRight = imageSize[0] / 2  # 左右
    # upOrDown = imageSize[1] / 2  # 上下
    predBoxesNew = []
    for i, j in zip(pred_boxes, pred_classes):
        Label = 'Left' if leftOrRight >= 0.5 * (i[0] + i[2]) else 'Right'
        predBoxesNew.append(Label + " : " + str(j))
    return predBoxesNew


def boxesToImage(boxes, image):
    for box in boxes:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        r, g, b, = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), (r, g, b), 3)  # (0, 0, 255) 是红色, 2 是线条的厚度
    return image
