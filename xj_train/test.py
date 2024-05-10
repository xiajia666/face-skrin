from detectron2.engine import DefaultPredictor
import os
import pickle
from utils import *

cfg_save_path = "OD_cfg.pickle"

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
print(cfg.MODEL.WEIGHTS)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

image_path = "./images/train/115.jpg"
a = cv2.imread(image_path)
on_Image(a, predictor)

# video_path = ""
# on_Video(0, predictor)#0 表示调用电脑的摄像头来实时预测
