import os
import configparser
import time

import cv2
import numpy as np
import oss2 as oss2
import requests
from flask import Flask, request, send_file, jsonify, render_template
from facedetect import face_text
from PIL import Image
import io
import json

from numpy import shape

from detectron2.engine import DefaultPredictor
import base64
import pickle
from utils import on_Image

cfg_save_path = "OD_cfg.pickle"

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = "./model_final.pth"

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor_path = "./shape_predictor_68_face_landmarks.dat"
face_detect = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
predictor = DefaultPredictor(cfg)

# 读取阿里云配置
config = configparser.ConfigParser()
config.read('./config/config.ini')
access_id = config.get("oss-config", 'access_id')
access_key = config.get("oss-config", 'access_key')
bucket_name = config.get("oss-config", 'bucket_name')
endpoint = config.get("oss-config", 'endpoint')


# 连接阿里云
auth = oss2.Auth(access_id, access_key)
bucket = oss2.Bucket(auth, endpoint, bucket_name)


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['JSON_AS_ASCII'] = False

def validate(args):
    validations = {
        'img_url': 'img_url is required',
        'id': 'id is required'
    }
    for key, message in validations.items():
        if key not in args:
            return ValueError(message)




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data', methods=['POST'])
def get_data():
    res = []
    if request.method != 'POST':
        return jsonify({'error': 'Only POST requests are allowed'}), 400

    if validate(request.json):
        return jsonify({'error': 'Parameters {}'.format(validate(request.json))}), 400

    try:
        saveUrlPrefix = "skinrun-face/" + time.strftime('%Y%m%d', time.localtime()) + "/" + \
                          str(request.json['img_original']) + "/" + \
                        str(request.json['img_type']) + "/"
        img_url = request.json['img_url']
        local_file = (bucket.get_object(img_url)).read()
        img_array = np.asarray(bytearray(local_file), dtype=np.uint8)
        img_np = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        # ------------------------- 识别图像 -----------------------
        pred_boxes, pred_classes, pred_scores, image = on_Image(img_np, predictor)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        if bucket.object_exists(saveUrlPrefix + "RecognitionImg.jpg"):
            # 删除文件
            bucket.delete_object(saveUrlPrefix + "RecognitionImg.jpg")
        bucket.put_object(saveUrlPrefix + "RecognitionImg.jpg", img_byte_arr)

        # ------------------------- 肤色 -----------------------
        color = face_text.colorList().get_color(img_np, face_detect)  # 肤色

        # ------------------------- 轮廓 -----------------------
        allPoints, contourPoints, imageContour = face_text.contour().contourImg(img_np, predictor_path)
        img_byte_arr = io.BytesIO()
        imageContour.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        if bucket.object_exists(saveUrlPrefix + "ContourImg.jpg"):
            # 删除文件
            bucket.delete_object(saveUrlPrefix + "ContourImg.jpg")
        bucket.put_object(saveUrlPrefix + "ContourImg.jpg", img_byte_arr)

        # ------------------------- 敏感肌 -----------------------
        sensitiveSkinImg = face_text.sensitiveSkin().sensitiveSkinImg(img_np)
        img_byte_arr = io.BytesIO()
        sensitiveSkinImg.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        if bucket.object_exists(saveUrlPrefix + "sensitiveSkinImg.jpg"):
            # 删除文件
            bucket.delete_object(saveUrlPrefix + "sensitiveSkinImg.jpg")
        bucket.put_object(saveUrlPrefix + "sensitiveSkinImg.jpg", img_byte_arr)

        return jsonify({"status_code": "200",
                        "message": "操作成功",
                        "response_data": {'pred_classes': pred_classes, 'pred_scores': pred_scores, 'color': color,
                                          "image_url": {
                                              "image_contour_url": saveUrlPrefix + "ContourImg.jpg",
                                              "image_sensitive_url": saveUrlPrefix + "sensitiveSkinImg.jpg",
                                              "image_recognition_url": saveUrlPrefix + "RecognitionImg.jpg",
                                          }}})
    except:
        return jsonify({"status_code": "400", "message": "请求失败", "response_data": {}})

@app.route('/ContourImg.jpg', methods=['GET'])
def getContourImg():
    image_array = 'ContourImg.jpg'

    # 将数组转换为列表，以便于JSON序列化
    # image_array_list = image_array.tolist()

    return send_file(image_array, mimetype='image/jpeg')



@app.route('/SensitiveSkinImg.jpg', methods=['GET'])
def getSensitiveSkinImg():
    image_array = 'SensitiveSkinImg.jpg'

    # 将数组转换为列表，以便于JSON序列化
    # image_array_list = image_array.tolist()

    return send_file(image_array, mimetype='image/jpeg')


@app.route('/RecognitionImg.jpg', methods=['GET'])
def getRecognitionImg():
    image_array = 'RecognitionImg.jpg'

    # 将数组转换为列表，以便于JSON序列化
    # image_array_list = image_array.tolist()

    return send_file(image_array, mimetype='image/jpeg')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)
