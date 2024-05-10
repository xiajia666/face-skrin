from detectron2.utils.logger import setup_logger

setup_logger()
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.engine import DefaultTrainer

import os
import pickle

from utils import *

# config_file_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
# checkpoint_url = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"


'''
instance segmentation
'''
config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
checkpoint_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

output_dir = "./output/number_detection"


num_classes = 9
# class_names = ["five","four","one","three","two"]
class_names = ["Bags under the eyes", "Crow feet", "Dark circles", "Fine lines around the eyes",
               "Indian pattern", "Mouth and corner lines", "Nasal pattern", "Tear trough", "forehead lines"]
# class_names = ['doudou', 'falingwen', 'meimao', 'moouxian', 'nose']


device = "cuda"

train_dataset_name = "custom_train"
train_images_path = "./images/train/"
train_json_annot_path = "xj_train/images/train/train.json"

test_dataset_name = "custom_test"
test_images_path = "images/test/"
test_json_annot_path = "/xj_train/images/test/test.json"

cfg_save_path = "OD_cfg.pickle"


###########################################################
# 注册训练集

DatasetCatalog.register("custom_train", lambda: load_coco_json(train_json_annot_path, train_images_path, "custom_train"))

# register_coco_instances("custom_train", {},train_json_annot_path, train_images_path)
MetadataCatalog.get("custom_train").set(thing_classes = class_names,
                                    evaluator_type = 'coco',
                                    json_file=train_json_annot_path,
                                    image_root=train_images_path)
# print(MetadataCatalog.get("custom_train"))

# 注册测试集
DatasetCatalog.register("custom_test", lambda: load_coco_json(test_json_annot_path, test_images_path, "custom_test"))

# register_coco_instances("custom_test", {}, test_json_annot_path, test_images_path)
MetadataCatalog.get("custom_test").set(thing_classes = class_names,
                                    evaluator_type = 'coco',
                                    json_file=test_json_annot_path,
                                    image_root=test_images_path)
# plot_samples(dataset_name=train_dataset_name, n=3)

#####################################################
def main():
    cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir)
    # cfg.DATALOADER.NUM_WORKERS = 2

    with open(cfg_save_path, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    trainer.train()

if __name__ == '__main__':
    main()
