# -*- coding: utf-8 -*-
from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode

import numpy as np, os, glob, cv2, json, random
import matplotlib.pyplot as plt
import tensorflow as tf
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from detectron2.data import DatasetCatalog, build_detection_test_loader

from detectron2.evaluation import inference_on_dataset, COCOEvaluator
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.evaluation.pascal_voc_evaluation import PascalVOCDetectionEvaluator

__all__ = ["load_voc_instances", "register_pascal_voc"]


CLASS_NAMES = (
    "bonnet",
    "bumper",
    "car",
    "center_reflector",
    "front_left_door",
    "front_left_wheel",
    "front_left_door_glass",
    "front_left_door_handle",
    "front_license_plate",
    "front_right_door",
    "front_right_wheel",
    "front_right_door_handle",
    "front_right_door_glass",
    "front_windshield",
    "left_a_pillar",
    "left_fender",
    "left_headlamp",
    "left_mirror",
    "left_quarter_panel",
    "left_reflector",
    "left_rocker_panel",
    "left_roof_rail",
    "left_fog_light",
    "left_fog_light_lamp",
    "left_fender_indicator",
    "left_taillamp",
    "left_quarter_glass",
    "left_wiper",
    "left_fender_emblem",
    "lower_bumper_grill",
    "make",
    "model",
    "petrol_cap",
    "rear_bumper",
    "rear_left_door",
    "rear_left_wheel",
    "rear_license_plate",
    "rear_right_door",
    "rear_right_wheel",
    "rear_windshield",
    "rear_windshield_taillight",
    "rear_emblem",
    "rear_left_door_handle",
    "rear_left_door_glass",
    "rear_right_door_glass",
    "rear_right_door_handle",
    "right_a_pillar",
    "right_fender",
    "right_headlamp",
    "right_mirror",
    "right_quarter_panel",
    "right_reflector",
    "right_rocker_panel",
    "right_roof_rail",
    "right_taillamp",
    "right_wiper",
    "right_fog_light",
    "right_fog_light_lamp",
    "right_fender_indicator",
    "right_fender_emblem",
    "right_quarter_glass",
    "other_emblem",
    "spoiler",
    "spoiler_reflector",
    "trunk",
    "trunk_handle",
    "upper_bumper_grill",
    "front_emblem",
    "varient",
)

dicts = []

input_dir = "/home/manusaravia/data/part_detection"

isGPU = True

if not isGPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)
)
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)


def load_voc_instances(dirname, split, CLASS_NAMES):
    with PathManager.open(os.path.join(dirname, split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(dirname, "annotations", fileid + ".xml")
        jpeg_file = os.path.join(dirname, "images", fileid + ".jpg")

        tree = ET.parse(anno_file)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {
                    "category_id": CLASS_NAMES.index(cls),
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                }
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_pascal_voc(name, dirname, split, CLASS_NAMES):
    if name not in DatasetCatalog.list():
        DatasetCatalog.register(
            name, lambda: load_voc_instances(dirname, split, CLASS_NAMES)
        )

    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES, dirname=dirname, split=split, year=2007
    )


register_pascal_voc(
    "my_dataset_train", dirname=input_dir, split="train", CLASS_NAMES=CLASS_NAMES
)

register_pascal_voc(
    "my_dataset_test", dirname=input_dir, split="test", CLASS_NAMES=CLASS_NAMES
)


cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")
)
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_test",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"
)
cfg.SOLVER.BASE_LR = 0.0001
cfg.SOLVER.WARMUP_ITERS = 500
cfg.SOLVER.MAX_ITER = 15000
cfg.SOLVER.STEPS = (2000, 2500)
cfg.SOLVER.GAMMA = 0.05
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.TEST.EVAL_PERIOD = 2000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
# cfg.INPUT.MIN_SIZE_TRAIN = 800
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)  # lo cambio a True cuando solo quiero que evalue
trainer.train()


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

# evaluator = PascalVOCDetectionEvaluator("my_dataset_test")
# con este no me da error y con el propio de pascal si
evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "my_dataset_test")
print(inference_on_dataset(trainer.model, val_loader, evaluator))


dataset_dicts = DatasetCatalog.get("my_dataset_test")
my_metadata = MetadataCatalog.get("my_dataset_test")

i = 0
k = 10
for d in random.sample(dataset_dicts, k):
    i += 1
    print(f'image{i}: {d["file_name"]}')
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata=my_metadata, scale=0.8)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imshow("image", v.get_image()[:, :, ::-1])
    plt.imshow(out.get_image())
    plt.xticks([]), plt.yticks([])
    plt.show()
    plt.savefig(input_dir + "/outputs/image" + str(i) + "test.jpg")
