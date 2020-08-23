import matplotlib
import matplotlib.pyplot as plt
from time import time
import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import csv
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

# Đọc dữ liệu từ video
#%%
import sys
import os
import cv2
import argparse
from tqdm import tqdm


# @markdown Your videos is stored in:

input_dir = "../sample_datadata/videos"

# @markdown  Frames extracted from videos will be stored in:
output_dir = "../sample_datadata/frames"


video_paths = []
for r, d, f in os.walk(input_dir):
    for file in f:
        if ".mp4" in file:
            video_paths.append(os.path.join(r, file))


for video_path in video_paths:
    print(video_path)


for video_path in video_paths:
    video_dir_path = os.path.join(
        output_dir, os.path.splitext(os.path.basename(video_path))[0]
    )
    if not os.path.isdir(video_dir_path):
        os.makedirs(video_dir_path)

    vid_cap = cv2.VideoCapture(video_path)
    num_frms, original_fps = (
        int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        vid_cap.get(cv2.CAP_PROP_FPS),
    )

    ## Number of skip frames
    time_stride = 1

    for frm_id in tqdm(range(0, num_frms, time_stride)):
        vid_cap.set(cv2.CAP_PROP_POS_FRAMES, frm_id)
        _, im = vid_cap.read()

        cv2.imwrite(os.path.join(video_dir_path, str(frm_id) + ".jpg"), im)


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
    img_data = tf.io.gfile.GFile(path, "rb").read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def get_keypoint_tuples(eval_config):
    """Return a tuple list of keypoint edges from the eval config.

  Args:
    eval_config: an eval config containing the keypoint edges

  Returns:
    a list of edge tuples, each in the format (start, end)
  """
    tuple_list = []
    kp_list = eval_config.keypoint_edge
    for edge in kp_list:
        tuple_list.append((edge.start, edge.end))
    return tuple_list


def get_model_zoo_list(model_zoo_file):
    """Return a dictionary of model with config and pretrained weight.

    Args:
      eval_config: an eval config containing the keypoint edges

    Returns:
      a dict of tuples, each in the format model_name:(config_file, pretrained_weight_link)
    """
    model_zoo_dict = dict()
    with open(model_zoo_file) as csvfile:
        model_reader = csv.reader(csvfile, delimiter=",")
        for row in model_reader:
            model_zoo_dict[row[0]] = (row[1], row[2])

    return model_zoo_dict


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
    img_data = tf.io.gfile.GFile(path, "rb").read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def get_keypoint_tuples(eval_config):
    """Return a tuple list of keypoint edges from the eval config.

  Args:
    eval_config: an eval config containing the keypoint edges

  Returns:
    a list of edge tuples, each in the format (start, end)
  """
    tuple_list = []
    kp_list = eval_config.keypoint_edge
    for edge in kp_list:
        tuple_list.append((edge.start, edge.end))
    return tuple_list


model_zoo_dict = get_model_zoo_list("model_zoo.txt")
model_name = "CenterNet HourGlass104 512x512"
model_config_file, model_weight_file = model_zoo_dict[model_name]
model_weight_link = (
    "http://download.tensorflow.org/models/object_detection/tf2/20200711/"
    + model_weight_file
)

pipeline_config = os.path.join(
    "models/research/object_detection/configs/tf2/", model_config_file
)
model_dir = model_weight_file[:-7] + "/checkpoint"

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs["model"]
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(model_dir, "ckpt-0")).expect_partial()


def get_model_detection_function(model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn


detect_fn = get_model_detection_function(detection_model)

label_map_path = "models/research/object_detection/data/mscoco_label_map.pbtxt"
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True,
)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

image_dir = output_dir + "/sample_01"

import json


def load_zone_anno(json_filename):
    """
  Load the json with ROI and MOI annotation.

  """
    with open(json_filename) as jsonfile:
        dd = json.load(jsonfile)
        polygon = [(int(x), int(y)) for x, y in dd["shapes"][0]["points"]]
        paths = {}
        for it in dd["shapes"][1:]:
            kk = str(int(it["label"][-2:]))
            paths[kk] = [(int(x), int(y)) for x, y in it["points"]]
    return polygon, paths


polygon, paths = load_zone_anno("../sample_data/videos/sample_01.json")


import bb_polygon


def check_bbox_intersect_polygon(polygon, bbox):
    """

  Args:
    polygon: List of points (x,y)
    bbox: A tuple (xmin, ymin, xmax, ymax)

  Returns:
    True if the bbox intersect the polygon
  """
    x1, y1, x2, y2 = bbox
    bb = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return bb_polygon.is_bounding_box_intersect(bb, polygon)


from sort import *

moto_tracker = Sort()
truck_tracker = Sort()

# Create an motobikes tracker with default parameter.
# Please read the sort documentation for the custom paramenters.

moto_tracker = Sort()

# If you want to track another vehicle class, you need to declare a new tracker.
# truck_tracker = Sort()

track_dict = {}

N_FRAMES = 20

for frame_id in range(1, N_FRAMES):
    image_path = os.path.join(image_dir, "{}.jpg".format(frame_id))
    image_np = load_image_into_numpy_array(image_path)

    im_width, im_height, _ = image_np.shape
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    boxes = detections["detection_boxes"][0]
    scores = detections["detection_scores"][0]
    classes = detections["detection_classes"][0]

    dets = []
    for bb, s, c in zip(boxes, scores, classes):
        ymin, xmin, ymax, xmax = bb.numpy()
        xmin, ymin, xmax, ymax = (
            xmin * im_width,
            ymin * im_height,
            xmax * im_width,
            ymax * im_height,
        )
        if check_bbox_intersect_polygon(polygon, (xmin, ymin, xmax, ymax)):
            # check if the bbox is in ROI
            dets.append((frame_id, c.numpy(), xmin, ymin, xmax, ymax, s.numpy()))

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    dets = np.array(dets)

    # Only get the detections with the class label is '3' which indicate the motobike class.
    moto_dets = dets[dets[:, 1] == 3]
    moto_dets = np.array(moto_dets)

    trackers = moto_tracker.update(moto_dets)
    for xmin, ymin, xmax, ymax, track_id in trackers:
        track_id = int(track_id)
        # print(track_id)
        if track_id not in track_dict.keys():
            track_dict[track_id] = [(xmin, ymin, xmax, ymax, frame_id)]
        else:
            track_dict[track_id].append((xmin, ymin, xmax, ymax, frame_id))


# moto_vector_list: list of tuples (first_point, last_point, last_frame_id)
# list of moto movement vector and the last frame_id when it is still in the ROI.

moto_vector_list = []
for tracker_id, tracker_list in track_dict.items():
    if len(tracker_list) > 1:
        first = tracker_list[0]
        last = tracker_list[-1]
        first_point = ((first[2] - first[0]) / 2, (first[3] - first[1]) / 2)
        last_point = ((last[2] - last[0]) / 2, (last[3] - last[1]) / 2)
        moto_vector_list.append((first_point, last_point, last[4]))


def cosin_similarity(a2d, b2d):

    a = np.array((a2d[1][0] - a2d[0][0], a2d[1][1] - a2d[0][1]))
    b = np.array((b2d[1][0] - b2d[0][1], b2d[1][1] - b2d[1][0]))
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


MOTO_CLASS_ID = 1

# Đếm số lương
def counting_moi(paths, moto_vector_list):
    """
  Args:
    paths: List of MOI - (first_point, last_point)
    moto_vector_list: List of tuples (first_point, last_point, last_frame_id)

  Returns:
    A list of tuples (frame_id, movement_id, vehicle_class_id)
  """
    moi_detection_list = []
    for moto_vector in moto_vector_list:
        max_cosin = -2
        movement_id = ""
        last_frame = 0
        for movement_label, movement_vector in paths.items():
            cosin = cosin_similarity(movement_vector, moto_vector)
            if cosin > max_cosin:
                max_cosin = cosin
                movement_id = movement_label
                last_frame = moto_vector[2]

        moi_detection_list.append((last_frame, movement_id, MOTO_CLASS_ID))
    return moi_detection_list


moto_moi_detections = counting_moi(paths, moto_vector_list)
print(moto_moi_detections)

result_filename = "result.txt"
video_id = "000"
with open(result_filename, "w") as result_file:
    for frame_id, movement_id, vehicle_class_id in moto_moi_detections:
        result_file.write(
            "{} {} {} {}\n".format(video_id, frame_id, movement_id, vehicle_class_id)
        )

