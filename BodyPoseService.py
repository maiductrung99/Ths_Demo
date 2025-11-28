import pandas as pd
import os
import sys
sys.path.append(os.path.abspath("body_pose\\openposeBodyCustom"))
from openpose_pytorch import torch_openpose
import cv2
import torch
import time
import numpy as np
import pandas as pd
import math
import load_config as config

class BodyPoseService:

  def __init__(self):
    loadConfig = config.yaml_load().get("body-pose")
    model_type = "body_25"
    model_path = "body_25.pth"
    self.model = torch_openpose.torch_openpose(model_type=model_type,
                                       model_path=model_path)
  """
  Method to extraction body data from detections 

  Parameters
  ----------
  img: is path file input

  Return
  ------
  Frame Data related Body pose
  """
  def extractionBodyPose(self, person):
    result_body = None
    dataBody = self.handleOpenPosePytorch(tp = self.model, frame = person)
    if type(dataBody) == list:
      dataBody = pd.DataFrame(dataBody)
    result_body = pd.concat([result_body, dataBody], ignore_index=True)
    return result_body

  def calculate_angle(self,x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    angle_ba = math.atan2(dy, dx)
    return angle_ba

  def handleOpenPosePytorch(self,tp , frame):
    num_bf = 25
    start = time.time()
    flag_break = False
    iteration_ = 0
    iteration_ += 1
    img = frame
    image_width = img.shape[1]
    image_height = img.shape[0]
    df = []
    poses = tp(img)

    input_features = np.ones((1, num_bf)) * (1e-6)
    iter_features = 0
    if poses:
      # 10 point after apply OpenPose -> 20 first feature
      for i in [0, 1, 2, 3, 4, 5, 6, 7, 15, 16]:
        if len(poses[0]) > i:
          normalized_x = poses[0][i][0] / image_width
          input_features[0][iter_features] = normalized_x
          iter_features += 1
          normalized_y = poses[0][i][1] / image_height
          input_features[0][iter_features] = normalized_y
          iter_features += 1
          # 45 feature distance between each point
      #for i in range(10):
      #  for j in range(i + 1, 10):
      #    distance = math.sqrt((input_features[0][2 * i] - input_features[0][2 * j]) ** 2 + (
      #            input_features[0][2 * i + 1] - input_features[0][2 * j + 1]) ** 2)
      #    input_features[0][iter_features] = distance
      #    iter_features = iter_features + 1
      # calculate angle of 5 last feature

      # point 0 with point 1
      angle_01 = self.calculate_angle(input_features[0][0], input_features[0][1], input_features[0][2],
                                 input_features[0][3])
      angle_12h = math.pi / 2
      angle_clockwise_01 = angle_01 - angle_12h
      if angle_clockwise_01 < 0:
        angle_clockwise_01 += 2 * math.pi
      input_features[0][iter_features] = angle_clockwise_01
      iter_features += 1
      # point 2 with point 3
      angle_23 = self.calculate_angle(input_features[0][4], input_features[0][5], input_features[0][6],
                                 input_features[0][7])
      angle_9h = math.pi
      angle_anticlockwise_23 = angle_9h - angle_23
      if angle_anticlockwise_23 < 0:
        angle_anticlockwise_23 += 2 * math.pi
      input_features[0][iter_features] = angle_anticlockwise_23
      iter_features += 1

      # point 3 with point 4
      angle_34 = self.calculate_angle(input_features[0][6], input_features[0][7], input_features[0][8],
                                 input_features[0][9])
      angle_12h = math.pi / 2
      angle_clockwise_34 = angle_34 - angle_12h
      if angle_clockwise_34 < 0:
        angle_clockwise_34 += 2 * math.pi
      input_features[0][iter_features] = angle_clockwise_34
      iter_features += 1

      # point 5 with point 6
      angle_56 = self.calculate_angle(input_features[0][10], input_features[0][11], input_features[0][12],
                                 input_features[0][13])
      angle_3h = 0
      angle_anticlockwise_56 = angle_56 - angle_3h
      if angle_anticlockwise_56 < 0:
        angle_anticlockwise_56 += 2 * math.pi
      input_features[0][iter_features] = angle_anticlockwise_56
      iter_features += 1

      # point 6 with point 7
      angle_67 = self.calculate_angle(input_features[0][12], input_features[0][13], input_features[0][14],
                                 input_features[0][15])
      angle_12h = math.pi / 2
      angle_clockwise_67 = angle_12h - angle_67
      if angle_clockwise_67 < 0:
        angle_clockwise_67 += 2 * math.pi
      input_features[0][iter_features] = angle_clockwise_67
      iter_features += 1

      df = pd.DataFrame(input_features)
      if iteration_ != 0 and iteration_ % 40 == 0:
        print("\tImage Processed: ", iteration_)
    end = time.time()
    return df
