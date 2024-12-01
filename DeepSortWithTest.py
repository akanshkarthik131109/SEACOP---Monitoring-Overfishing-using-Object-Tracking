from __future__ import absolute_import
import numpy as np
'''
Class for Detection Objects

'''
class Detection(object):

  '''
  xywh: center x; center y; width; height
  '''
  def __init__(self, tlwh, conf, feature, cls):
    self.tlwh = np.asarray(tlwh, dtype=np.float32)
    self.conf = float(conf)
    self.feature = np.asarray(feature, dtype=np.float32)
    self.cls = cls

  def toCornerPoints(self):
    ret = self.tlwh.copy()
    ret[2:] += ret[:2]
    return ret

  def to_xyah(self):
    ret = self.tlwh.copy()
    ret[:2] += ret[2:] / 2
    ret[2] /= ret[3]
    return ret

'''
Kalman Filter: To help create the predicitons of where each object will go in the next frame

'''

import numpy as np
import scipy.linalg

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

class KalmanFilter(object):

  def __init__(self):
    ndim, dt = 4, 1.

    self._motion_mat = np.eye(2*ndim, 2*ndim)
    for i in range(ndim):
      self._motion_mat[i, ndim + i] = dt
    self._update_mat = np.eye(ndim, 2 * ndim)

    self._std_weight_position = 1. / 20
    self._std_weight_velocity = 1. / 160

  def initiate(self, measurement):
    '''
    measurement is an ndarray of xyah
    '''

    mean_pos = measurement
    mean_vel = np.zeros_like(mean_pos)
    mean = np.r_[mean_pos, mean_vel]

    std = [
        2 * self._std_weight_position * measurement[3],
        2 * self._std_weight_position * measurement[3],
        1e-2,
        2 * self._std_weight_position * measurement[3],
        10 * self._std_weight_velocity * measurement[3],
        10 * self._std_weight_velocity * measurement[3],
        1e-5,
        10 * self._std_weight_velocity * measurement[3]]
    covariance = np.diag(np.square(std))
    return mean, covariance

  def predict(self, mean, covariance):
    std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
    std_vel = [
        self._std_weight_velocity * mean[3],
        self._std_weight_velocity * mean[3],
        1e-5,
        self._std_weight_velocity * mean[3]]
    motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

    mean = np.dot(self._motion_mat, mean)
    covariance = np.linalg.multi_dot((
        self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

    return mean, covariance

  def project(self, mean, covariance):
    std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
    innovation_cov = np.diag(np.square(std))

    mean = np.dot(self._update_mat, mean)
    covariance = np.linalg.multi_dot((
        self._update_mat, covariance, self._update_mat.T))
    return mean, covariance + innovation_cov

  def update(self, mean, covariance, measurement):

    projected_mean, projected_cov = self.project(mean, covariance)

    chol_factor, lower = scipy.linalg.cho_factor(
        projected_cov, lower=True, check_finite=False)
    kalman_gain = scipy.linalg.cho_solve(
        (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
        check_finite=False).T
    innovation = measurement - projected_mean

    new_mean = mean + np.dot(innovation, kalman_gain.T)
    new_covariance = covariance - np.linalg.multi_dot((
        kalman_gain, projected_cov, kalman_gain.T))
    return new_mean, new_covariance

  def gating_distance(self, mean, covariance, measurements, only_position=False):

    mean, covariance = self.project(mean, covariance)
    if only_position:
        mean, covariance = mean[:2], covariance[:2, :2]
        measurements = measurements[:, :2]

    cholesky_factor = np.linalg.cholesky(covariance)
    d = measurements - mean
    z = scipy.linalg.solve_triangular(
        cholesky_factor, d.T, lower=True, check_finite=False,
        overwrite_b=True)
    squared_maha = np.sum(z * z, axis=0)
    return squared_maha

'''
Linear assignment section: Hungarian algorithm to match the new detections to the ID's of the previous frame
'''

import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment


INFTY_COST = 1e+5

def min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices = None, detection_indices = None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
      return [], track_indices, detection_indices

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    cost_matrix = np.array(distance_metric(tracks, detections, track_indices, detection_indices))
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    indices = linear_assignment(cost_matrix)
    row_indices, col_indices = indices
    indices = np.asarray(indices).reshape(-1, 2)
    print(f'indicies: {len(indices)}')
    matches, unmatched_tracks, unmatched_detections = [], [], []

    for col, detection_idx in enumerate(detection_indices):
      if col not in col_indices:
        unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
      if row not in row_indices:
        unmatched_tracks.append(track_idx)
    for row, col in indices:
      track_idx = track_indices[row]
      detection_idx = detection_indices[col]
      if cost_matrix[row, col] > max_distance:
        unmatched_tracks.append(track_idx)
        unmatched_detections.append(detection_idx)
      else:
        matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections

def matching_cascade(distance_metric, max_distance, cascade_depth, tracks, detections, track_indices=None, detection_indices=None):

  if track_indices is None:
    track_indices = list(range(len(tracks)))
  if detection_indices is None:
    detection_indices = list(range(len(detections)))

  unmatched_detections = detection_indices
  matches = []

  for level in range(cascade_depth):
    if len(unmatched_detections) == 0:
      break
    track_indices_in_level = [k for k in track_indices if tracks[k].time_since_update == 1+level]

    if len(track_indices_in_level) == 0:
      continue

    matches_in_level, _, unmatched_detections = min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices_in_level, unmatched_detections)
    matches += matches_in_level

  unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))

  return matches, unmatched_tracks, unmatched_detections

def gate_cost_matrix(kf, cost_matrix, tracks, detections, track_indices, detection_indices, gated_cost=INFTY_COST, only_position=False):
  gating_dim = 2 if only_position else 4
  gating_threshold = chi2inv95[gating_dim]
  #cost_matrix = cost_matrix.reshape((len(track_indices), len(detection_indices)))
  print(f"Detction Indices: {len(detection_indices)}")
  measurements = np.asarray(
      [detections[i].to_xyah() for i in detection_indices])
  
  
  for row, track_idx in enumerate(track_indices):
      track = tracks[track_idx]
      
      gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)
      print("cost_matrix[row, :].shape:", cost_matrix[row, :].shape)
      print("gating_distance.shape:", gating_distance.shape)
      cost_matrix[row, gating_distance > gating_threshold] = gated_cost
  return cost_matrix

'''
IoU Matching: Intersection over Union to help match the detections/tracks together across frames
'''


import numpy as np

def iou(bbox, wanted_class, candidates, classes):

  for i in range(len(classes)):
    if classes[i] == wanted_class:
      continue
    else:
      candidates[i] = np.array([0, 0, 0, 0])

  bbox_top_left, bbox_bottom_right = bbox[:2], bbox[:2] + bbox[2:]
  candidates_top_left = candidates[:, :2]
  candidates_bottom_right = candidates[:, :2] + candidates[:, 2:]

  print("bbox_top_left shape:", bbox_top_left.shape)
  print("candidates_top_left shape:", candidates_top_left.shape)
  print("First part shape:", np.maximum(bbox_top_left[0], candidates_top_left[:, 0]).shape)
  print("Second part shape:", np.maximum(bbox_top_left[1], candidates_top_left[:, 1]).shape)

  top_left = np.c_[np.maximum(bbox_top_left[0], candidates_top_left[:, 0])[:, np.newaxis],
                 np.maximum(bbox_top_left[1], candidates_top_left[:, 1])[:, np.newaxis]]
  bottom_right = np.c_[np.minimum(bbox_bottom_right[0], candidates_bottom_right[:, 0])[:, np.newaxis], 
                       np.minimum(bbox_bottom_right[1], candidates_bottom_right[:, 1])[:, np.newaxis]]
  width_height = np.maximum(0., bottom_right - top_left)

  area_intersection = width_height.prod(axis=1)
  area_bbox = bbox[2:].prod()
  area_candidates = candidates[:, 2:].prod(axis=1)

  return area_intersection / (area_bbox + area_candidates - area_intersection)

def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
  if track_indices is None:
    track_indices = np.arange(len(tracks))
  if detection_indices is None:
    detection_indices = np.arange(len(detections))

  cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
  for row, track_idx in enumerate(track_indices):
    if tracks[track_idx].time_since_update > 1:
      cost_matrix[row, ] = INFTY_COST
      continue

    #Method in Tracks class
    bbox = tracks[track_idx].to_tlwh_Track()
    wanted_class = tracks[track_idx].cls
    candidates = np.asarray([detections[i].tlwh for i in detection_indices])
    classes = np.asarray([detections[i].cls for i in detection_indices])
    cost_matrix[row, :] = 1. - iou(bbox, wanted_class, candidates, classes)

  return cost_matrix

'''
Track Class
'''

class TrackState:

  # State of a track
  Tentative = 1
  Confirmed = 2
  Deleted = 3

class Track:
  #Relating to the Pos and Vel of the Track

  def __init__(self, mean, covariance, track_id, n_init, max_age, feature = None, cls = None):
    #Mean is in xyah In case there is an error with the iou section that comes from here
    self.mean = mean
    self.track_id = track_id
    self.covariance = covariance
    self.hits = 1
    self.age = 1
    self.time_since_update = 0
    self.cls = cls

    self.state = TrackState.Tentative
    self.features = []

    if feature is not None:
      self.features.append(feature)

    self._n_init = n_init
    self._max_age = max_age

  def to_tlwh_Track(self):
    #Mean is in xyah
    
    ret = self.mean[:4].copy()
    ret[2] *= ret[3]
    ret[:2] -= ret[2:] / 2
    return ret

  def to_tlbr_Track(self):

    ret = self.to_tlwh_Track()
    ret[2:] = ret[:2] + ret[2:]
    return ret

  def predict(self, kf):
    #kf = kalman filter

    self.mean, self.covariance = kf.predict(self.mean, self.covariance)
    self.age +=1
    self.time_since_update += 1

  def update(self, kf, detection):
    self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())
    self.features.append(detection.feature)
    self.hits += 1
    self.time_since_update = 0
    if self.state == TrackState.Tentative and self.hits >= self._n_init:
      self.state = TrackState.Confirmed
  def mark_missed(self):
    if self.state == TrackState.Tentative:
        self.state = TrackState.Deleted
    elif self.time_since_update > self._max_age:
        self.state = TrackState.Deleted
  def is_tentative(self):
    return self.state == TrackState.Tentative

  def is_confirmed(self):
    return self.state == TrackState.Confirmed

  def is_deleted(self):
    return self.state == TrackState.Deleted

'''
NN Matching: Track the distance between two different tracks
'''
import numpy as np

def _pdist(a, b):
  a, b = np.asarray(a), np.asarray(b)

  if len(a) == 0 or len(b) == 0:
    return np.zeros(len(a), len(b))

  a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
  r2 = -2 * np.dot(a, b.T) + a2[:, None] + b2[None, :]
  r2 = np.clip(r2, 0., float(np.inf))
  return r2

def _cosine_distance(a, b, data_is_normalized=False):
  if not data_is_normalized:
    a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
    b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
  a = np.asarray(a).reshape(-1, np.shape(a)[-1])
  b = np.asarray(b).reshape(-1, b.shape[-1])
  a = np.astype(a, np.float16)
  a = np.astype(b, np.float16)
  return np.astype((1. - np.dot(a, b.T)), np.float16)

def _nn_euclidean_distance(x, y):
  distances = _pdist(x, y)
  return np.maximum(0.0, distances.min(axis=0))

def _nn_cosine_distance(x, y):
  distances = _cosine_distance(x, y)
  return distances.min(axis=0)

class NearestNeighborDistanceMetric(object):

  def __init__(self, metric, matching_threshold, budget=None):
    if metric == "euclidean":
      self._metric = _nn_euclidean_distance
    elif metric == "cosine":
      self.metric = _nn_cosine_distance
    else:
      raise ValueError("Invalid metric; must be either 'euclidean' or 'cosine'")
    self.matching_threshold = matching_threshold
    self.budget = budget
    self.samples = {}

  def partial_fit(self, features, targets, active_targets):
    for feature, target in zip(features, targets):
      self.samples.setdefault(target, []).append(feature)
      if self.budget is not None:
        self.samples[target] = self.samples[target][-self.budget:]
    print(f'Active Targets: {len(active_targets)}')
    print(f'Samples: {len(self.samples)}')

    self.samples = {k: self.samples[k] for k in active_targets}

  def distance(self, features, targets):
    temp_features = features
    features = features.reshape(1, -1)
    cost_matrix = np.zeros((len(targets), len(temp_features)))
    for i, target in enumerate(targets):
      #cost_matrix = cost_matrix.reshape(1, -1)
      print(f'Cost Matrix Shape:{cost_matrix.shape}')
      print(f'Cost Matrix Splice Shape: {cost_matrix[i, :].shape}')
      print(f'Samples Shape: {self.metric(self.samples[target], features).reshape(1, -1).shape}')
      recalculated_cost_matrix = self.metric(self.samples[target], features).reshape(1, -1)
      cost_matrix[i, :] = recalculated_cost_matrix
      print(f'Cost Matrix Recalculated Shape:{cost_matrix.shape}')
      print(f'Cost Matrix Recalculated Splice Shape: {cost_matrix[i, :].shape}')
    return cost_matrix



'''
Multi-Target Tracker Class
'''

import numpy as np

class Tracker:

  def __init__(self, metric, max_iou_distance=0.9, max_age = 30, n_init=2):
    self.metric = metric
    self.max_iou_distance = max_iou_distance
    self.max_age = max_age
    self.n_init = n_init

    self.kf = KalmanFilter()
    self.tracks = []
    self._next_id = 1

  def predict(self):
    for track in self.tracks:
      track.predict(self.kf)

  def update(self, detections):
    matches, unmatched_tracks, unmatched_detections = self._match(detections)
    # Check if tracks are being unnecessarily deleted
    self.tracks = [t for t in self.tracks if not t.is_deleted()]
    print("Matches found:", matches)
    print("Track states before update:", [(i, t.state) for i, t in enumerate(self.tracks)])
    
    for track_idx, detection_idx in matches:
        print(f"Updating track {track_idx} with detection {detection_idx}")
        print(f"Track hits before update: {self.tracks[track_idx].hits}")
        self.tracks[track_idx].update(self.kf, detections[detection_idx])
        print(f'All Tracks States: {[k.state for k in self.tracks]}')
        print(f"Track hits after update: {self.tracks[track_idx].hits}")
        print(f"Track state after update: {self.tracks[track_idx].state}")
    print(f'Num Unmatched Tracks: {len(unmatched_tracks)}')
    for track_idx in unmatched_tracks:
      self.tracks[track_idx].mark_missed()
    for detection_idx in unmatched_detections:
      self._initiate_track(detections[detection_idx])
    self.tracks = [t for t in self.tracks if not t.is_deleted()]

    active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
    features, targets = [], []
    print(f"Num Tracks: {len([k for k in self.tracks if k.is_confirmed()])}")
    confirmed_tracks = [k for k in self.tracks if k.is_confirmed()]
    for track in confirmed_tracks:
      features += track.features
      targets += [track.track_id for _ in track.features]
      track.features = []
    self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

  def _match(self, detections):

    def gated_metric(tracks, dets, track_indices, detection_indices):

      print(f"Tracks: {len(tracks)} | Tracks Indicies: {len(track_indices)}")
      print(f"Dets: {len(dets)} | Detection Indicies: {len(detection_indices)}")
      features = np.array([dets[i].feature for i in detection_indices])
      targets = np.array([tracks[i].track_id for i in track_indices])
      cost_matrix = self.metric.distance(features, targets)
      print(f"Cost Matrix Init Shape: {cost_matrix.shape}")
      cost_matrix = linear_assignment(gate_cost_matrix(self.kf, cost_matrix, tracks, dets, track_indices, detection_indices))

      return cost_matrix

    confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
    unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

    matches_a, unmatched_tracks_a, unmatched_detections = matching_cascade(gated_metric, self.metric.matching_threshold, self.max_age, self.tracks, detections, confirmed_tracks)

    iou_track_candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
    unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]
    matches_b, unmatched_tracks_b, unmatched_detections = min_cost_matching(iou_cost, self.max_iou_distance, self.tracks, detections, iou_track_candidates, unmatched_detections)

    matches = matches_a + matches_b
    unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
    return matches, unmatched_tracks, unmatched_detections

  def _initiate_track(self, detection):
    mean, covariance = self.kf.initiate(detection.to_xyah())
    cls = detection.cls
    self.tracks.append(Track(mean, covariance, self._next_id, self.n_init, self.max_age, detection.feature, cls))
    self._next_id += 1





'''
Testing Section
'''
#Load Model
from ultralytics import YOLO

model = YOLO('C:/Users/akans/OneDrive/Desktop/VSCode/YOLO11Best.pt')


#Important function to get the bounding boxes into the right format
import cv2
import numpy as np

def to_correct_result_format(result):
  result = result[0]
  boxes = result.boxes

  return_matrix = []
  coords = boxes.xyxy.tolist()
  classes = boxes.cls.tolist()
  confidences = boxes.conf.tolist()
  for idx in range(len(coords)):
    object_list = []
    object_list.append(coords[idx][0])
    object_list.append(coords[idx][1])
    object_list.append(coords[idx][2])
    object_list.append(coords[idx][3])
    object_list.append(confidences[idx])
    object_list.append(classes[idx])
    return_matrix.append(object_list)

  return return_matrix


#Testing loop that takes in a video

metric = NearestNeighborDistanceMetric("cosine", matching_threshold=0.1, budget=100)
tracker = Tracker(metric)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns
from matplotlib.colors import to_rgba_array 

# Define hook function
def hook_fn(module, input, output):
    intermediate_features.append(output)

# Define feature extraction function
def extract_features(model, img, layer_index=20): ##Choose the layer that fit your application
    global intermediate_features
    intermediate_features = []
    print(f"Input shape before forward pass: {img.shape}")
    print(f"Input type before forward pass: {type(img)}")
    print(f"Input min/max values: {img.min()}, {img.max()}")
    
    intermediate_features.clear()
    
    # Get target layer
    layer = list(model.model.model.children())[layer_index]
    hook_handle = layer.register_forward_hook(hook_fn)
    
    # Convert to torch tensor if needed
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).float()
    
    # Forward pass with debugging
    with torch.no_grad():
        _ = model(img, verbose=False)
    
    hook_handle.remove()
    return intermediate_features[0]

# Make sure to preprocess the image since the input image must be 640x640x3
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
    #    transforms.Grayscale(num_output_channels=3),  # Convert to RGB
        transforms.ToTensor(),
        transforms.Normalize(mean=0., std=1.)
    ])
    img = Image.fromarray(image)
    img = transform(img)
    img = img.unsqueeze(0)
    
    return np.array(img)
def preprocess_my_image(frame):
    # Convert frame to expected YOLO input format
    img = cv2.resize(frame, (640, 640))  # Standard YOLO input size
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
    img = np.transpose(img, (2, 0, 1))    # Convert to channel-first format
    img = np.expand_dims(img, axis=0)      # Add batch dimension
    return img




def process_video(video_path, output_path=None):
    # Load video
    cap = cv2.VideoCapture(video_path)

    # Prepare output if needed
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    frame_count = 0
    while cap.isOpened():
        print("\n--- New Frame ---")
        print(f"Frame number: {frame_count}")
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: YOLOv8 object detection
        results = model.predict(frame)
        results = to_correct_result_format(results)
        # Extract the bounding boxes and confidences
        detections = []
        for result in results:  # Bounding box format: [x1, y1, x2, y2, confidence, class]
          x1, y1, x2, y2, conf, cls = result[0], result[1], result[2], result[3], result[4], result[5]
          print(result)
          width, height = x2 - x1, y2 - y1
          bbox = [x1, y1, width, height]  # Convert to x, y, w, h

          #Pass Frame through the feature extractor and then put the new 'frame' into the det variable
          #Maybe also create a subset image where it is just the area in the bbox coords before passing though feature extractor
          cropped_bbox_img = frame[int(y1):int(y2), int(x1):int(x2)]
          cropped_bbox_img = preprocess_image(cropped_bbox_img)
          feature = extract_features(model, preprocess_image(frame), layer_index=20)
          det = Detection(bbox, conf, feature, cls)
          detections.append(det)  # Use detection class

        # Step 2: Update DeepSORT tracker
        print("Detections:", detections)
        tracker.predict()
        #Pass the feature vectors instead of detections
        tracker.update(detections)

        print("Current tracks status:")
        for t in tracker.tracks:
            print(f"Track ID: {t.track_id}, Hits: {t.hits}, State: {t.state}")

        # Step 3: Draw tracking results on the frame
        for track in tracker.tracks:
          
          if not track.is_confirmed() or track.time_since_update > 1:
              print(track.state)
              print("Not confirmed")
              continue
          print("Running drawing")
          bbox = track.to_tlwh_Track()  # Get bounding box in format (top-left x, y, width, height)
          x1, y1, w, h = [int(i) for i in bbox]

          # Draw the bounding box and track ID
          print(f"Drawing bounding box: {x1}, {y1}, {w}, {h}")  # Add this line for debugging
          cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
  
          print(f"Drawing track ID: {track.track_id} Class: {track.cls}")  # Add this line for debugging
          cv2.putText(frame, f"ID: {track.track_id} Class: {track.cls}",  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        # Save the output frame if needed
        if output_path is not None:
            out.write(frame)

        # Display the frame with tracking
        cv2.imshow('Frame', frame)
      
        frame_count += 1

    cap.release()
    if output_path is not None:
        out.release()
    cv2.destroyAllWindows()

# Run the function on video

input_video = 'C:/Users/akans/OneDrive/Desktop/VSCode/Two_Sharks_Video.mp4'
output_vid_path = 'C:/Users/akans/OneDrive/Desktop/VSCode/Two_Sharks_Video_Post_DeepSORT(1).mp4'
process_video(input_video, output_vid_path)
'''
input_video2 = 'C:/Users/akans/OneDrive/Desktop/VSCode/Turtle_Video_From_Field.mp4'
output_vid_path2 = 'C:/Users/akans/OneDrive/Desktop/VSCode/Turtle_Video_Field_Post_DeepSORT(1).mp4'
process_video(input_video2, output_vid_path2)
'''