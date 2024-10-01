import numpy as np
from src.utils import feature_extraction, feature_matching

def create_map(frames):
    features = []
    for frame in frames:
        feat = feature_extraction(frame)
        features.append(feat)
    matches = feature_matching(features)
    map_points = triangulate(matches)
    return map_points

def triangulate(matches):
    map_points = []
    for match in matches:
        point = np.mean(match, axis=0)
        map_points.append(point)
    return np.array(map_points)
