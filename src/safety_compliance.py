import numpy as np
from src.models.object_detection_model import ObjectDetectionModel

def ensure_safety(frames):
    model = ObjectDetectionModel()
    safety_reports = []
    for frame in frames:
        detections = model.detect_objects(frame)
        report = analyze_safety(detections)
        safety_reports.append(report)
    return safety_reports

def analyze_safety(detections):
    required_equipment = ['helmet', 'vest', 'gloves'] # from c script previously
    present = {item: False for item in required_equipment}
    for det in detections:
        if det['class'] in required_equipment:
            present[det['class']] = True
    compliance = all(present.values())
    return {'compliance': compliance, 'details': present}
