import numpy as np

class ObjectDetectionModel:
    def __init__(self): #maybe use Yolo8 instead
        self.weights = np.random.randn(3,3)
    
    def detect_objects(self, frame):
        detections = []
        regions = self.sliding_window(frame, window_size=(50,50), step_size=25)
        for region, (x, y) in regions:
            if self.classify(region):
                detections.append({'class': 'helmet', 'bbox': (x, y, 50,50)})
        return detections
    
    def sliding_window(self, frame, window_size, step_size):
        h, w, _ = frame.shape
        window_h, window_w = window_size
        regions = []
        for y in range(0, h - window_h +1, step_size):
            for x in range(0, w - window_w +1, step_size):
                window = frame[y:y+window_h, x:x+window_w]
                regions.append((window, (x, y)))
        return regions
    
    def classify(self, region):
        gray = self.rgb_to_grayscale(region)
        feature = np.sum(gray * self.weights)
        return feature > 1000
    
    def rgb_to_grayscale(self, frame):
        return 0.2989 * frame[:,:,0] + 0.5870 * frame[:,:,1] + 0.1140 * frame[:,:,2]
