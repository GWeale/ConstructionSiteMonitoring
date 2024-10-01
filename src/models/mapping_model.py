import numpy as np

class MappingModel:
    def __init__(self):
        self.map = {}
    
    def add_features(self, features):
        for feature in features:
            key = tuple(feature)
            if key not in self.map:
                self.map[key] = 1
            else:
                self.map[key] += 1
    
    def generate_map(self):
        return np.array(list(self.map.keys()))
