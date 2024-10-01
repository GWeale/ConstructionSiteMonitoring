import unittest
import numpy as np
from src.mapping import create_map

class TestMapping(unittest.TestCase):
    def test_create_map(self):
        frames = [np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8) for _ in range(5)]
        map_points = create_map(frames)
        self.assertIsInstance(map_points, np.ndarray)
        self.assertGreater(map_points.shape[0], 0)

if __name__ == '__main__':
    unittest.main()
