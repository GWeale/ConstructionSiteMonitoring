import unittest
import numpy as np
from src.video_processing import read_video_frames, equirectangular_to_cartesian

class TestVideoProcessing(unittest.TestCase):
    def test_read_video_frames(self):
        frames = read_video_frames('data/raw/footage360.mp4')
        self.assertIsInstance(frames, list)
        self.assertGreater(len(frames), 0)
    
    def test_equirectangular_to_cartesian(self):
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        cartesian, processed = equirectangular_to_cartesian(frame)
        self.assertEqual(cartesian.shape, (1080, 1920, 3))
        self.assertEqual(processed.shape, (1080, 1920, 3))

if __name__ == '__main__':
    unittest.main()
