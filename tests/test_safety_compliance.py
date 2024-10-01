import unittest
import numpy as np
from src.safety_compliance import ensure_safety

class TestSafetyCompliance(unittest.TestCase):
    def test_ensure_safety(self):
        frames = [np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8) for _ in range(5)]
        reports = ensure_safety(frames)
        self.assertIsInstance(reports, list)
        self.assertEqual(len(reports), 5)
        for report in reports:
            self.assertIn('compliance', report)
            self.assertIn('details', report)

if __name__ == '__main__':
    unittest.main()
