import unittest
import numpy as np
from src.config import config
from src.dataset import preprocess_image
from src.predict import ImagePredictor

class TestInferencePipeline(unittest.TestCase):
    def test_preprocess_image(self):
        dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
        processed = preprocess_image(dummy_img)
        
        self.assertIn("batch_tf", processed)
        self.assertIn("batch_pt", processed)
        self.assertEqual(processed["batch_tf"].shape, (1, 256, 256, 3))
        self.assertEqual(processed["batch_pt"].shape, (1, 3, 256, 256))

    def test_predictor_prediction(self):
        predictor = ImagePredictor()
        dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
        res = predictor.predict(dummy_img)
        
        self.assertIn("label", res)
        self.assertIn("class_id", res)
        self.assertIn(res["class_id"], [0, 1])
        self.assertTrue(0.0 <= res["confidence"] <= 100.0)

if __name__ == "__main__":
    unittest.main()
