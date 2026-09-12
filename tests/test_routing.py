"""Small, model-free checks for the public routing replay."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "replay_routing.py"
spec = importlib.util.spec_from_file_location("routing", SOURCE)
routing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(routing)


class RoutingTests(unittest.TestCase):
    def test_strict_threshold_and_correction(self):
        labels = np.array([0, 1])
        cnn = np.array([1, 1])
        qwen = np.array([0, 0])
        scores = np.array([0.4, 0.9])
        boundary = routing.replay(labels, cnn, qwen, scores, 0.4)
        self.assertEqual(boundary["calls"], 0)
        selected = routing.replay(labels, cnn, qwen, scores, np.nextafter(0.4, 1.0))
        self.assertEqual(selected["calls"], 1)
        self.assertEqual(selected["correct"], 2)
        self.assertEqual(selected["cnn_to_qwen_corrected"], 1)
        self.assertEqual(selected["cnn_to_qwen_harmed"], 0)

    def test_invalid_prediction_counts_as_error(self):
        result = routing.metrics(np.array([0, 1]), np.array([-1, 1]))
        self.assertEqual(result["invalid"], 1)
        self.assertEqual(result["accuracy"], 0.5)

    def test_mismatched_ids_are_rejected(self):
        qwen = [{"source_id": "one", "label": 0, "gt": "class0", "parsed": "class0"}]
        cnn = [{"source_id": "two", "label": 0, "prediction": 0, "valid": True,
                "confidence": 1.0, "probabilities": [1.0] + [0.0] * 7}]
        with self.assertRaises(ValueError):
            routing.aligned_pair(qwen, cnn)


if __name__ == "__main__":
    unittest.main()
