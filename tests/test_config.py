import unittest
from pathlib import Path

from src.config import PATHS, VALIDATION_CONTEXTS


class ConfigTests(unittest.TestCase):
    def test_paths_are_path_objects(self):
        self.assertIsInstance(PATHS.drive_root, Path)
        self.assertIsInstance(PATHS.training_data, Path)
        self.assertIsInstance(PATHS.custom_model_checkpoint, Path)

    def test_custom_model_paths_match_reference_notebook(self):
        self.assertEqual(
            PATHS.custom_model_config,
            PATHS.drive_root / "project" / "configs" / "pointpillars_30seg.yaml",
        )
        self.assertEqual(
            PATHS.custom_model_checkpoint,
            PATHS.drive_root / "project" / "30seg" / "checkpoints" / "latest.pt",
        )

    def test_five_validation_contexts(self):
        self.assertEqual(len(VALIDATION_CONTEXTS), 5)
        self.assertTrue(all(isinstance(context, str) for context, _ in VALIDATION_CONTEXTS))
        self.assertTrue(all(frames > 0 for _, frames in VALIDATION_CONTEXTS))


if __name__ == "__main__":
    unittest.main()
