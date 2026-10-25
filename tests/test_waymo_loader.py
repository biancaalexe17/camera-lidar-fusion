import tempfile
import unittest
from pathlib import Path

from src.data.waymo_loader import component_parquet_path, create_cache_links


class WaymoLoaderTests(unittest.TestCase):
    def test_component_paths_follow_notebook_naming(self):
        root = Path('/project')
        self.assertEqual(
            component_parquet_path('lidar', 'segment_a', data_root=root),
            root / 'training' / 'lidar' / 'training_lidar_segment_a.parquet',
        )
        self.assertEqual(
            component_parquet_path('camera_image', 'segment_a', data_root=root),
            root / 'training' / 'camera_image' / 'segment_a.parquet',
        )

    def test_cache_links_are_created_without_copying_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'drive'
            cache = Path(temp) / 'cache'
            source_dir = root / 'training' / 'lidar'
            source_dir.mkdir(parents=True)
            source = source_dir / 'training_lidar_segment_a.parquet'
            source.write_bytes(b'test')

            counts = create_cache_links(
                ['lidar'], data_root=root, cache_root=cache
            )
            link = cache / 'training' / 'lidar' / 'segment_a.parquet'
            self.assertEqual(counts['lidar'], 1)
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), source.resolve())
            self.assertEqual(source.read_bytes(), b'test')

    def test_unknown_component_is_rejected(self):
        with self.assertRaises(ValueError):
            component_parquet_path('radar', 'segment_a')


if __name__ == '__main__':
    unittest.main()
