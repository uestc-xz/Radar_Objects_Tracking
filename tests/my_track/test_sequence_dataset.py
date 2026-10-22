import tempfile
import unittest
from pathlib import Path

import numpy as np

from original.my_dataset.track_sequence_dataset import (
    TrackSequenceColdDataset,
    TrackSequenceDataset,
)


class SequenceDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "data"
        self.cache_root = Path(self.temp_dir.name) / "cache"
        for folder, count in (("record_a", 5), ("record_b", 2)):
            signals = self.root / folder / "signals"
            labels = self.root / folder / "labels"
            signals.mkdir(parents=True)
            labels.mkdir(parents=True)
            for index in range(count):
                np.save(signals / f"{index}.npy", np.zeros((8, 89, 10), dtype=np.complex64))
                label = np.zeros((10, 6), dtype=np.float32)
                label[:, 0:2] = (1.0, 2.0)
                np.save(labels / f"{index}.npy", label)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_sequence_mode_returns_path_index_without_loading_all_frames(self):
        dataset = TrackSequenceDataset(
            str(self.root), window_size=3, stride=2, add_noise=False, full_sequence=True
        )

        self.assertEqual(len(dataset), 2)
        first = dataset[0]
        self.assertEqual(first["folder"], "record_a")
        self.assertEqual(len(first["frames"]), 5)
        self.assertTrue(Path(first["frames"][0][0]).is_file())
        self.assertEqual(len(dataset[1]["frames"]), 2)

    def test_default_mode_keeps_existing_sliding_window_count(self):
        dataset = TrackSequenceDataset(
            str(self.root), window_size=3, stride=2, add_noise=False
        )
        self.assertEqual(len(dataset), 2)
        signals, labels = dataset[0]
        self.assertEqual(tuple(signals.shape), (3, 8, 89, 10))
        self.assertEqual(tuple(labels.shape), (3, 3, 2))

    def test_cold_full_sequence_mode_returns_cache_paths_lazily(self):
        dataset = TrackSequenceColdDataset(
            str(self.root), str(self.cache_root), window_size=3, stride=2, full_sequence=True
        )
        record = dataset[0]
        self.assertEqual(len(record["frames"]), 5)
        cache_path, label_path, sample_id = record["frames"][0]
        self.assertTrue(cache_path.endswith("record_a/m1_cache/0.npz") or
                        cache_path.endswith("record_a\\m1_cache\\0.npz"))
        self.assertTrue(Path(label_path).is_file())
        self.assertEqual(sample_id, 0)
        self.assertFalse(Path(cache_path).exists())


if __name__ == "__main__":
    unittest.main()
