import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.screen_tracker import (
    create_output_directory,
    make_teacher_schedule,
    select_window_indices,
)
from original.my_track import config


class TrackerScreeningTests(unittest.TestCase):
    def test_teacher_schedule_override_uses_explicit_values(self):
        schedule = make_teacher_schedule(config.teacher_assignment_ratio, 0, 2)
        self.assertEqual([schedule(epoch) for epoch in (1, 2, 3)], [0.5, 0.0, 0.0])

    def test_selection_covers_each_sequence_and_its_time_range(self):
        windows = [(0, start) for start in (0, 25, 50, 75, 100)] + [(1, 0), (1, 25)]
        self.assertEqual(select_window_indices(windows, 3), [0, 2, 4, 5, 6])
        self.assertEqual(select_window_indices(windows, 3), select_window_indices(windows, 3))

    def test_invalid_selection_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            select_window_indices([(0, 0)], 0)

    def test_output_directory_cannot_overwrite_previous_experiment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "screen"
            create_output_directory(output)
            sentinel = output / "previous.txt"
            sentinel.write_text("keep", encoding="ascii")
            with self.assertRaises(FileExistsError):
                create_output_directory(output)
            self.assertEqual(sentinel.read_text(encoding="ascii"), "keep")

    def test_cli_rejects_invalid_limits_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_args = (
                ("--epochs", "0"),
                ("--val-frames", "1"),
                ("--lr", "nan"),
                ("--teacher-warmup-epochs", "0"),
                ("--teacher-warmup-epochs", "-1", "--teacher-anneal-epochs", "2"),
            )
            for extra in invalid_args:
                output = Path(temp_dir) / "invalid"
                result = subprocess.run(
                    [sys.executable, "scripts/screen_tracker.py", "--output", str(output), *extra],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 2, msg=result.stderr)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
