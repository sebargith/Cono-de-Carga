from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_pipeline_from_videos import run_yolo_masks  # noqa: E402


class LocalInferenceTests(unittest.TestCase):
    def test_yolo_masks_uses_current_python_without_docker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            perception_dir = root / "perception"
            prediction_script = (
                perception_dir / "scripts" / "predict_to_analytic_masks.py"
            )
            prediction_script.parent.mkdir(parents=True)
            prediction_script.write_text("", encoding="utf-8")
            model = perception_dir / "models" / "model.pt"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"model")
            frames_dir = root / "frames"
            masks_dir = root / "masks"
            frames_dir.mkdir()
            masks_dir.mkdir()

            with patch("run_pipeline_from_videos.run_command") as run_command:
                run_yolo_masks(
                    perception_dir=perception_dir,
                    yolo_model=model,
                    frames_dir=frames_dir,
                    masks_dir=masks_dir,
                    conf=0.35,
                    imgsz=640,
                    device="0",
                    mask_mode="largest",
                    class_id=1,
                )

            command, cwd, label = run_command.call_args.args
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(command[1], str(prediction_script))
            self.assertNotIn("docker", [part.lower() for part in command])
            self.assertIn(str(model), command)
            self.assertIn(str(frames_dir), command)
            self.assertIn(str(masks_dir), command)
            self.assertEqual(cwd, ROOT)
            self.assertEqual(label, "inferencia YOLO local a mascaras analiticas")


if __name__ == "__main__":
    unittest.main()
