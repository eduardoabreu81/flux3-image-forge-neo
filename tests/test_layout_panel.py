# FLUX 3 Image Klein support for Forge Neo
# The FLUX 3 Layout panel's JSON logic (javascript/flux3_layout.js), through Node; skipped where Node is missing.

import shutil
import subprocess
import unittest
from pathlib import Path

NODE = shutil.which("node")
TEST = Path(__file__).parent / "js" / "layout_test.js"


@unittest.skipIf(NODE is None, "node not installed")
class TestLayoutPanel(unittest.TestCase):
    def test_layout_json(self):
        result = subprocess.run([NODE, str(TEST)], capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
