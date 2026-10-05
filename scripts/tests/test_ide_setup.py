#!/usr/bin/env python3
"""
Class A cover for scripts/ide-setup.sh, which records the TornadoVM SDK's jar version for the
ide Maven profile. No build, no real SDK: the script runs from a copy in a temporary tree, next
to a fake SDK whose jars are empty files.

Run with: python3 -m unittest discover -s scripts/tests
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class IdeSetupTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        (self.root / "scripts").mkdir()
        shutil.copy(REPO_ROOT / "scripts" / "ide-setup.sh", self.root / "scripts")
        self.config = self.root / ".mvn" / "maven.config"

    def sdk(self, *jars):
        sdk = self.root / "sdk"
        modules = sdk / "share" / "java" / "tornado"
        modules.mkdir(parents=True)
        for jar in jars:
            (modules / jar).touch()
        return sdk

    def run_setup(self, sdk):
        env = dict(os.environ)
        env.pop("TORNADOVM_HOME", None)
        if sdk is not None:
            env["TORNADOVM_HOME"] = str(sdk)
        return subprocess.run(["bash", str(self.root / "scripts" / "ide-setup.sh")],
                              env=env, capture_output=True, text=True)

    def test_records_the_version_of_a_released_sdk(self):
        r = self.run_setup(self.sdk("tornado-api-7.0.1.jar", "tornado-runtime-7.0.1.jar"))
        self.assertEqual(0, r.returncode, r.stderr)
        self.assertEqual("-Dtornadovm.sdk.version=7.0.1\n", self.config.read_text())

    def test_records_the_version_of_a_develop_sdk(self):
        r = self.run_setup(self.sdk("tornado-api-7.1.1-jdk21-dev.jar"))
        self.assertEqual(0, r.returncode, r.stderr)
        self.assertEqual("-Dtornadovm.sdk.version=7.1.1-jdk21-dev\n", self.config.read_text())

    def test_replaces_its_own_line_and_keeps_the_others(self):
        self.config.parent.mkdir()
        self.config.write_text("-T1C\n-Dtornadovm.sdk.version=7.0.1\n-Dfoo=bar\n")
        r = self.run_setup(self.sdk("tornado-api-7.1.0.jar"))
        self.assertEqual(0, r.returncode, r.stderr)
        self.assertEqual("-T1C\n-Dfoo=bar\n-Dtornadovm.sdk.version=7.1.0\n", self.config.read_text())

    def test_refuses_without_tornadovm_home(self):
        r = self.run_setup(None)
        self.assertNotEqual(0, r.returncode)
        self.assertIn("TORNADOVM_HOME is not set", r.stderr)
        self.assertFalse(self.config.exists())

    def test_refuses_a_directory_that_is_not_an_sdk(self):
        r = self.run_setup(self.sdk())
        self.assertNotEqual(0, r.returncode)
        self.assertIn("found 0", r.stderr)
        self.assertFalse(self.config.exists())


if __name__ == "__main__":
    unittest.main()
