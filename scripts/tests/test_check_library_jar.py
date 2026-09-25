#!/usr/bin/env python3
"""
Class A cover for scripts/check_library_jar.py, the guard that keeps TornadoVM and its
dependencies out of the published jar (issue #176). No build, no SDK: the jars are
written here with zipfile.

Run with: python3 -m unittest discover -s scripts/tests
"""

import importlib.util
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_library_jar", REPO_ROOT / "scripts" / "check_library_jar.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

LIBRARY_ENTRIES = [
    "META-INF/",
    "META-INF/MANIFEST.MF",
    "META-INF/jpms.args",
    "META-INF/services/org.beehive.jitllm.model.provider.ModelProvider",
    "META-INF/maven/io.github.beehive-lab/jitllm/pom.xml",
    "META-INF/maven/io.github.beehive-lab/jitllm/pom.properties",
    "org/beehive/jitllm/",
    "org/beehive/jitllm/JitllmApp.class",
]


class CheckLibraryJarTest(unittest.TestCase):
    def jar(self, entries):
        fd, path = tempfile.mkstemp(suffix=".jar")
        os.close(fd)
        self.addCleanup(os.remove, path)
        with zipfile.ZipFile(path, "w") as z:
            for e in entries:
                z.writestr(e, b"")
        return path

    def test_the_library_layout_is_clean(self):
        self.assertEqual([], checker.check(self.jar(LIBRARY_ENTRIES)))

    def test_bundled_tornadovm_and_graal_classes_are_named(self):
        problems = checker.check(self.jar(LIBRARY_ENTRIES + [
            "uk/ac/manchester/tornado/api/TaskGraph.class",
            "org/graalvm/collections/UnmodifiableEconomicMap.class",
        ]))
        joined = "\n".join(problems)
        self.assertIn("uk/ac/manchester/tornado/", joined)
        self.assertIn("org/graalvm/", joined)

    def test_any_other_bundled_library_fails(self):
        problems = checker.check(self.jar(LIBRARY_ENTRIES + [
            "org/apache/logging/log4j/Logger.class",
            "META-INF/services/org.apache.logging.log4j.spi.Provider",
        ]))
        self.assertEqual(1, len(problems))
        self.assertIn("org/apache/logging/log4j/Logger.class", problems[0])

    def test_a_jar_without_project_classes_fails(self):
        self.assertTrue(checker.check(self.jar(["META-INF/MANIFEST.MF"])))

    def test_main_reports_through_the_exit_status(self):
        clean, dirty = self.jar(LIBRARY_ENTRIES), self.jar(LIBRARY_ENTRIES + ["uk/ac/manchester/tornado/X.class"])
        stderr = sys.stderr
        try:
            sys.stderr = open(os.devnull, "w")
            self.assertEqual(0, checker.main(["check", clean]))
            self.assertEqual(1, checker.main(["check", clean, dirty]))
            self.assertEqual(2, checker.main(["check"]))
        finally:
            sys.stderr.close()
            sys.stderr = stderr


if __name__ == "__main__":
    unittest.main()
