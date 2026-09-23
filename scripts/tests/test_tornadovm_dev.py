#!/usr/bin/env python3
"""
Class A cover for the TornadoVM development-dependency tooling: the shared setup script's
identity and version derivation, the release-version setter's transitions, the POM's
development/release coordinate selection, and the workflows' single-revision propagation.
No accelerator, no TornadoVM build, no network: the script pieces under test are exercised on
fake SDK layouts, and network-facing commands are not called.

Run with: python3 -m unittest discover -s scripts/tests
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV = REPO_ROOT / "scripts" / "tornadovm-dev.sh"
SETTER = REPO_ROOT / "scripts" / "set-tornadovm-release.sh"
SHA = "65f06c5d162c36022f9d5724dd38b9784d85cfcc"


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


class RecipeIdentity(unittest.TestCase):
    def test_recipe_is_stable_and_content_based(self):
        a = run([str(DEV), "recipe"]).stdout.strip()
        b = run([str(DEV), "recipe"]).stdout.strip()
        self.assertRegex(a, r"^[0-9a-f]{12}$")
        self.assertEqual(a, b)
        # A different script text is a different recipe: copy it with one extra comment line.
        with tempfile.TemporaryDirectory() as d:
            copy = Path(d) / "tornadovm-dev.sh"
            copy.write_text(DEV.read_text() + "\n# recipe change\n")
            copy.chmod(0o755)
            c = run([str(copy), "recipe"]).stdout.strip()
        self.assertNotEqual(a, c)


class SetupArgumentValidation(unittest.TestCase):
    def test_rejects_a_short_revision(self):
        r = run([str(DEV), "setup", "--ref", "65f06c5d1", "--backend", "cuda", "--root", tempfile.mkdtemp()])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("40-hex", r.stderr)

    def test_rejects_an_unsupported_jdk(self):
        r = run([str(DEV), "setup", "--ref", SHA, "--jdk", "17", "--backend", "cuda", "--root", tempfile.mkdtemp()])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--jdk must be 21 or 25", r.stderr)


class ArtifactVersionDerivation(unittest.TestCase):
    """The version comes from what the build produced: the dist name and the jar must agree."""

    def _fake_sdk(self, root, version, backend, jar_version=None):
        sdk = root / "TornadoVM" / "dist" / f"tornadovm-{version}-{backend}-linux-amd64" / f"tornadovm-{version}-{backend}"
        (sdk / "share" / "java" / "tornado").mkdir(parents=True)
        (sdk / "share" / "java" / "tornado" / f"tornado-api-{jar_version or version}.jar").write_bytes(b"jar")
        return sdk

    def test_dist_name_and_jar_agree(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._fake_sdk(root, "6.1.1-jdk21-dev", "cuda")
            r = run(["bash", "-c", f"""
                set -e
                sdk_dir_of() {{ find "$1/TornadoVM/dist" -maxdepth 3 -type d -name "tornadovm-*-$BACKEND" | head -n 1; }}
                {self._function_source('artifact_version_of')}
                die() {{ echo "$@" >&2; exit 1; }}
                BACKEND=cuda
                artifact_version_of "$(sdk_dir_of {root})"
            """])
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.strip(), "6.1.1-jdk21-dev")

    def test_a_jar_from_another_version_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._fake_sdk(root, "6.1.1-jdk21-dev", "cuda", jar_version="6.1.0-jdk21")
            r = run(["bash", "-c", f"""
                set -e
                sdk_dir_of() {{ find "$1/TornadoVM/dist" -maxdepth 3 -type d -name "tornadovm-*-$BACKEND" | head -n 1; }}
                {self._function_source('artifact_version_of')}
                die() {{ echo "$@" >&2; exit 1; }}
                BACKEND=cuda
                artifact_version_of "$(sdk_dir_of {root})"
            """])
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("but the jar is", r.stderr)

    def test_jdk25_line_expects_the_jdk22plus_suffix(self):
        src = DEV.read_text()
        self.assertIn('if [ "$JDK" = 21 ]; then echo "-jdk21-dev"; else echo "-jdk22plus-dev"; fi', src)
        # And the build itself takes TornadoVM's jdk22plus make target for JDK 25.
        self.assertIn('if [ "$JDK" = 21 ]; then make BACKEND=$BACKEND; else make jdk22plus BACKEND=$BACKEND; fi', src)
        # No bare mvn install repair path: a damaged repository is rebuilt through make.
        self.assertNotRegex(src, r"mvn -q .*install")

    @staticmethod
    def _function_source(name):
        """The text of one function from the script, so it can be tested in isolation."""
        src = DEV.read_text()
        m = re.search(rf"^{name}\(\) \{{.*?^\}}", src, re.S | re.M)
        assert m, name
        return m.group(0)


class ReleaseVersionSetter(unittest.TestCase):
    def _pom_with(self, release_line):
        return f"""<project><properties>
        <tornadovm.base.version>6.1.1</tornadovm.base.version>
        {release_line}
        <tornadovm.dev.qualifier>-dev</tornadovm.dev.qualifier>
        </properties></project>"""

    def _apply(self, pom_text, version):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            (repo / "scripts").mkdir()
            shutil.copy(SETTER, repo / "scripts" / "set-tornadovm-release.sh")
            (repo / "pom.xml").write_text(pom_text)
            r = run([str(repo / "scripts" / "set-tornadovm-release.sh"), version])
            return r, (repo / "pom.xml").read_text()

    def test_sets_an_empty_release_property(self):
        r, pom = self._apply(self._pom_with("<tornadovm.release.version/>"), "6.2.0")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("<tornadovm.release.version>6.2.0</tornadovm.release.version>", pom)
        # The development line is untouched: base version and -dev qualifier stay.
        self.assertIn("<tornadovm.base.version>6.1.1</tornadovm.base.version>", pom)
        self.assertIn("<tornadovm.dev.qualifier>-dev</tornadovm.dev.qualifier>", pom)

    def test_replaces_a_previous_release_and_works_when_base_matches(self):
        # Development base 6.1.1 and release 6.1.1: the two are distinct properties, so the
        # transition is a change even though the numbers are equal.
        r, pom = self._apply(self._pom_with("<tornadovm.release.version>6.1.0</tornadovm.release.version>"), "6.1.1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("<tornadovm.release.version>6.1.1</tornadovm.release.version>", pom)
        self.assertNotIn("6.1.0", pom)

    def test_rejects_a_malformed_version(self):
        r, pom = self._apply(self._pom_with("<tornadovm.release.version/>"), "6.2")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("<tornadovm.release.version/>", pom)


class PomCoordinateSelection(unittest.TestCase):
    """The POM's own structure: JDK profiles select TornadoVM's suffix, release drops -dev."""

    def setUp(self):
        self.tree = ET.parse(REPO_ROOT / "pom.xml")
        self.ns = {"m": "http://maven.apache.org/POM/4.0.0"}

    def _profile(self, pid):
        for p in self.tree.getroot().findall("m:profiles/m:profile", self.ns):
            if p.find("m:id", self.ns).text == pid:
                return p
        raise AssertionError(pid)

    def _prop(self, parent, name):
        e = parent.find(f"m:properties/m:{name}", self.ns)
        return None if e is None else (e.text or "")

    def test_development_coordinates_per_jdk(self):
        self.assertEqual(self._prop(self._profile("jdk21"), "jdk.version.suffix"), "-jdk21")
        self.assertEqual(self._prop(self._profile("jdk21"), "tornadovm.jdk.suffix"), "-jdk21")
        self.assertEqual(self._prop(self._profile("jdk25"), "jdk.version.suffix"), "-jdk25")
        self.assertEqual(self._prop(self._profile("jdk25"), "tornadovm.jdk.suffix"), "-jdk22plus")
        for pid in ("jdk21", "jdk25"):
            self.assertEqual(
                self._prop(self._profile(pid), "tornadovm.version"),
                "${tornadovm.base.version}${tornadovm.jdk.suffix}${tornadovm.dev.qualifier}",
            )
        root = self.tree.getroot()
        self.assertEqual(self._prop(root, "tornadovm.dev.qualifier"), "-dev")
        # Either unset or a published release (the bump workflow sets it): never a -dev version.
        self.assertRegex(self._prop(root, "tornadovm.release.version"), r"^(\d+\.\d+\.\d+)?$")

    def test_release_profile_substitutes_the_release_and_drops_dev(self):
        rel = self._profile("release")
        self.assertEqual(self._prop(rel, "tornadovm.base.version"), "${tornadovm.release.version}")
        self.assertEqual(self._prop(rel, "tornadovm.dev.qualifier"), "")
        text = ET.tostring(rel, encoding="unicode")
        self.assertIn("enforce-release-tornadovm", text)
        self.assertIn("tornadovm.release.version", text)
        self.assertIn("-jdk(21|22plus)", text)

    def test_no_snapshot_or_range_coordinates(self):
        text = (REPO_ROOT / "pom.xml").read_text()
        self.assertNotIn("SNAPSHOT", text)
        self.assertNotIn("LATEST", text)
        # Version ranges: a dependency version starting with "[" or "(" (the enforcer's JDK range is not a dependency).
        self.assertNotRegex(text, r"<artifactId>tornado-[a-z]+</artifactId>\s*<version>[\[(]")


class WorkflowRevisionPropagation(unittest.TestCase):
    """One resolved TornadoVM commit per workflow run, passed to every use of the action."""

    def _load(self, name):
        return yaml.safe_load((REPO_ROOT / ".github" / "workflows" / name).read_text())

    def test_no_hardcoded_commit_in_the_action_or_workflows(self):
        for path in [REPO_ROOT / ".github" / "actions" / "setup-tornadovm" / "action.yml"] + list(
            (REPO_ROOT / ".github" / "workflows").glob("*.yml")
        ):
            self.assertNotRegex(path.read_text(), r"\b[0-9a-f]{40}\b", f"{path.name} hardcodes a commit")

    def test_action_requires_the_revision(self):
        action = yaml.safe_load((REPO_ROOT / ".github" / "actions" / "setup-tornadovm" / "action.yml").read_text())
        self.assertTrue(action["inputs"]["ref"]["required"])
        self.assertEqual(action["inputs"]["backend"]["required"], True)
        self.assertIn("jdk", action["inputs"])
        self.assertIn("scripts/tornadovm-dev.sh setup --ref", yaml.dump(action))
        # CI consumes the result file of the installation it prepared, never the shared
        # `current` pointer, which another run could move after the lock is released.
        self.assertIn("--result", yaml.dump(action))
        self.assertNotIn("current/provenance.json", yaml.dump(action))
        self.assertIn("install_dir", action["outputs"])

    def test_build_and_run_resolves_once_and_passes_it_everywhere(self):
        wf = self._load("build-and-run.yml")
        jobs = wf["jobs"]
        self.assertIn("resolve-tornadovm", jobs)
        expected = "${{ needs.resolve-tornadovm.outputs.sha }}"
        for name, job in jobs.items():
            if name == "resolve-tornadovm":
                continue
            uses_action = [s for s in job.get("steps", []) if s.get("uses") == "./.github/actions/setup-tornadovm"]
            calls_reusable = job.get("uses", "").endswith("standalone-inference.yml")
            needs = job.get("needs", [])
            needs = [needs] if isinstance(needs, str) else needs
            if uses_action or calls_reusable:
                self.assertIn("resolve-tornadovm", needs, f"{name} does not depend on resolve-tornadovm")
            for step in uses_action:
                self.assertEqual(step["with"]["ref"], expected, f"{name} passes a different revision")
                self.assertNotIn("env", step, f"{name} still sets TORNADO_ROOT by hand")
            if calls_reusable:
                self.assertEqual(job["with"]["tornadovm_ref"], expected, f"{name} does not forward the revision")

    def test_reusable_workflow_takes_the_revision_and_forwards_it(self):
        wf = self._load("standalone-inference.yml")
        inputs = wf[True]["workflow_call"]["inputs"] if True in wf else wf["on"]["workflow_call"]["inputs"]
        self.assertTrue(inputs["tornadovm_ref"]["required"])
        steps = wf["jobs"]["run"]["steps"]
        uses = [s for s in steps if s.get("uses") == "./.github/actions/setup-tornadovm"]
        self.assertEqual(len(uses), 1)
        self.assertEqual(uses[0]["with"]["ref"], "${{ inputs.tornadovm_ref }}")

    def test_reproduction_input_exists(self):
        wf = self._load("build-and-run.yml")
        on = wf[True] if True in wf else wf["on"]
        self.assertIn("tornadovm_ref", on["workflow_dispatch"]["inputs"])

    def test_development_build_does_not_require_pom_agreement(self):
        text = (REPO_ROOT / ".github" / "workflows" / "build-and-run.yml").read_text()
        self.assertNotIn("update tornadovm.base.version", text)
        self.assertIn('-Dtornadovm.version="$TORNADOVM_VERSION"', text)
        self.assertIn('-Dmaven.repo.local="$MAVEN_REPO_LOCAL"', text)
        self.assertIn("tornado-runtime", text)

    def test_release_workflows_verify_against_an_empty_repository(self):
        for name in ("prepare-release.yml", "bump-tornadovm-version.yml", "deploy-maven-central.yml"):
            text = (REPO_ROOT / ".github" / "workflows" / name).read_text()
            self.assertIn("-P release", text, name)
            self.assertIn('-Dmaven.repo.local="$(mktemp -d)"', text, name)
        self.assertIn("set-tornadovm-release.sh", (REPO_ROOT / ".github" / "workflows" / "prepare-release.yml").read_text())



def fake_install(root, backend, jdk, sha, recipe, version, java_major=None, jars=("tornado-api", "tornado-runtime", "tornado-drivers-cuda"), repo_jars=None, corrupt=()):
    """A prepared installation as setup would leave it: SDK jars, a repository holding the same
    bytes (or missing/different ones, for the failure cases), and a provenance file."""
    d = root / f"{backend}-jdk{jdk}" / f"{sha}-r{recipe}"
    sdk = d / "TornadoVM" / "dist" / f"tornadovm-{version}-{backend}-linux-amd64" / f"tornadovm-{version}-{backend}"
    (sdk / "share" / "java" / "tornado").mkdir(parents=True)
    (sdk / "bin").mkdir()
    repo = d / "m2" / "io" / "github" / "beehive-lab"
    for j in jars:
        content = f"{j}-{version}-{sha}".encode()
        (sdk / "share" / "java" / "tornado" / f"{j}-{version}.jar").write_bytes(content)
        if repo_jars is None or j in repo_jars:
            (repo / j / version).mkdir(parents=True, exist_ok=True)
            (repo / j / version / f"{j}-{version}.jar").write_bytes(content if j not in corrupt else b"other")
    prov = {"ref": sha, "artifact_version": version, "backend": backend, "jdk": str(jdk), "java_major": str(java_major or jdk),
            "java": f"openjdk version \"{java_major or jdk}\"", "recipe": recipe, "install_dir": str(d), "sdk_dir": str(sdk),
            "maven_repo_local": str(d / "m2"), "built_at": "2026-09-21T00:00:00Z", "host": "test"}
    (d / "provenance.json").write_text(json.dumps(prov))
    return d


class ArtifactConsistency(unittest.TestCase):
    """build refuses an installation whose repository does not hold the SDK's own artifacts."""

    SHA = "a" * 40

    def _build_dry(self, install, env_extra=None):
        env = {**os.environ, **(env_extra or {})}
        return run([str(DEV), "build", "--install", str(install), "--dry-run"], env=env)

    def test_cache_hit_builds_with_the_recorded_version_and_repository(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            major = run(["java", "-version"]).stderr.split('"')[1].split(".")[0]
            d = fake_install(root, "cuda", major, self.SHA, "r1", f"6.1.1-jdk{'21' if major == '21' else '22plus'}-dev")
            r = self._build_dry(d)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn(f"-Dtornadovm.version=6.1.1-jdk{'21' if major == '21' else '22plus'}-dev", r.stdout)
            self.assertIn(f"-Dmaven.repo.local={d}/m2", r.stdout)

    def test_a_newer_base_version_than_the_pom_default_is_used_as_produced(self):
        # develop moved to 6.2.0: the build passes what the SDK produced, no POM edit required.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            major = run(["java", "-version"]).stderr.split('"')[1].split(".")[0]
            v = f"6.2.0-jdk{'21' if major == '21' else '22plus'}-dev"
            d = fake_install(root, "cuda", major, self.SHA, "r1", v)
            r = self._build_dry(d)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn(f"-Dtornadovm.version={v}", r.stdout)

    def test_missing_repository_artifacts_are_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            major = run(["java", "-version"]).stderr.split('"')[1].split(".")[0]
            d = fake_install(root, "cuda", major, self.SHA, "r1", "6.1.1-jdk21-dev", repo_jars=("tornado-api",))
            r = self._build_dry(d)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("tornado-runtime (missing", r.stderr)
            self.assertIn("setup --ref", r.stderr)

    def test_mismatched_repository_artifacts_are_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            major = run(["java", "-version"]).stderr.split('"')[1].split(".")[0]
            d = fake_install(root, "cuda", major, self.SHA, "r1", "6.1.1-jdk21-dev", corrupt=("tornado-runtime",))
            r = self._build_dry(d)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("tornado-runtime (repository jar differs", r.stderr)

    def test_the_installations_jdk_line_must_match_the_running_java(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            major = run(["java", "-version"]).stderr.split('"')[1].split(".")[0]
            other = "25" if major == "21" else "21"
            d = fake_install(root, "cuda", other, self.SHA, "r1", f"6.1.1-jdk{'22plus' if other == '25' else '21'}-dev")
            r = self._build_dry(d)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn(f"built for JDK {other}", r.stderr)


class ConcurrentInstallationsDoNotMix(unittest.TestCase):
    """Two revisions prepared on one line: a command bound to one installation reads that
    installation's version, SDK and repository even while `current` is moved to the other."""

    def test_bound_commands_ignore_a_moving_current_pointer(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            major = run(["java", "-version"]).stderr.split('"')[1].split(".")[0]
            suffix = "21" if major == "21" else "22plus"
            a = fake_install(root, "cuda", major, "a" * 40, "r1", f"6.1.1-jdk{suffix}-dev")
            b = fake_install(root, "cuda", major, "b" * 40, "r1", f"6.2.0-jdk{suffix}-dev")
            line = root / f"cuda-jdk{major}"
            for current, bound in ((a, b), (b, a)):
                (line / "current").unlink(missing_ok=True)
                (line / "current").symlink_to(current.name)
                # Bound to the OTHER installation: every field must come from it, not from current.
                r = run([str(DEV), "build", "--install", str(bound), "--dry-run"])
                self.assertEqual(r.returncode, 0, r.stderr)
                bound_prov = json.loads((bound / "provenance.json").read_text())
                self.assertIn(f"-Dtornadovm.version={bound_prov['artifact_version']}", r.stdout)
                self.assertIn(f"-Dmaven.repo.local={bound}/m2", r.stdout)
                self.assertNotIn(str(current / "m2"), r.stdout)
                e = run([str(DEV), "env", "--install", str(bound)])
                self.assertIn(bound_prov["sdk_dir"], e.stdout)
                self.assertNotIn(json.loads((current / "provenance.json").read_text())["sdk_dir"], e.stdout)
            # Unbound commands resolve current once, to an immutable directory.
            (line / "current").unlink(); (line / "current").symlink_to(a.name)
            e = run([str(DEV), "env", "--backend", "cuda", "--jdk", major, "--root", str(root)])
            self.assertIn(str(a), e.stdout)
            self.assertNotIn("current", e.stdout)

    def test_setup_never_prunes_and_prune_is_explicit(self):
        src = DEV.read_text()
        self.assertNotIn("xargs -r rm -rf", src)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            major = run(["java", "-version"]).stderr.split('"')[1].split(".")[0]
            a = fake_install(root, "cuda", major, "a" * 40, "r1", "6.1.1-jdk21-dev")
            b = fake_install(root, "cuda", major, "b" * 40, "r1", "6.1.1-jdk21-dev")
            line = root / f"cuda-jdk{major}"
            (line / "current").symlink_to(a.name)
            r = run([str(DEV), "prune", "--backend", "cuda", "--jdk", major, "--root", str(root)])
            self.assertIn("would remove", r.stdout)
            self.assertTrue(b.exists())
            r = run([str(DEV), "prune", "--backend", "cuda", "--jdk", major, "--root", str(root), "--yes"])
            self.assertFalse(b.exists())
            self.assertTrue(a.exists())


if __name__ == "__main__":
    unittest.main()
