#!/usr/bin/env python3
"""Assert that a jitllm jar carries this project's classes and nothing bundled.

The published jar is a library: TornadoVM and everything it depends on come from the
TornadoVM SDK at runtime, as named modules on the module path. A bundled copy of any of
them (uk/ac/manchester/tornado, org/graalvm, log4j, JMH, ...) sits on the consumer's
class path next to those modules, and a container with its own class loader, such as
Quarkus, then loads two copies of the same type and fails with a loader constraint
violation (issue #176).

Usage: scripts/check_library_jar.py target/jitllm-<version>.jar [...]
Exit status 0 when every jar is clean, 1 otherwise.
"""

import sys
import zipfile

PROJECT_PREFIX = "org/beehive/jitllm/"

# Entries outside the project's package that a plain library jar legitimately has.
ALLOWED_META_INF = (
    "META-INF/MANIFEST.MF",
    "META-INF/jpms.args",  # written by maven-compiler-plugin for --add-modules
)
ALLOWED_META_INF_PREFIXES = (
    "META-INF/services/org.beehive.jitllm.",
    "META-INF/maven/io.github.beehive-lab/jitllm/",
)

# Named in the report because they are the ones that have broken consumers before.
KNOWN_BUNDLED = ("uk/ac/manchester/tornado/", "org/graalvm/")


def foreign_entries(names):
    """Return the entries that do not belong in the library jar."""
    foreign = []
    for name in names:
        if name.endswith("/"):
            continue  # directory entries carry no content
        if name.startswith(PROJECT_PREFIX):
            continue
        if name in ALLOWED_META_INF or name.startswith(ALLOWED_META_INF_PREFIXES):
            continue
        foreign.append(name)
    return foreign


def check(path):
    """Return a list of problems with the jar at path (empty when it is clean)."""
    with zipfile.ZipFile(path) as jar:
        names = jar.namelist()
    problems = []
    if not any(n.startswith(PROJECT_PREFIX) and n.endswith(".class") for n in names):
        problems.append(f"{path}: contains no {PROJECT_PREFIX} classes")
    foreign = foreign_entries(names)
    for prefix in KNOWN_BUNDLED:
        hits = [n for n in foreign if n.startswith(prefix)]
        if hits:
            problems.append(f"{path}: bundles {len(hits)} entries under {prefix} (e.g. {hits[0]}); the TornadoVM SDK supplies these")
    other = [n for n in foreign if not n.startswith(KNOWN_BUNDLED)]
    if other:
        shown = ", ".join(other[:5]) + (", ..." if len(other) > 5 else "")
        problems.append(f"{path}: {len(other)} entries outside {PROJECT_PREFIX} and the project's META-INF files: {shown}")
    return problems


def main(argv):
    if len(argv) < 2:
        print("usage: check_library_jar.py <jar> [<jar> ...]", file=sys.stderr)
        return 2
    problems = []
    for path in argv[1:]:
        problems.extend(check(path))
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    if not problems:
        print("library jar clean: " + ", ".join(argv[1:]))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
