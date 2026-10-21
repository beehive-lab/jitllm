---
name: release-jitllm
description: Cut a jitllm release (patch, minor or major) to Maven Central through the repo's release workflows, then verify the published artifacts and hand off to the framework integrations. Use when asked to "release jitllm X.Y.Z", "do a minor/patch release", bump the published version, retry a failed Maven Central deploy, or check whether a release is on Maven Central yet. See also update-langchain4j-integration, update-spring-ai-integration and update-quarkus-langchain4j-integration for what comes after.
---

# Release jitllm

A release is outward-facing and cannot be undone: once a version is on Maven Central it can never
be re-uploaded or changed. Go step by step, verify each step, and stop at the first surprise.

The mechanics live in `.github/workflows/` (see `docs/RELEASE-AUTOMATION.md`):

| workflow | trigger | does |
|---|---|---|
| `prepare-release.yml` ("Prepare jitllm Release") | manual | bumps `<revision>`, `CITATION.cff` and the README snippets, writes the `CHANGELOG.md` section, pins the TornadoVM release, compiles against it on JDK 21 and 25, opens PR `release/X.Y.Z` |
| `finalize-release.yml` | release PR merged | tags `jitllm-vX.Y.Z`, creates the GitHub Release, deletes the branch |
| `deploy-maven-central.yml` | finalize completed | builds `-jdk21` (JDK 21) and `-jdk22plus` (JDK 25, release 22), signs and publishes |

Tags are `jitllm-vX.Y.Z` (older tags without the prefix predate the rename).

## Inputs

Determine, or ask for if you cannot determine them:

- **version**: `X.Y.Z`. Patch for fixes, minor for new features or API additions, major when
  users of the previous release must change something or will see different behaviour: new
  defaults, different outputs for the same model and prompt, a higher TornadoVM requirement, a
  renamed or removed flag, property or API.
- **previous_version**: the last published version, i.e. the newest `jitllm-v*` tag
  (`git tag --sort=-creatordate | head`).
- **tornadovm_release**: a TornadoVM release **published on Maven Central** that carries every
  TornadoVM API `main` uses. Default to the one the previous release used
  (`git show jitllm-v<prev>:pom.xml | grep tornadovm.release.version`), and check for newer ones:
  `curl -s https://repo1.maven.org/maven2/io/github/beehive-lab/tornado-api/maven-metadata.xml`.
  `main` builds against TornadoVM `develop`, so this is the input most likely to be wrong.

Never guess a version. If `main` has nothing but `perf: record run` commits since the last tag,
say so instead of releasing.

## 1. Know what is going out

```bash
git fetch --tags origin
git log --oneline jitllm-v<prev>..origin/main | grep -v "perf: record run"
```

The changelog is generated from merged PRs, so check the list matches what you expect. Look for
anything half-landed (a feature merged without its follow-up) and raise it before releasing, and
ask whether open PRs the user owns should go in first (`gh pr list -R beehive-lab/jitllm --state
open`); including one means merging it before step 3, not editing the release PR.

For a **major** release, collect the breaking changes from the merged PRs' own descriptions; their
titles rarely say so:

```bash
gh pr list -R beehive-lab/jitllm --state merged --search "merged:>$(git log -1 --format=%cI jitllm-v<prev>)" \
  --limit 100 --json number -q '.[].number' | while read n; do
  gh pr view $n -R beehive-lab/jitllm --json number,title,body -q '"#\(.number) \(.title)\n" +
    ((.body // "") | split("\n") | map(select(test("(?i)break|remov|renam|default|no longer|deprecat|drop|behaviou?r change"))) | join("\n"))'
done
```

Keep the ones a user of the previous release would notice; you add them to the changelog in step 3.

## 2. Dry run

```bash
gh workflow run prepare-release.yml -R beehive-lab/jitllm \
  -f version=X.Y.Z -f previous_version=<prev> -f tornadovm_release=<tvm> -f dry_run=true
gh run list -R beehive-lab/jitllm --workflow prepare-release.yml --limit 1
```

Wait for it (`gh run watch <id> --exit-status`), then read the log. It queues behind `main`
builds on the shared runners (2.0.0: about 6 minutes queued, 6 running). It must show
`jitllm X.Y.Z-jdk21 against tornado-api <tvm>-jdk21` and the JDK 25 build succeeding. A compile
failure here means `main` uses a TornadoVM API the chosen release lacks: stop, and report which
API and which TornadoVM release would carry it. Do not "fix" it by releasing against `-dev`
coordinates; the `release` profile refuses them anyway.

## 3. Prepare

The same command with `dry_run=false`. It opens PR `release/X.Y.Z` titled `Release X.Y.Z`.
Review its diff: only `pom.xml` (`<revision>`), `CITATION.cff`, the README install snippets and a
new `CHANGELOG.md` section should change.

The changelog groups PRs by label, so unlabelled PRs all land under "Other Changes", and it has no
section for breaking changes. The GitHub Release notes are this same section (finalize extracts
it), so for a major release add a `### Breaking changes` list at the top of the version's section,
one line per change with its PR, as a commit on `release/X.Y.Z` before merging (this re-runs the
PR's checks):

```bash
git fetch origin release/X.Y.Z && git worktree add ../jitllm-release origin/release/X.Y.Z -b release-X.Y.Z-local
# edit CHANGELOG.md, then
git commit -am "CHANGELOG: list the breaking changes of X.Y.Z" && git push origin HEAD:release/X.Y.Z
```

## 4. Merge

Wait for `code-quality` and `jitllm Build & Run` to pass. They run on the self-hosted GPU runners
(`cyclone`, `Mac`), which are often busy with `main` builds, so queued is normal; check with
`gh run list -R beehive-lab/jitllm --limit 10`. `license/cla` stays pending on bot-created release
PRs; previous release PRs were merged with it pending. `create-release-tag` and `cleanup-branch`
belong to finalize and only turn green *after* the merge, so seeing them pass means someone has
already merged. (`Finalize jitllm Release` and `Deploy to Maven Central` runs also appear after
every ordinary PR merge; they skip themselves and are not a release.) Check `gh pr view <n> --json state,mergedBy` before merging yourself. Merge only when the rest is green, and only
with the user's go-ahead if they did not ask for the release end to end.

```bash
gh pr merge <n> -R beehive-lab/jitllm --merge
```

## 5. Finalize and deploy (automatic), then verify

Merging triggers finalize, which triggers deploy (1.0.2: finalize about 1 minute, deploy about
7 minutes; 2.0.0: finalize 14 minutes, most of it queued for a runner, deploy 8 minutes). Merging
also cancels the release PR's own in-flight checks, so a `code-quality` or build job shown as
failed after the merge is usually cancelled (`gh run view <id> --json jobs`). What counts then is
what changed since the last green `main` build: `gh run list -R beehive-lab/jitllm --workflow
"jitllm Build & Run" --branch main --status success --limit 1`. Watch both runs:

```bash
gh run list -R beehive-lab/jitllm --workflow finalize-release.yml --limit 1
gh run list -R beehive-lab/jitllm --workflow deploy-maven-central.yml --limit 1
git fetch --tags origin && git tag -l "jitllm-vX.Y.Z"
gh release view jitllm-vX.Y.Z -R beehive-lab/jitllm
```

Check the release notes: `gh release view jitllm-vX.Y.Z -R beehive-lab/jitllm`. Finalize is meant
to copy the version's `CHANGELOG.md` section, but for 1.0.2 and 2.0.0 it fell back to "See
CHANGELOG.md for details.". Set them from the tag's changelog:

```bash
git show jitllm-vX.Y.Z:CHANGELOG.md | awk '/^## \[X\.Y\.Z\]/{f=1;next} f && /^## \[/{exit} f{print}' > notes.md
gh release edit jitllm-vX.Y.Z -R beehive-lab/jitllm --notes-file notes.md
```

Maven Central takes a while to serve a newly published version (1.0.2: about 14 minutes after the
deploy finished; 2.0.0: under 11 minutes; allow up to an hour). Poll until both lines resolve and have the right bytecode (Java 21 = major 65 for
`-jdk21`, Java 22 = major 66 for `-jdk22plus`):

```bash
.claude/skills/release-jitllm/scripts/wait-for-central.sh X.Y.Z
```

If one line's deploy failed, retry only that line. The other is already published and cannot be
uploaded again:

```bash
gh workflow run deploy-maven-central.yml -R beehive-lab/jitllm -f tag=jitllm-vX.Y.Z -f jdk=jdk22plus
```

## 6. Hand off

Once both artifacts resolve, update the framework integrations to the new version with their
skills (`update-langchain4j-integration`, `update-spring-ai-integration`,
`update-quarkus-langchain4j-integration`), and update the tracking issue for the integrations
(e.g. #178) with the new version.

## Report

State the version, the TornadoVM release it depends on, the release PR, the tag, the GitHub
Release, the two Maven Central coordinates and when they resolved, and anything skipped or retried.
