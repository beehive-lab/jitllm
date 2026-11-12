# jitllm Release Workflows

GitHub Actions workflows for automating releases to Maven Central. Releases are tagged
`jitllm-vX.Y.Z` and published as `io.github.beehive-lab:jitllm:X.Y.Z-jdk21` and
`X.Y.Z-jdk22plus`. Every release job runs only in `beehive-lab/jitllm`.

## 📁 Files

Available in `.github/workflows/`:

| File | Name in Actions | Trigger | Purpose |
|------|-----------------|---------|---------|
| `release.yml` | **Release jitllm** | manual | One click: prepare → merge the release PR → finalize → deploy |
| `prepare-release.yml` | **Prepare jitllm Release** | manual, or called by `release.yml` | Creates `release/X.Y.Z`, sets the version and the TornadoVM release, regenerates the README dependency snippets, `CITATION.cff` and the changelog entry, opens the release PR |
| `finalize-release.yml` | **Finalize jitllm Release** | a merged PR from a `release/*` branch, manual, or called by `release.yml` | Creates tag `jitllm-vX.Y.Z` and the GitHub Release (notes from `CHANGELOG.md`), deletes the release branch |
| `deploy-maven-central.yml` | **Deploy to Maven Central** | after a successful **Finalize jitllm Release** run, manual, or called by `release.yml` | Builds both JDK lines with `-P release` and deploys them |
| `bump-tornadovm-version.yml` | **Bump TornadoVM version** | `repository_dispatch` (`tornadovm-release`) from TornadoVM's release pipeline, or manual | Sets `tornadovm.release.version`, checks the build against the published TornadoVM release on JDK 21 and 25, opens a PR |

## 🔄 Release Flow

```
1. PREPARE (manual: Release jitllm, or Prepare jitllm Release)
   └── Creates release/X.Y.Z branch + PR "Release X.Y.Z"

2. REVIEW & MERGE (manual, or automatic under Release jitllm)
   └── Review PR, CI runs, merge when ready

3. FINALIZE (auto on merge of a release/* PR)
   └── Creates tag jitllm-vX.Y.Z + GitHub Release "jitllm X.Y.Z"

4. DEPLOY (auto after a successful Finalize jitllm Release run)
   └── Publishes the -jdk21 and -jdk22plus artifacts to Maven Central
```

Under **Release jitllm** the four steps run as one workflow: the merge and tag are made there,
so the `pull_request` and `workflow_run` triggers of the individual workflows do not fire and
nothing runs twice.

## 🚀 Usage

### Starting a Release

1. Go to **Actions** → **Release jitllm** (or **Prepare jitllm Release** to stop at the PR)
2. Click **Run workflow**
3. Enter:
   - `version`: the new version, `X.Y.Z` (e.g. `2.0.1`)
   - `tornadovm_release`: the published TornadoVM release this release depends on (e.g. `8.0.0`);
     it must be on Maven Central, and the workflow compiles against it on JDK 21 and 25 before
     opening the PR
   - `previous_version`: the last release (e.g. `2.0.0`), used to collect the merged PRs for the
     changelog
   - `dry_run`: show the changes without creating a PR, tag, release or deployment
4. With **Prepare jitllm Release**: review and merge the PR; finalize and deploy follow on their own

### Manual Override

Run the later steps by hand from **Actions** when one of them needs repeating:

- **Finalize jitllm Release** with `version` `X.Y.Z`: tags the current `main` as `jitllm-vX.Y.Z`
  and creates the GitHub Release; deployment then follows automatically.
- **Deploy to Maven Central** with `tag` `jitllm-vX.Y.Z`: builds and deploys that tag. `jdk`
  (`all`, `jdk21`, `jdk22plus`) retries a single line; a version already on Maven Central cannot
  be uploaded again. `dry_run` builds and verifies without deploying.

Pushing a tag by hand does not deploy anything: no workflow is triggered by tags.

### TornadoVM releases

When TornadoVM publishes a release, its pipeline dispatches `tornadovm-release` to this
repository and **Bump TornadoVM version** opens a PR that moves `tornadovm.release.version`
in `pom.xml` (the dependency of `-P release` builds) to it. Run it manually with `version`
`X.Y.Z` to do the same for a given release. Development builds are not affected: they compile
against the SDK at `TORNADOVM_HOME`.

## 🔐 Required Secrets

| Secret | Description |
|--------|-------------|
| `OSSRH_USERNAME` | Maven Central username |
| `OSSRH_TOKEN` | Maven Central token |
| `GPG_PRIVATE_KEY` | `gpg --armor --export-secret-keys KEY_ID` |
| `GPG_KEYNAME` | GPG key ID |
| `GPG_PASSPHRASE` | GPG passphrase |
