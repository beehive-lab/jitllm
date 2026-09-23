# jitllm Release Workflows

GitHub Actions workflows for automating releases to Maven Central.

## 📁 Files

Available in `.github/workflows/`:

| File | Purpose |
|------|---------|
| `prepare-release.yml` | Creates release branch, bumps versions, generates changelog, opens PR |
| `finalize-release.yml` | Creates git tag and GitHub Release when release PR merges |
| `deploy-maven-central.yml` | Deploys to Maven Central when tag is pushed |

## 🔄 Release Flow

```
1. PREPARE (manual trigger)
   └── Creates release/X.Y.Z branch + PR
   
2. REVIEW & MERGE (manual)
   └── Review PR, CI runs, merge when ready
   
3. FINALIZE (auto on PR merge)
   └── Creates tag vX.Y.Z + GitHub Release
   
4. DEPLOY (auto on tag push)
   └── Publishes to Maven Central
```

## 🚀 Usage

### Starting a Release

1. Go to **Actions** → **Prepare jitllm Release**
2. Click **Run workflow**
3. Enter version (e.g., `0.2.3`) and previous version (e.g., `0.2.2`)
4. Review and merge the created PR
5. Everything else is automatic!

### Manual Override

```bash
# Just create tag manually (skips prepare/finalize)
git tag -a v0.2.3 -m "Release 0.2.3"
git push origin v0.2.3
# → deploy-maven-central triggers automatically
```

## 🔐 Required Secrets

| Secret | Description |
|--------|-------------|
| `OSSRH_USERNAME` | Maven Central username |
| `OSSRH_TOKEN` | Maven Central token |
| `GPG_PRIVATE_KEY` | `gpg --armor --export-secret-keys KEY_ID` |
| `GPG_KEYNAME` | GPG key ID |
| `GPG_PASSPHRASE` | GPG passphrase |
