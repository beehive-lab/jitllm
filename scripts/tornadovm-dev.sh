#!/usr/bin/env bash
# tornadovm-dev.sh — the one way this repository obtains its development TornadoVM.
#
#   scripts/tornadovm-dev.sh setup   [--latest | --ref <sha>] [--backend cuda|opencl|metal]
#                                    [--jdk 21|25] [--root DIR] [--isolated-repo]
#   scripts/tornadovm-dev.sh refresh [--backend ...] [--jdk ...]      # advance to latest develop
#   scripts/tornadovm-dev.sh build   [mvn args...]                     # ./mvnw with the recorded version
#   scripts/tornadovm-dev.sh status                                    # what is prepared
#   scripts/tornadovm-dev.sh env                                       # shell lines to run jllm with it
#   scripts/tornadovm-dev.sh resolve-develop                           # print upstream develop's sha
#   scripts/tornadovm-dev.sh recipe                                    # print the build-recipe identity
#
# `setup` resolves upstream develop once (or takes an exact revision), clones that commit,
# builds the SDK for the backend and JDK, installs TornadoVM's Maven artifacts, and records
# what it built in <root>/<backend>-jdk<N>/<sha>-r<recipe>/provenance.json. A prepared
# revision is reused as is; only `refresh` or a different --ref builds again. Nothing here
# runs when jllm is launched.
#
# Layout (JLLM_DEV_ROOT, default ~/.jllm/tornadovm):
#   <root>/<backend>-jdk<N>/<sha>-r<recipe>/TornadoVM      the checkout and its dist/
#   <root>/<backend>-jdk<N>/<sha>-r<recipe>/provenance.json
#   <root>/<backend>-jdk<N>/<sha>-r<recipe>/m2             (--isolated-repo) Maven local repo
#   <root>/<backend>-jdk<N>/current -> <sha>-r<recipe>     what `build`/`env` use
#
# The user's own TornadoVM (TORNADOVM_HOME, SDKMAN) is never touched. Maven artifacts go to
# ~/.m2 by default — with the installed tornado-api jar checked against the SDK's before every
# build, so a stale coordinate from another revision is reinstalled — or to the per-SDK
# repository with --isolated-repo, which is what CI uses so concurrent builds of different
# revisions cannot overwrite one another.
set -euo pipefail

UPSTREAM=${TORNADOVM_UPSTREAM:-https://github.com/beehive-lab/TornadoVM.git}
ROOT=${JLLM_DEV_ROOT:-$HOME/.jllm/tornadovm}
BACKEND=${JLLM_DEV_BACKEND:-}
JDK=${JLLM_DEV_JDK:-21}
REF=""
LATEST=0
ISOLATED=0
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

die() { echo "tornadovm-dev: $*" >&2; exit 1; }

# The identity of everything in the recipe that can change what a build of the same commit
# produces: this script, the JDK, and the Python that TornadoVM's bin/compile runs under.
recipe_id() {
  {
    sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1
    "${JAVA_HOME:+$JAVA_HOME/bin/}java" -version 2>&1 | head -1
    python3 --version 2>&1
  } | sha256sum | cut -c1-12
}

resolve_develop() {
  local sha
  sha=$(git ls-remote "$UPSTREAM" refs/heads/develop | cut -f1)
  [[ "$sha" =~ ^[0-9a-f]{40}$ ]] || die "could not resolve develop from $UPSTREAM"
  echo "$sha"
}

detect_backend() {
  if [ -n "$BACKEND" ]; then return; fi
  if command -v nvidia-smi >/dev/null 2>&1; then BACKEND=cuda
  elif [ "$(uname -s)" = Darwin ]; then BACKEND=metal
  else BACKEND=opencl; fi
  echo "tornadovm-dev: backend not given, using $BACKEND"
}

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --latest) LATEST=1 ;;
      --ref) REF=$2; shift ;;
      --backend) BACKEND=$2; shift ;;
      --jdk) JDK=$2; shift ;;
      --root) ROOT=$2; shift ;;
      --isolated-repo) ISOLATED=1 ;;
      *) die "unknown option $1" ;;
    esac
    shift
  done
  case "$JDK" in 21|25) ;; *) die "--jdk must be 21 or 25 (got $JDK)" ;; esac
}

line_dir() { echo "$ROOT/$BACKEND-jdk$JDK"; }

sdk_dir_of() { # <install dir> -> dist SDK dir
  find "$1/TornadoVM/dist" -maxdepth 3 -type d -name "tornadovm-*-$BACKEND" 2>/dev/null | head -n 1
}

# The Maven coordinates the build produced, read from what it produced rather than guessed:
# the dist directory is tornadovm-<version>-<backend>, and share/java/tornado holds
# tornado-api-<version>.jar with the same version.
artifact_version_of() { # <sdk dir>
  local sdk=$1 v jar
  v=$(basename "$sdk"); v=${v#tornadovm-}; v=${v%-$BACKEND}
  jar=$(ls "$sdk/share/java/tornado/tornado-api-"*.jar 2>/dev/null | head -n 1)
  [ -n "$jar" ] || die "no tornado-api jar under $sdk/share/java/tornado"
  [ "$(basename "$jar")" = "tornado-api-$v.jar" ] || die "dist says $v but the jar is $(basename "$jar")"
  echo "$v"
}

expected_suffix() { if [ "$JDK" = 21 ]; then echo "-jdk21-dev"; else echo "-jdk22plus-dev"; fi; }

write_provenance() { # <install dir> <sha> <version> <sdk dir> <recipe>
  local built_at; built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  cat > "$1/provenance.json" <<EOF
{
  "ref": "$2",
  "artifact_version": "$3",
  "backend": "$BACKEND",
  "jdk": "$JDK",
  "recipe": "$5",
  "sdk_dir": "$4",
  "maven_repo_local": "$([ $ISOLATED = 1 ] && echo "$1/m2" || echo "")",
  "built_at": "$built_at",
  "host": "$(hostname)",
  "java": "$("${JAVA_HOME:+$JAVA_HOME/bin/}java" -version 2>&1 | head -1 | sed 's/"/\\"/g')"
}
EOF
}

# The SDK's own tornado-api jar and the one Maven will resolve must be the same bytes.
verify_or_install_artifacts() { # <install dir> <sdk dir> <version>
  local dir=$1 sdk=$2 v=$3 repo want have
  repo=$([ $ISOLATED = 1 ] && echo "$dir/m2" || echo "$HOME/.m2/repository")
  want=$(sha256sum "$sdk/share/java/tornado/tornado-api-$v.jar" | cut -d' ' -f1)
  have=$(sha256sum "$repo/io/github/beehive-lab/tornado-api/$v/tornado-api-$v.jar" 2>/dev/null | cut -d' ' -f1 || true)
  if [ "$want" != "$have" ]; then
    echo "tornadovm-dev: installing TornadoVM $v artifacts into $repo"
    ( cd "$dir/TornadoVM" && mvn -q -Dmaven.repo.local="$repo" -DskipTests install >/dev/null 2>&1 ) \
      || die "mvn install of the TornadoVM artifacts failed"
    have=$(sha256sum "$repo/io/github/beehive-lab/tornado-api/$v/tornado-api-$v.jar" | cut -d' ' -f1)
    [ "$want" = "$have" ] || die "installed tornado-api-$v.jar still differs from the SDK's"
  fi
}

do_setup() {
  parse_args "$@"
  detect_backend
  local sha recipe dir sdk v
  if [ -n "$REF" ]; then
    [[ "$REF" =~ ^[0-9a-f]{40}$ ]] || die "--ref must be a full 40-hex commit (got $REF)"
    sha=$REF
  elif [ $LATEST = 1 ] || [ ! -e "$(line_dir)/current" ]; then
    sha=$(resolve_develop)
    echo "tornadovm-dev: upstream develop is $sha"
  else
    sha=$(python3 -c "import json;print(json.load(open('$(line_dir)/current/provenance.json'))['ref'])")
    echo "tornadovm-dev: reusing prepared $sha (use --latest or refresh to advance)"
  fi
  recipe=$(recipe_id)
  dir="$(line_dir)/$sha-r$recipe"
  mkdir -p "$(line_dir)"
  exec 9>"$(line_dir)/.lock"; flock 9
  if [ -f "$dir/provenance.json" ] && [ -n "$(sdk_dir_of "$dir")" ]; then
    echo "tornadovm-dev: $BACKEND jdk$JDK $sha (recipe $recipe) already built"
  else
    rm -rf "$dir"; mkdir -p "$dir"
    git init -q "$dir/TornadoVM"
    ( cd "$dir/TornadoVM" \
      && git remote add origin "$UPSTREAM" \
      && git fetch -q --depth 1 origin "$sha" \
      && git checkout -q FETCH_HEAD )
    [ "$(git -C "$dir/TornadoVM" rev-parse HEAD)" = "$sha" ] || die "checked out $(git -C "$dir/TornadoVM" rev-parse HEAD), wanted $sha"
    # TornadoVM's make runs mvn install; with --isolated-repo MAVEN_OPTS points that install
    # at the per-SDK repository, so ~/.m2 is never written by a build meant to stay apart.
    local mvn_opts="${MAVEN_OPTS:-}"
    [ $ISOLATED = 1 ] && mvn_opts="$mvn_opts -Dmaven.repo.local=$dir/m2"
    ( cd "$dir/TornadoVM" \
      && export MAVEN_OPTS="$mvn_opts" \
      && python3 -m venv venv && . venv/bin/activate \
      && python -m pip install --quiet --upgrade pip && python -m pip install --quiet requests tqdm \
      && rm -rf graalJars && mkdir -p graalJars \
      && if [ "$JDK" = 21 ]; then make BACKEND=$BACKEND; else make jdk22plus BACKEND=$BACKEND; fi ) \
      || die "TornadoVM build failed in $dir/TornadoVM"
  fi
  sdk=$(sdk_dir_of "$dir"); [ -n "$sdk" ] || die "no SDK directory under $dir/TornadoVM/dist"
  v=$(artifact_version_of "$sdk")
  case "$v" in *"$(expected_suffix)") ;; *) die "SDK version $v does not end in $(expected_suffix) for JDK $JDK" ;; esac
  verify_or_install_artifacts "$dir" "$sdk" "$v"
  write_provenance "$dir" "$sha" "$v" "$sdk" "$recipe"
  ln -sfn "$sha-r$recipe" "$(line_dir)/current"
  # Keep the three newest identities of this line.
  ls -1dt "$(line_dir)"/*-r* 2>/dev/null | tail -n +4 | xargs -r rm -rf
  echo "tornadovm-dev: prepared $v from $sha ($BACKEND, jdk$JDK) at $dir"
  do_status
}

current_prov() {
  detect_backend >/dev/null
  local p="$(line_dir)/current/provenance.json"
  [ -f "$p" ] || die "nothing prepared for $BACKEND jdk$JDK; run: scripts/tornadovm-dev.sh setup"
  echo "$p"
}

prov_field() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2"; }

do_build() {
  # Options for this script come before "--"; everything after goes to Maven.
  local mvn_args=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --backend) BACKEND=$2; shift ;;
      --jdk) JDK=$2; shift ;;
      --root) ROOT=$2; shift ;;
      --) shift; mvn_args=("$@"); break ;;
      *) mvn_args+=("$1") ;;
    esac
    shift
  done
  local p v sdk repo
  p=$(current_prov); v=$(prov_field "$p" artifact_version); sdk=$(prov_field "$p" sdk_dir); repo=$(prov_field "$p" maven_repo_local)
  [ -n "$repo" ] && ISOLATED=1
  verify_or_install_artifacts "$(dirname "$p")" "$sdk" "$v"
  local extra=()
  [ -n "$repo" ] && extra+=("-Dmaven.repo.local=$repo")
  echo "tornadovm-dev: ./mvnw -Dtornadovm.version=$v ${extra[*]:-} ${mvn_args[*]:-}"
  ( cd "$HERE" && ./mvnw "-Dtornadovm.version=$v" "${extra[@]}" "${mvn_args[@]}" )
}

do_status() {
  detect_backend >/dev/null
  local p="$(line_dir)/current/provenance.json"
  if [ ! -f "$p" ]; then echo "nothing prepared for $BACKEND jdk$JDK under $ROOT"; return; fi
  python3 - "$p" <<'EOF'
import json,sys
p=json.load(open(sys.argv[1]))
print(f"TornadoVM {p['artifact_version']} from {p['ref']} ({p['backend']}, jdk{p['jdk']}, recipe {p['recipe']}), built {p['built_at']} on {p['host']}")
print(f"  SDK: {p['sdk_dir']}")
print(f"  Maven repo: {p['maven_repo_local'] or '~/.m2 (verified against the SDK jar before each build)'}")
EOF
}

do_env() {
  local p sdk
  p=$(current_prov); sdk=$(prov_field "$p" sdk_dir)
  echo "export TORNADOVM_HOME=$sdk"
  echo "export PATH=$sdk/bin:\$PATH"
  echo "# then: ./jllm --gpu --model <model.gguf> --prompt '...'"
}

case "${1:-}" in
  setup) shift; do_setup "$@" ;;
  refresh) shift; do_setup --latest "$@" ;;
  build) shift; do_build "$@" ;;
  status) shift; parse_args "$@"; do_status ;;
  env) shift; parse_args "$@"; do_env ;;
  resolve-develop) resolve_develop ;;
  recipe) echo "$(recipe_id)" ;;
  *) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
