#!/usr/bin/env bash
# tornadovm-dev.sh — the one way this repository obtains its development TornadoVM, for local
# builds and for CI alike.
#
#   scripts/tornadovm-dev.sh setup   [--latest | --ref <sha>] [--backend cuda|opencl|metal]
#                                    [--jdk 21|25] [--root DIR] [--result FILE]
#   scripts/tornadovm-dev.sh refresh [--backend ...] [--jdk ...]      # advance to latest develop
#   scripts/tornadovm-dev.sh build   [--install DIR] [--dry-run] [mvn args...]
#   scripts/tornadovm-dev.sh status  [--install DIR]                   # what is prepared
#   scripts/tornadovm-dev.sh env     [--install DIR]                   # shell lines to run jllm with it
#   scripts/tornadovm-dev.sh prune   [--yes]                           # remove non-current installations
#   scripts/tornadovm-dev.sh resolve-develop                           # print upstream develop's sha
#   scripts/tornadovm-dev.sh recipe                                    # print the build-recipe identity
#
# `setup` resolves upstream develop once (or takes an exact revision), clones that commit, builds
# the SDK for the backend and JDK with TornadoVM's own make, and records what it built. Every
# installation is immutable and self-contained:
#
#   <root>/<backend>-jdk<N>/<sha>-r<recipe>/TornadoVM        the checkout and its dist/
#   <root>/<backend>-jdk<N>/<sha>-r<recipe>/m2               the Maven repository holding ITS artifacts
#   <root>/<backend>-jdk<N>/<sha>-r<recipe>/provenance.json  ref, version, backend, jdk, java, recipe, paths
#   <root>/<backend>-jdk<N>/current -> <sha>-r<recipe>       a convenience pointer for local commands
#
# The Maven repository is per installation, always: TornadoVM's develop artifacts carry the same
# coordinates (6.1.1-jdk21-dev) for every commit, so a shared ~/.m2 cannot say which commit a jar
# came from and two builds of different commits would overwrite each other. `build` runs ./mvnw
# with -Dmaven.repo.local=<that repository> and -Dtornadovm.version=<the version that SDK
# produced>; plain ./mvnw does NOT see these artifacts (see README, "Build from source").
# A prepared revision is reused as is; only `refresh`, a different --ref, or a changed recipe
# builds again. Nothing here runs when jllm is launched, and nothing is deleted except by `prune`.
# The user's own TornadoVM (TORNADOVM_HOME, SDKMAN) is never touched.
set -euo pipefail

UPSTREAM=${TORNADOVM_UPSTREAM:-https://github.com/beehive-lab/TornadoVM.git}
ROOT=${JLLM_DEV_ROOT:-$HOME/.jllm/tornadovm}
BACKEND=${JLLM_DEV_BACKEND:-}
JDK=${JLLM_DEV_JDK:-21}
REF=""
LATEST=0
RESULT=""
INSTALL=""
DRY_RUN=0
YES=0
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

die() { echo "tornadovm-dev: $*" >&2; exit 1; }

# macOS has neither sha256sum nor flock (no util-linux); the same script must run there.
sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum | cut -d' ' -f1; else shasum -a 256 | cut -d' ' -f1; fi; }
sha256_file() { sha256_of < "$1"; }

# One builder per line at a time: flock where it exists, otherwise an atomic mkdir lock that is
# released on exit.
take_lock() {
  local lock=$1
  if command -v flock >/dev/null 2>&1; then
    exec 9>"$lock"; flock 9
  else
    local d="$lock.d" waited=0
    until mkdir "$d" 2>/dev/null; do
      sleep 5; waited=$((waited + 5))
      [ $waited -lt 7200 ] || die "could not take $d within two hours; remove it if no build is running"
    done
    trap 'rmdir "'"$d"'" 2>/dev/null' EXIT
  fi
}

java_bin() { echo "${JAVA_HOME:+$JAVA_HOME/bin/}java"; }

# The major version of the java that will build (and that TornadoVM's make will use): --jdk is a
# request, this is the fact.
java_major() {
  local v
  v=$("$(java_bin)" -version 2>&1 | head -1 | sed -E 's/.*version "([0-9]+).*/\1/')
  [[ "$v" =~ ^[0-9]+$ ]] || die "cannot read the java version from $(java_bin)"
  echo "$v"
}

# The identity of everything in the recipe that can change what a build of the same commit
# produces: this script, the JDK, and the Python that TornadoVM's bin/compile runs under.
recipe_id() {
  {
    sha256_file "${BASH_SOURCE[0]}"
    "$(java_bin)" -version 2>&1 | head -1
    python3 --version 2>&1
  } | sha256_of | cut -c1-12
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
  echo "tornadovm-dev: backend not given, using $BACKEND" >&2
}

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --latest) LATEST=1 ;;
      --ref) REF=$2; shift ;;
      --backend) BACKEND=$2; shift ;;
      --jdk) JDK=$2; shift ;;
      --root) ROOT=$2; shift ;;
      --result) RESULT=$2; shift ;;
      --install) INSTALL=$2; shift ;;
      --dry-run) DRY_RUN=1 ;;
      --yes) YES=1 ;;
      --isolated-repo) ;; # always the case now; accepted for older callers
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

# The Maven coordinates the build produced, read from what it produced rather than guessed: the
# dist directory is tornadovm-<version>-<backend>, and share/java/tornado holds
# tornado-api-<version>.jar with the same version.
artifact_version_of() { # <sdk dir>
  local sdk=$1 v jar
  v=$(basename "$sdk"); v=${v#tornadovm-}; v=${v%-$BACKEND}
  jar=$(ls "$sdk/share/java/tornado/tornado-api-"*.jar 2>/dev/null | head -n 1)
  [ -n "$jar" ] || die "no tornado-api jar under $sdk/share/java/tornado"
  [ "$(basename "$jar")" = "tornado-api-$v.jar" ] || die "dist says $v but the jar is $(basename "$jar")"
  echo "$v"
}

# TornadoVM's JDK lines: JDK 21 builds -jdk21-dev artifacts, everything from 22 up -jdk22plus-dev.
expected_suffix() { if [ "$JDK" = 21 ]; then echo "-jdk21-dev"; else echo "-jdk22plus-dev"; fi; }

# The SDK's own jars and the ones Maven will resolve from the installation's repository must be the
# same bytes, for every artifact the SDK ships and the repository holds under the same version:
# tornado-api, tornado-runtime, and the driver/other modules. Prints the mismatching names, one
# per line, and nothing when they all agree.
artifact_mismatches() { # <install dir> <sdk dir> <version>
  local dir=$1 sdk=$2 v=$3 repo="$1/m2/io/github/beehive-lab" jar name want have
  for jar in "$sdk"/share/java/tornado/tornado-*-"$v".jar; do
    [ -e "$jar" ] || { echo "no-sdk-jars"; return; }
    name=$(basename "$jar" "-$v.jar")
    if [ ! -f "$repo/$name/$v/$name-$v.jar" ]; then
      echo "$name (missing from $dir/m2)"
      continue
    fi
    want=$(sha256_file "$jar"); have=$(sha256_file "$repo/$name/$v/$name-$v.jar")
    [ "$want" = "$have" ] || echo "$name (repository jar differs from the SDK's)"
  done
}

write_provenance() { # <install dir> <sha> <version> <sdk dir> <recipe>
  local built_at; built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  cat > "$1/provenance.json" <<EOF
{
  "ref": "$2",
  "artifact_version": "$3",
  "backend": "$BACKEND",
  "jdk": "$JDK",
  "java_major": "$(java_major)",
  "java": "$("$(java_bin)" -version 2>&1 | head -1 | sed 's/"/\\"/g')",
  "recipe": "$5",
  "install_dir": "$1",
  "sdk_dir": "$4",
  "maven_repo_local": "$1/m2",
  "built_at": "$built_at",
  "host": "$(hostname)"
}
EOF
}

# TornadoVM's supported build entrypoint, with the install's own Maven repository so its make
# (which runs mvn install) writes the artifacts there and nowhere else.
build_tornadovm() { # <install dir>
  local dir=$1
  ( cd "$dir/TornadoVM" \
    && export MAVEN_OPTS="${MAVEN_OPTS:-} -Dmaven.repo.local=$dir/m2" \
    && { [ -d venv ] || python3 -m venv venv; } && . venv/bin/activate \
    && python -m pip install --quiet --upgrade pip && python -m pip install --quiet requests tqdm \
    && rm -rf graalJars && mkdir -p graalJars \
    && if [ "$JDK" = 21 ]; then make BACKEND=$BACKEND; else make jdk22plus BACKEND=$BACKEND; fi ) \
    || die "TornadoVM build failed in $dir/TornadoVM"
}

do_setup() {
  parse_args "$@"
  detect_backend
  local major; major=$(java_major)
  [ "$major" = "$JDK" ] || die "--jdk $JDK but the java on JAVA_HOME/PATH is $major ($(java_bin)); select JDK $JDK first"
  local sha recipe dir sdk v mism
  if [ -n "$REF" ]; then
    [[ "$REF" =~ ^[0-9a-f]{40}$ ]] || die "--ref must be a full 40-hex commit (got $REF)"
    sha=$REF
  elif [ $LATEST = 1 ] || [ ! -e "$(line_dir)/current" ]; then
    sha=$(resolve_develop)
    echo "tornadovm-dev: upstream develop is $sha"
  else
    sha=$(python3 -c "import json;print(json.load(open('$(cd "$(line_dir)/current" && pwd -P)/provenance.json'))['ref'])")
    echo "tornadovm-dev: reusing prepared $sha (use --latest or refresh to advance)"
  fi
  recipe=$(recipe_id)
  dir="$(line_dir)/$sha-r$recipe"
  mkdir -p "$(line_dir)"
  take_lock "$(line_dir)/.lock"
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
    build_tornadovm "$dir"
  fi
  sdk=$(sdk_dir_of "$dir"); [ -n "$sdk" ] || die "no SDK directory under $dir/TornadoVM/dist"
  v=$(artifact_version_of "$sdk")
  case "$v" in *"$(expected_suffix)") ;; *) die "SDK version $v does not end in $(expected_suffix) for JDK $JDK" ;; esac
  # The repository must hold exactly the SDK's artifacts. A reused installation whose repository
  # was damaged is rebuilt through the same entrypoint, never patched with a bare mvn install.
  mism=$(artifact_mismatches "$dir" "$sdk" "$v")
  if [ -n "$mism" ]; then
    echo "tornadovm-dev: repository artifacts do not match the SDK; rebuilding:" >&2
    echo "$mism" | sed 's/^/  /' >&2
    build_tornadovm "$dir"
    sdk=$(sdk_dir_of "$dir"); v=$(artifact_version_of "$sdk")
    mism=$(artifact_mismatches "$dir" "$sdk" "$v")
    [ -z "$mism" ] || die "repository still does not match the SDK after a rebuild: $mism"
  fi
  write_provenance "$dir" "$sha" "$v" "$sdk" "$recipe"
  ln -sfn "$sha-r$recipe" "$(line_dir)/current"
  if [ -n "$RESULT" ]; then cp "$dir/provenance.json" "$RESULT"; fi
  echo "tornadovm-dev: prepared $v from $sha ($BACKEND, jdk$JDK) at $dir"
  echo "install_dir=$dir"
}

# The installation a local command works on: --install if given, else `current` resolved ONCE to
# its immutable directory. Everything after reads that directory's provenance, never the pointer.
resolve_install() {
  if [ -n "$INSTALL" ]; then
    [ -f "$INSTALL/provenance.json" ] || die "$INSTALL has no provenance.json"
    cd "$INSTALL" && pwd -P; return
  fi
  detect_backend
  local cur="$(line_dir)/current"
  [ -e "$cur" ] || die "nothing prepared for $BACKEND jdk$JDK; run: scripts/tornadovm-dev.sh setup"
  cd "$cur" && pwd -P
}

prov_field() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2"; }

do_build() {
  local mvn_args=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --backend) BACKEND=$2; shift ;;
      --jdk) JDK=$2; shift ;;
      --root) ROOT=$2; shift ;;
      --install) INSTALL=$2; shift ;;
      --dry-run) DRY_RUN=1 ;;
      --) shift; mvn_args=("$@"); break ;;
      *) mvn_args+=("$1") ;;
    esac
    shift
  done
  local dir p v sdk repo jdk major mism
  dir=$(resolve_install); p="$dir/provenance.json"
  v=$(prov_field "$p" artifact_version); sdk=$(prov_field "$p" sdk_dir); repo=$(prov_field "$p" maven_repo_local); jdk=$(prov_field "$p" jdk)
  BACKEND=$(prov_field "$p" backend)
  major=$(java_major)
  [ "$major" = "$jdk" ] || die "the installation was built for JDK $jdk but the java on JAVA_HOME/PATH is $major; select JDK $jdk"
  mism=$(artifact_mismatches "$dir" "$sdk" "$v")
  [ -z "$mism" ] || die "the repository at $repo does not match the SDK ($mism); run: scripts/tornadovm-dev.sh setup --ref $(prov_field "$p" ref) --backend $BACKEND --jdk $jdk"
  echo "tornadovm-dev: ./mvnw -Dtornadovm.version=$v -Dmaven.repo.local=$repo ${mvn_args[*]:-}"
  [ $DRY_RUN = 1 ] && return 0
  ( cd "$HERE" && ./mvnw "-Dtornadovm.version=$v" "-Dmaven.repo.local=$repo" "${mvn_args[@]}" )
}

do_status() {
  local dir; dir=$(resolve_install)
  python3 - "$dir/provenance.json" <<'EOF'
import json,sys
p=json.load(open(sys.argv[1]))
print(f"TornadoVM {p['artifact_version']} from {p['ref']} ({p['backend']}, jdk{p['jdk']}, {p['java']}, recipe {p['recipe']}), built {p['built_at']} on {p['host']}")
print(f"  install: {p['install_dir']}")
print(f"  SDK:     {p['sdk_dir']}")
print(f"  Maven:   {p['maven_repo_local']}  (build with: scripts/tornadovm-dev.sh build ...)")
EOF
}

do_env() {
  local dir p sdk
  dir=$(resolve_install); p="$dir/provenance.json"; sdk=$(prov_field "$p" sdk_dir)
  echo "export TORNADOVM_HOME=$sdk"
  echo "export PATH=$sdk/bin:\$PATH"
  echo "# TornadoVM $(prov_field "$p" artifact_version) from $(prov_field "$p" ref); then: ./jllm --gpu --model <model.gguf> --prompt '...'"
}

# Explicit cleanup only: nothing is deleted by setup, because another build or a running jllm may
# be using an older installation. Lists what would go; --yes removes it.
do_prune() {
  parse_args "$@"
  detect_backend
  local cur d n=0
  cur=$( [ -e "$(line_dir)/current" ] && (cd "$(line_dir)/current" && pwd -P) || echo "")
  for d in "$(line_dir)"/*-r*; do
    [ -d "$d" ] || continue
    [ "$(cd "$d" && pwd -P)" = "$cur" ] && continue
    n=$((n + 1))
    if [ $YES = 1 ]; then echo "removing $d"; rm -rf "$d"; else echo "would remove $d (pass --yes)"; fi
  done
  [ $n -gt 0 ] || echo "nothing to prune under $(line_dir)"
}

case "${1:-}" in
  setup) shift; do_setup "$@" ;;
  refresh) shift; do_setup --latest "$@" ;;
  build) shift; do_build "$@" ;;
  status) shift; parse_args "$@"; do_status ;;
  env) shift; parse_args "$@"; do_env ;;
  prune) shift; do_prune "$@" ;;
  resolve-develop) resolve_develop ;;
  recipe) echo "$(recipe_id)" ;;
  *) sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
