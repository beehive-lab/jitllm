#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]]; then
    echo "Usage: $0 <jitllm-base-version>" >&2
    exit 2
fi

for command in curl javap sha256sum; do
    command -v "$command" >/dev/null || {
        echo "Required command not found: $command" >&2
        exit 2
    }
done

version=$1
repository_url=${MAVEN_REPOSITORY_URL:-https://repo.maven.apache.org/maven2}
group_path=io/github/beehive-lab
artifact=jitllm
inspection_dir=$(mktemp -d)
trap 'rm -rf "$inspection_dir"' EXIT

for jdk in 21 25; do
    artifact_version="${version}-jdk${jdk}"
    artifact_url="${repository_url}/${group_path}/${artifact}/${artifact_version}"
    jar_path="${inspection_dir}/${artifact}-${artifact_version}.jar"
    pom_path="${inspection_dir}/${artifact}-${artifact_version}.pom"

    curl --fail --silent --show-error --location \
        "${artifact_url}/${artifact}-${artifact_version}.jar" \
        --output "$jar_path"
    curl --fail --silent --show-error --location \
        "${artifact_url}/${artifact}-${artifact_version}.pom" \
        --output "$pom_path"

    major_version=$(javap -verbose -classpath "$jar_path" \
        org.beehive.jitllm.model.Model | awk '/major version:/ { print $3; exit }')
    expected_major=$((44 + jdk))
    if [[ $major_version != "$expected_major" ]]; then
        echo "Unexpected class-file version for ${artifact_version}: ${major_version}; expected ${expected_major}" >&2
        exit 1
    fi

    echo "${artifact_version}: class major ${major_version}"
    sha256sum "$jar_path"
done

echo
echo "ChatFormat API from ${version}-jdk21:"
javap -public -classpath "${inspection_dir}/${artifact}-${version}-jdk21.jar" \
    org.beehive.jitllm.model.format.ChatFormat
