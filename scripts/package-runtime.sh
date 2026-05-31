#!/bin/sh
set -eu

if [ $# -ne 3 ]; then
  echo "Usage: $0 <source-runtime-dir> <llamacpp-version> <backend>" >&2
  exit 1
fi

source_dir=$1
llamacpp_version=$2
backend=$3

case "$backend" in
  cuda|rocm|cpu) ;;
  *)
    echo "Error: backend must be one of: cuda, rocm, cpu" >&2
    exit 1
    ;;
esac

if ! command -v patchelf >/dev/null 2>&1; then
  echo "Error: patchelf is required to package a relocatable runtime" >&2
  exit 1
fi

case "$(uname -m)" in
  x86_64|amd64)
    arch=amd64
    ;;
  aarch64|arm64)
    arch=arm64
    ;;
  *)
    echo "Error: unsupported architecture: $(uname -m)" >&2
    exit 1
    ;;
esac

if [ ! -d "$source_dir" ]; then
  echo "Error: source runtime dir does not exist: $source_dir" >&2
  exit 1
fi

if [ ! -f "$source_dir/llama-server" ]; then
  echo "Error: source runtime is missing llama-server: $source_dir" >&2
  exit 1
fi

if ! find "$source_dir" -maxdepth 1 -type f -name 'libggml*.so*' | grep -q .; then
  echo "Error: source runtime must contain at least one libggml*.so file: $source_dir" >&2
  exit 1
fi

case "$llamacpp_version" in
  llama-server-*)
    llamacpp_version=${llamacpp_version#llama-server-}
    ;;
esac

staging_dir=$(mktemp -d "${TMPDIR:-/tmp}/anvil-runtime.XXXXXX")
cleanup() {
  rm -rf "$staging_dir"
}
trap cleanup EXIT HUP INT TERM

copy_runtime_file() {
  src=$1
  cp -a "$src" "$staging_dir/"
}

copy_runtime_file "$source_dir/llama-server"
for path in "$source_dir"/*.so*; do
  [ -e "$path" ] || continue
  copy_runtime_file "$path"
done

printf '%s\n' "$backend" > "$staging_dir/backend"
cat > "$staging_dir/anvil-runtime.json" <<EOF
{"llamacpp_version":"$llamacpp_version","backend":"$backend","os":"linux","arch":"$arch"}
EOF

find "$staging_dir" -maxdepth 1 -type f \( -name 'llama-server' -o -name '*.so*' \) -exec patchelf --set-rpath '$ORIGIN' {} +

tarball="llama-server-${llamacpp_version}-linux-${arch}-${backend}.tar.gz"
tarball_path="$(pwd)/$tarball"

(
  cd "$staging_dir"
  tar czf "$tarball_path" ./*
)

sha256sum "$tarball_path" > "$tarball_path.sha256"
sha256=$(sha256sum "$tarball_path" | awk '{print $1}')

printf '%s\n%s\n' "$tarball_path" "$sha256"
