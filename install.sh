#!/bin/sh
set -e

# anvil installer
# Usage: curl -fsSL https://raw.githubusercontent.com/sovereignty-labs/anvil/main/install.sh | sh

REPO="sovereignty-labs/anvil"
BINARY="anvil"

OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$ARCH" in
  x86_64|amd64) ARCH="amd64" ;;
  aarch64|arm64) ARCH="arm64" ;;
  *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

case "$OS" in
  linux) ;;
  darwin) ;;
  *) echo "Unsupported OS: $OS"; exit 1 ;;
esac

ASSET="${BINARY}-${OS}-${ARCH}"
URL="https://github.com/${REPO}/releases/latest/download/${ASSET}"

# Prefer a system path already on PATH so anvil works immediately.
# Use sudo when not root; fall back to a user-local dir (and persist PATH) only if sudo is unavailable.
SUDO=""
if [ "$(id -u)" -eq 0 ]; then
  INSTALL_DIR="/usr/local/bin"
elif command -v sudo >/dev/null 2>&1; then
  INSTALL_DIR="/usr/local/bin"
  SUDO="sudo"
else
  INSTALL_DIR="${HOME}/.local/bin"
fi
$SUDO mkdir -p "$INSTALL_DIR"

echo "Installing anvil..."
echo "  Platform: ${OS}/${ARCH}"
echo "  From:     ${URL}"
echo "  To:       ${INSTALL_DIR}/${BINARY}"

TMP="$(mktemp)"
if command -v curl >/dev/null 2>&1; then
  curl -fsSL "$URL" -o "$TMP"
elif command -v wget >/dev/null 2>&1; then
  wget -q "$URL" -O "$TMP"
else
  echo "Error: curl or wget required"; rm -f "$TMP"; exit 1
fi

$SUDO install -m 0755 "$TMP" "${INSTALL_DIR}/${BINARY}"
rm -f "$TMP"

if "${INSTALL_DIR}/${BINARY}" version >/dev/null 2>&1; then
  echo ""
  "${INSTALL_DIR}/${BINARY}" version
  echo ""
  echo "Installed successfully."
else
  echo ""
  echo "Binary installed but failed to execute. Check your platform."
  exit 1
fi

ON_PATH=0
case ":$PATH:" in *":${INSTALL_DIR}:"*) ON_PATH=1 ;; esac

if [ "$ON_PATH" -eq 0 ]; then
  # Only reached in the user-local fallback (sudo unavailable).
  LINE="export PATH=\"${INSTALL_DIR}:\$PATH\""
  MARKER="# added by anvil installer"
  case "$(basename "${SHELL:-/bin/sh}")" in
    zsh)  RC="${ZDOTDIR:-$HOME}/.zshrc" ;;
    bash) RC="$HOME/.bashrc" ;;
    *)    RC="$HOME/.profile" ;;
  esac
  if ! { [ -f "$RC" ] && grep -qF "$MARKER" "$RC"; }; then
    printf '\n%s\n%s\n' "$MARKER" "$LINE" >> "$RC"
  fi
  echo ""
  echo "${INSTALL_DIR} was added to your PATH in ${RC}."
  echo "Use anvil now in this shell (or open a new terminal):"
  echo "  ${LINE}"
fi

echo ""
echo "Next steps:"
echo "  anvil runtime install         # download llama-server"
echo "  anvil pull org/model:Q4_K_M   # download a model"
echo "  anvil serve &                 # start the daemon"
echo "  anvil load model.gguf         # load and serve"
