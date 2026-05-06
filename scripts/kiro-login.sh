#!/usr/bin/env bash
#
# kiro-login.sh — Log in with kiro-cli and copy the SQLite DB to the right account file.
#
# Usage:
#   ./scripts/kiro-login.sh primary      # Log in with the primary account (opus)
#   ./scripts/kiro-login.sh secondary    # Log in with the secondary account (sonnet, haiku, etc.)
#

set -euo pipefail

KIRO_CLI_DB="$HOME/.local/share/kiro-cli/data.sqlite3"

usage() {
    echo "Usage: $0 <primary|secondary>"
    echo ""
    echo "  primary    — Log in with the primary account (opus)"
    echo "  secondary  — Log in with the secondary account (sonnet, haiku, auto, etc.)"
    exit 1
}

if [[ $# -ne 1 ]]; then
    usage
fi

ACCOUNT="$1"

case "$ACCOUNT" in
    primary)
        TARGET="$HOME/.local/share/kiro-cli/data-primary.sqlite3"
        ;;
    secondary)
        TARGET="$HOME/.local/share/kiro-cli/data-secondary.sqlite3"
        ;;
    *)
        echo "Error: unknown account '$ACCOUNT'"
        usage
        ;;
esac

echo "==> Logging in for the $ACCOUNT account..."
echo "    Please complete the login in your browser."
echo ""

kiro-cli login

if [[ ! -f "$KIRO_CLI_DB" ]]; then
    echo "Error: kiro-cli database not found at $KIRO_CLI_DB"
    echo "Did the login succeed?"
    exit 1
fi

cp "$KIRO_CLI_DB" "$TARGET"
echo ""
echo "==> Done! Copied credentials to $TARGET"
