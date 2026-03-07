#!/usr/bin/env bash
#
# kiro-login.sh — Log in with kiro-cli and copy the SQLite DB to the right account file.
#
# Usage:
#   ./scripts/kiro-login.sh primary    # Log in with the primary (opus/sonnet) account
#   ./scripts/kiro-login.sh haiku      # Log in with the haiku account
#

set -euo pipefail

KIRO_CLI_DB="$HOME/.local/share/kiro-cli/data.sqlite3"

usage() {
    echo "Usage: $0 <primary|haiku>"
    echo ""
    echo "  primary  — Log in with the primary account (opus, sonnet, auto)"
    echo "  haiku    — Log in with the haiku account"
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
    haiku)
        TARGET="$HOME/.local/share/kiro-cli/data-haiku.sqlite3"
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
