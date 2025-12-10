#!/usr/bin/env bash
set -euo pipefail

# Helper script to trigger the "Publish to PyPI" GitHub Actions workflow
# Usage:
#   ./scripts/trigger_publish.sh [branch]
# Example:
#   ./scripts/trigger_publish.sh main

REPO="AliMehdi512/ilovetools"
WORKFLOW_FILE="publish-to-pypi.yml"
BRANCH=${1:-main}

# Check gh is installed
if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI not found. Install GitHub CLI and run 'gh auth login' first." >&2
  exit 2
fi

# Check auth
if ! gh auth status --hostname github.com >/dev/null 2>&1; then
  echo "You are not authenticated with gh. Run: gh auth login" >&2
  exit 3
fi

echo "Triggering workflow $WORKFLOW_FILE on repo $REPO (branch: $BRANCH)"

# Trigger workflow dispatch
gh workflow run "$WORKFLOW_FILE" --repo "$REPO" --ref "$BRANCH"

# Get the latest run id for that workflow
sleep 2
RUN_ID=$(gh run list --repo "$REPO" --workflow "$WORKFLOW_FILE" --limit 1 --json databaseId --jq '.[0].databaseId')

if [ -n "$RUN_ID" ]; then
  echo "Dispatched run ID: $RUN_ID"
  echo "You can watch logs with: gh run watch $RUN_ID --repo $REPO"
else
  echo "Unable to determine run id; check the Actions tab in GitHub UI." >&2
fi
