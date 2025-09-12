#!/usr/bin/env bash
set -euo pipefail

# Defaults (can be overridden by flags)
HF_USERNAME_DEFAULT="mkieffer"
HF_REPO_NAME_DEFAULT="Medbullets"
PRIVATE_DEFAULT="false"

HF_USERNAME="$HF_USERNAME_DEFAULT"
HF_REPO_NAME="$HF_REPO_NAME_DEFAULT"
PRIVATE="$PRIVATE_DEFAULT"

usage() {
  cat <<'EOF'
Usage: ./run.sh [--username <HF_USERNAME>] [--repo <HF_REPO_NAME>] [--private <true|false>]

Examples:
  ./run.sh
  ./run.sh --username mkieffer --repo Medbullets --private true
EOF
}

# simple long-flag parser
while [[ $# -gt 0 ]]; do
  case "$1" in
    --username)
      HF_USERNAME="${2:-}"; shift 2 ;;
    --repo|--reponame|--repo-name)
      HF_REPO_NAME="${2:-}"; shift 2 ;;
    --private)
      PRIVATE="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage; exit 1 ;;
  esac
done

# Normalize PRIVATE to lowercase
PRIVATE="$(echo "$PRIVATE" | tr '[:upper:]' '[:lower:]')"
if [[ "$PRIVATE" != "true" && "$PRIVATE" != "false" ]]; then
  echo "Error: --private must be 'true' or 'false' (got '$PRIVATE')" >&2
  exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

# List of files to download
files=("medbullets_op4.csv" "medbullets_op4.json" "medbullets_op5.csv" "medbullets_op5.json")

# Download the files if not already present
for file in "${files[@]}"; do
  if [ ! -f "data/$file" ]; then
    echo "Downloading $file..."
    curl -L -o "data/$file" "https://raw.githubusercontent.com/HanjieChen/ChallengeClinicalQA/main/medbullets/$file"
  else
    echo "$file already exists, skipping download."
  fi
done

echo "Downloads complete."

# Run the uploader, forwarding the args
python3 upload_to_hf.py \
  --hf_username "$HF_USERNAME" \
  --hf_repo_name "$HF_REPO_NAME" \
  --private "$PRIVATE"
