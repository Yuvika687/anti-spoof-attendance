#!/usr/bin/env bash
# resume_celeba_download.sh
# ---------------------------
# Resumes the CelebA-Spoof zip-part download from Google Drive.
#
# Google Drive enforces a per-file quota on popular shared files
# ("Too many users have viewed or downloaded this file recently"),
# which is unrelated to network speed and can only be waited out
# (observed: clears anywhere from a few hours up to ~24h). gdown's
# --continue flag skips parts that are already fully downloaded, so
# this script is safe to re-run repeatedly until every part succeeds.
#
# Usage:
#   ./scripts/resume_celeba_download.sh
#
# Progress check:
#   ls raw_downloads/CelebA_Spoof_zip_parts/CelebA-Spoof-zips/ | sort -V
#   tail -f raw_downloads/gdown_log.txt

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_DIR="$PROJECT_ROOT/raw_downloads/CelebA_Spoof_zip_parts"
LOG_FILE="$PROJECT_ROOT/raw_downloads/gdown_log.txt"
DRIVE_FOLDER_URL="https://drive.google.com/drive/folders/1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z"

mkdir -p "$DOWNLOAD_DIR"
source "$PROJECT_ROOT/venv/bin/activate"

cd "$DOWNLOAD_DIR"
python3 -m gdown --folder "$DRIVE_FOLDER_URL" --continue --retries 3 >> "$LOG_FILE" 2>&1

echo "Download attempt finished (or was interrupted). Check $LOG_FILE for details."
