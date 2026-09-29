#!/usr/bin/env bash
# One-off helper: extracts each Drive "download bundle" zip (each bundle
# contains a handful of the real CelebA_Spoof.zip.0NN part files) into the
# project's target folder, deleting each bundle immediately after a
# successful extraction to avoid holding both copies on disk at once.
set -uo pipefail

BUNDLES_DIR="$HOME/Downloads/Dataset"
TARGET_DIR="$HOME/Documents/project/anti_spoof_attendance/raw_downloads/CelebA_Spoof_zip_parts/CelebA-Spoof-zips"

mkdir -p "$TARGET_DIR"

for bundle in "$BUNDLES_DIR"/drive-download-*.zip; do
    [ -f "$bundle" ] || continue
    echo "Extracting: $(basename "$bundle")"
    if unzip -o -q "$bundle" -d "$TARGET_DIR"; then
        rm -f "$bundle"
        echo "  OK, bundle deleted. Free space now: $(df -h "$HOME" | tail -1 | awk '{print $4}')"
    else
        echo "  FAILED to extract $(basename "$bundle") -- leaving it in place for inspection"
    fi
done

echo "All bundles processed."
