#!/usr/bin/env bash
# One-off: reassembles the 74 split CelebA-Spoof zip parts into one archive,
# verifies its integrity, extracts it, then frees the intermediate files to
# keep disk usage manageable.
set -euo pipefail

PARTS_DIR="$HOME/Documents/project/anti_spoof_attendance/raw_downloads/CelebA_Spoof_zip_parts/CelebA-Spoof-zips"
COMBINED="$HOME/Documents/project/anti_spoof_attendance/raw_downloads/CelebA_Spoof_combined.zip"
EXTRACT_DIR="$HOME/Documents/project/anti_spoof_attendance/raw_downloads/CelebA_Spoof"

echo "[1/5] Concatenating 74 parts into combined zip..."
cat "$PARTS_DIR"/CelebA_Spoof.zip.0* > "$COMBINED"
echo "  combined size: $(du -sh "$COMBINED" | cut -f1)"

echo "[2/5] Verifying combined size matches sum of parts..."
PARTS_TOTAL=$(du -ck "$PARTS_DIR"/CelebA_Spoof.zip.0* | tail -1 | cut -f1)
COMBINED_SIZE=$(du -k "$COMBINED" | cut -f1)
echo "  parts total: ${PARTS_TOTAL}K, combined: ${COMBINED_SIZE}K"

echo "[3/5] Testing zip integrity (this reads the whole archive, may take a while)..."
if unzip -tq "$COMBINED" > /dev/null; then
    echo "  Zip integrity OK."
else
    echo "  ZIP INTEGRITY CHECK FAILED. Aborting before deleting anything."
    exit 1
fi

echo "[4/5] Integrity confirmed -- deleting the 74 individual parts to free space..."
rm -f "$PARTS_DIR"/CelebA_Spoof.zip.0*
echo "  Freed. Disk now: $(df -h "$HOME" | tail -1 | awk '{print $4}') available"

echo "[5/5] Extracting combined zip to $EXTRACT_DIR ..."
mkdir -p "$EXTRACT_DIR"
unzip -q "$COMBINED" -d "$EXTRACT_DIR"
echo "  Extraction complete."

echo "Cleaning up combined zip..."
rm -f "$COMBINED"

echo "DONE. Dataset extracted at: $EXTRACT_DIR"
find "$EXTRACT_DIR" -maxdepth 3 -type d | head -20
