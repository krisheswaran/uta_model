#!/usr/bin/env bash
# Upload the synced data to Cloudflare R2.
#
# Prerequisites:
#   1. npm run sync  (copies data from ../data/ to public/data/)
#   2. Set bucket_name in wrangler.jsonc
#   3. Authenticate with: npx wrangler login
#
# Usage:
#   ./scripts/upload-to-r2.sh

set -euo pipefail

cd "$(dirname "$0")/.."

DATA_DIR="public/data"
BUCKET_NAME="uta-data"

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: $DATA_DIR not found. Run 'npm run sync' first."
    exit 1
fi

echo "Uploading to R2 bucket: $BUCKET_NAME"
echo ""

TOTAL=0
FAILED=0

find "$DATA_DIR" -type f \( -name "*.json" -o -name "*.npy" -o -name "*.pkl" \) | sort | while read -r filepath; do
    # Strip "public/data/" prefix to get the R2 object key
    key="${filepath#public/data/}"

    # Set content type based on extension
    case "$filepath" in
        *.json) ct="application/json" ;;
        *.npy)  ct="application/octet-stream" ;;
        *.pkl)  ct="application/octet-stream" ;;
        *)      ct="application/octet-stream" ;;
    esac

    echo -n "  $key ... "
    if npx wrangler r2 object put "$BUCKET_NAME/$key" \
        --file="$filepath" \
        --content-type="$ct" \
        --remote \
        2>/dev/null; then
        echo "ok"
    else
        echo "FAILED"
        FAILED=$((FAILED + 1))
    fi
    TOTAL=$((TOTAL + 1))
done

echo ""
if [ "$FAILED" -gt 0 ]; then
    echo "Done with $FAILED failures out of $TOTAL files."
    echo "Make sure you're logged in: npx wrangler login"
    exit 1
else
    echo "Done. All files uploaded to R2."
    echo "Deploy the worker with: npm run deploy"
fi
