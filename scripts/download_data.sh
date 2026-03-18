#!/usr/bin/env bash
# Download and extract the pre-computed data/ directory.
# Usage: ./scripts/download_data.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# TODO: Replace with your hosted URL
DATA_URL="${UTA_DATA_URL:-https://storage.googleapis.com/uta_model_storage/uta_model_data.tar.gz}"

if [ "$DATA_URL" = "https://REPLACE_ME/uta_model_data.tar.gz" ]; then
    echo "Error: Set the download URL."
    echo "  Either edit DATA_URL in this script, or set the UTA_DATA_URL environment variable."
    exit 1
fi

if [ -d "data/parsed" ] && [ "$(ls -A data/parsed 2>/dev/null)" ]; then
    echo "data/ already exists and contains files."
    read -p "Overwrite? [y/N] " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

echo "Downloading data from $DATA_URL ..."
curl -fSL "$DATA_URL" -o /tmp/uta_model_data.tar.gz

echo "Extracting to data/ ..."
tar xzf /tmp/uta_model_data.tar.gz

rm /tmp/uta_model_data.tar.gz
echo "Done. data/ is ready."
