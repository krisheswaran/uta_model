#!/usr/bin/env bash
# Package the data/ directory into a tar.gz for hosting.
# Usage: ./scripts/package_data.sh [output_path]
#
# Default output: uta_model_data.tar.gz in the project root.

set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT="${1:-uta_model_data.tar.gz}"

if [ ! -d "data" ]; then
    echo "Error: data/ directory not found. Run the pipeline first."
    exit 1
fi

echo "Packaging data/ ..."
tar czf "$OUTPUT" data/
SIZE=$(ls -lh "$OUTPUT" | awk '{print $5}')
echo "Created $OUTPUT ($SIZE)"
echo ""
echo "Upload this file to your hosting location, then update the URL in scripts/download_data.sh"
