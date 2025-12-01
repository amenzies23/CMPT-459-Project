#!/bin/bash
echo "Cleaning up generated files and caches..."

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +

# Remove model and feature files
rm -rf ./model/*.pkl 2>/dev/null

# Remove generated plots
find . -type f -name "*.png" -delete

# Remove other temporary cache files
find . -type f -name "*.log" -delete
find . -type f -name "*.tmp" -delete

echo "Cleanup complete."