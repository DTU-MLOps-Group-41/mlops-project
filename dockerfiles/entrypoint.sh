#!/bin/bash
set -e

echo "=== Pulling data from DVC remote ==="
dvc pull

echo "=== Starting training ==="
exec python src/customer_support/train.py "$@"
