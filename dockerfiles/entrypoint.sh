#!/bin/bash
set -e

echo "=== Pulling data from DVC remote ==="
dvc config core.no_scm true
dvc pull

echo "=== Starting training ==="
exec python src/customer_support/train.py "$@"
