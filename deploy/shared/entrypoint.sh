#!/bin/bash
set -e

# Wait for data directory
mkdir -p /app/data /app/logs

# Set Python path
export PYTHONPATH=/app

# Run the specified command
exec "$@"