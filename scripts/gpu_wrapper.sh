#!/bin/bash
# GPU training wrapper script
# This script clears LD_LIBRARY_PATH to avoid cuDNN version conflicts
# and ensures PyTorch uses its bundled cuDNN libraries

# Clear LD_LIBRARY_PATH to avoid cuDNN version conflicts
unset LD_LIBRARY_PATH

# Execute the command with the fixed environment
exec "$@"

