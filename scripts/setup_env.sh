#!/bin/bash
set -e

VENV_PATH="${UV_PROJECT_ENVIRONMENT:-/home/francesco/.venvs/ppcx-domains}"
PYTHON_VERSION="${1:-3.12}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Setting up ppcx-domains environment ==="
echo "  Venv path:      $VENV_PATH"
echo "  Python version: $PYTHON_VERSION"

# 1. Remove old environment if it exists
if [ -d "$VENV_PATH" ]; then
    echo "Removing existing environment..."
    rm -rf "$VENV_PATH"
fi

# 2. Create new environment
echo "Creating virtual environment..."
uv venv "$VENV_PATH" --python "$PYTHON_VERSION"

# 3. Set UV_PROJECT_ENVIRONMENT for this session
export UV_PROJECT_ENVIRONMENT="$VENV_PATH"

# 4. Install dependencies with CUDA extras
echo "Installing dependencies..."
cd "$PROJECT_DIR"
uv sync --extra cuda

# 5. Copy sitecustomize.py into the venv
SITE_PACKAGES="$VENV_PATH/lib/python${PYTHON_VERSION}/site-packages"
echo "Installing sitecustomize.py to $SITE_PACKAGES..."
cp "$SCRIPT_DIR/sitecustomize.py" "$SITE_PACKAGES/sitecustomize.py"

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $VENV_PATH/bin/activate"
echo "Or run with:   uv run <script.py>"