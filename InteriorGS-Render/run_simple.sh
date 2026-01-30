#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Error: at least two arguments required"
    echo "Usage: $0 <split1> [split2] [split3] ... <total_split>"
    echo "  split1, split2, ...: Split numbers (starting from 1)"
    echo "  total_split: Total number of splits (last argument)"
    echo ""
    echo "Example: $0 1 2 3 4 5 200"
    exit 1
fi

TOTAL_SPLIT="${@: -1}"

if ! [[ "$TOTAL_SPLIT" =~ ^[0-9]+$ ]]; then
    echo "Error: total_split (last argument) must be a positive integer"
    exit 1
fi

SPLITS=("${@:1:$#-1}")

SPLIT_LIST=""
for SPLIT in "${SPLITS[@]}"; do
    if ! [[ "$SPLIT" =~ ^[0-9]+$ ]]; then
        echo "Error: split argument must be a positive integer, got: $SPLIT"
        exit 1
    fi
    
    if [ "$SPLIT" -lt 1 ] || [ "$SPLIT" -gt "$TOTAL_SPLIT" ]; then
        echo "Error: split $SPLIT must be between 1 and $TOTAL_SPLIT"
        exit 1
    fi
    
    if [ -z "$SPLIT_LIST" ]; then
        SPLIT_LIST="$SPLIT"
    else
        SPLIT_LIST="$SPLIT_LIST, $SPLIT"
    fi
done

echo "============================================"
echo "Blender 3DGS Rendering - Multi-process Mode"
echo "Processing splits: [$SPLIT_LIST]"
echo "Total splits: $TOTAL_SPLIT"
echo "Will use ${#SPLITS[@]} processes in parallel"
echo "============================================"

BLENDER_PATH="blender-4.5.6-linux-x64/blender"
SCRIPT_PATH="render_clean.py"
DATASETS_FILE="valid_datasets.txt"

if [ ! -f "$BLENDER_PATH" ]; then
    echo "Error: Blender not found: $BLENDER_PATH"
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Render script not found: $SCRIPT_PATH"
    exit 1
fi

if [ ! -f "$DATASETS_FILE" ]; then
    echo "Error: Dataset list not found: $DATASETS_FILE"
    exit 1
fi

echo "Checking Blender Python dependencies..."
BLENDER_PYTHON="${BLENDER_PATH%/blender}/4.5/python/bin/python3.11"

if [ -f "$BLENDER_PYTHON" ]; then
    echo "Found Blender Python: $BLENDER_PYTHON"
    
    if ! "$BLENDER_PYTHON" -c "import tqdm" 2>/dev/null; then
        echo "Installing tqdm to Blender Python environment..."
        "$BLENDER_PYTHON" -m pip install --upgrade pip --quiet
        "$BLENDER_PYTHON" -m pip install tqdm --quiet
        echo "tqdm installed"
    else
        echo "tqdm already installed"
    fi
    
    if ! "$BLENDER_PYTHON" -c "import scipy" 2>/dev/null; then
        echo "Installing scipy to Blender Python environment..."
        "$BLENDER_PYTHON" -m pip install scipy --quiet
        echo "scipy installed"
    else
        echo "scipy already installed"
    fi
else
    echo "Warning: Blender Python not found, skipping dependency check"
fi
echo ""

echo "Checking Node.js and npm..."

REQUIRED_NODE_VERSION=18
CURRENT_NODE_VERSION=0

if command -v node &> /dev/null; then
    CURRENT_NODE_VERSION=$(node --version | sed 's/v\([0-9]*\).*/\1/')
fi

if ! command -v npm &> /dev/null || [ "$CURRENT_NODE_VERSION" -lt "$REQUIRED_NODE_VERSION" ]; then
    if [ "$CURRENT_NODE_VERSION" -gt 0 ] && [ "$CURRENT_NODE_VERSION" -lt "$REQUIRED_NODE_VERSION" ]; then
        echo "Current Node.js version (v$CURRENT_NODE_VERSION) is outdated, need >= v18.0.0"
    else
        echo "Node.js not found, installing..."
    fi
    
    echo "Installing Node.js 20.x LTS using nvm..."
    
    export NVM_DIR="$HOME/.nvm"
    if [ ! -d "$NVM_DIR" ]; then
        echo "Installing nvm..."
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
        
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
        [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
    else
        echo "nvm already installed"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
        [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
    fi
    
    if ! command -v nvm &> /dev/null; then
        if [ -s "/usr/local/share/nvm/nvm.sh" ]; then
            . /usr/local/share/nvm/nvm.sh
        fi
    fi
    
    echo "Installing Node.js 20 LTS via nvm..."
    nvm install 20
    nvm use 20
    nvm alias default 20
    
    if command -v node &> /dev/null && command -v npm &> /dev/null; then
        NODE_VERSION=$(node --version 2>/dev/null || echo "unknown")
        NPM_VERSION=$(npm --version 2>/dev/null || echo "unknown")
        CURRENT_NODE_MAJOR=$(echo $NODE_VERSION | sed 's/v\([0-9]*\).*/\1/')
        
        if [ "$CURRENT_NODE_MAJOR" -ge "$REQUIRED_NODE_VERSION" ]; then
            echo "Node.js and npm installed successfully"
            echo "  Node.js version: $NODE_VERSION"
            echo "  npm version: $NPM_VERSION"
        else
            echo "Error: Node.js version still outdated ($NODE_VERSION)"
            exit 1
        fi
    else
        echo "Error: Node.js installation failed"
        echo "Please install manually:"
        echo "  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash"
        echo "  source ~/.bashrc"
        echo "  nvm install 20"
        exit 1
    fi
else
    NODE_VERSION=$(node --version 2>/dev/null || echo "unknown")
    NPM_VERSION=$(npm --version 2>/dev/null || echo "unknown")
    echo "Node.js and npm already installed and version meets requirements"
    echo "  Node.js version: $NODE_VERSION"
    echo "  npm version: $NPM_VERSION"
fi
echo ""

echo "Checking splat-transform..."
if ! command -v splat-transform &> /dev/null; then
    echo "Installing splat-transform..."
    
    if npm install -g @playcanvas/splat-transform 2>/dev/null; then
        echo "splat-transform installed successfully"
    elif sudo npm install -g @playcanvas/splat-transform; then
        echo "splat-transform installed successfully (with sudo)"
    else
        echo "Error: splat-transform installation failed"
        exit 1
    fi
else
    echo "splat-transform already installed"
fi
echo ""

echo ""
echo "Starting render..."
echo ""

SPLIT_ARGS="--split"
for SPLIT in "${SPLITS[@]}"; do
    SPLIT_ARGS="$SPLIT_ARGS $SPLIT"
done

"$BLENDER_PATH" --background --python "$SCRIPT_PATH" -- \
    --dataset-file "$DATASETS_FILE" \
    $SPLIT_ARGS \
    --total-split "$TOTAL_SPLIT"

EXIT_CODE=$?

echo ""
echo "============================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Render complete!"
else
    echo "Render failed, exit code: $EXIT_CODE"
fi
echo "============================================"

exit $EXIT_CODE


