# OmniRoam - InteriorGS Rendering Pipeline

A batch rendering tool for rendering panoramic videos from 3D Gaussian Splatting (3DGS) datasets using Blender.

## Prerequisites

### System Requirements

- **Linux** (tested on Ubuntu 20.04+)
- **GPU** with CUDA support (recommended for faster rendering)
- **Python 3.11** (included with Blender 4.5.6)
- **Node.js >= 18.0.0**
- **FFmpeg**

### Required Software

1. **Blender 4.5.6** (Linux x64)
2. **Gaussian Splatting Blender Addon**
3. **splat-transform** (npm package for PLY decompression)
4. **Python packages**: tqdm, scipy, huggingface_hub

## Installation

### Step 1: Install Blender 4.5.6

```bash
wget https://www.blender.org/download/release/Blender4.5/blender-4.5.6-linux-x64.tar.xz

tar -xf blender-4.5.6-linux-x64.tar.xz
```

This will create a `blender-4.5.6-linux-x64` directory containing Blender.

### Step 2: Install Gaussian Splatting Blender Addon

```bash
git clone git@github.com:ReshotAI/gaussian-splatting-blender-addon.git

cd gaussian-splatting-blender-addon
./zip.sh
```

This creates `blender-addon.zip` in the addon directory.

Move the addon to your project root:

```bash
cd ..
mkdir -p gaussian-splatting-blender-addon
cp gaussian-splatting-blender-addon/blender-addon.zip gaussian-splatting-blender-addon/
```

### Step 3: Install Python Dependencies

The script will automatically install required Python packages (tqdm, scipy) to Blender's Python environment on first run. To manually install:

```bash
./blender-4.5.6-linux-x64/4.5/python/bin/python3.11 -m pip install tqdm scipy huggingface_hub
```

### Step 4: Install Node.js and splat-transform

The script will attempt to install Node.js via nvm if not present. To manually install:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install 20
nvm use 20

npm install -g @playcanvas/splat-transform
```

### Step 5: Project Structure

Ensure your project directory has this structure:

```
your-project/
├── blender-4.5.6-linux-x64/
│   └── blender
├── gaussian-splatting-blender-addon/
│   └── blender-addon.zip
├── render.py
├── run_simple.sh
├── valid_datasets.txt
├── datasets/              (will be created automatically)
└── output/                (will be created automatically)
```

## Dataset Download

The dataset files come from **two separate Hugging Face repositories** and require manual setup.

### Dataset Sources

1. **3D Gaussian Splatting (3DGS) Files**: 
   - Repository: `spatialverse/InteriorGS` (private, requires HF token)
   - File: `3dgs_compressed.ply` (will auto-download and decompress)

2. **Camera Path Files**: 
   - Repository: `Yuheng02/OmniRoam-InteriorGS-Path`
   - File: `path_3.zip` (requires manual download and extraction)

### Setup Instructions

#### Step 1: Configure Hugging Face Token

The InteriorGS dataset is private. You need to:

1. Get your Hugging Face token from https://huggingface.co/settings/tokens
2. Edit `render.py` and set your token:
   ```python
   HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxx"  # Your token here
   ```

Or provide it via command line:
```bash
./blender-4.5.6-linux-x64/blender --background --python render.py -- \
    --dataset-file valid_datasets.txt \
    --hf-token hf_xxxxxxxxxxxxxxxxxxxxx
```

#### Step 2: Download Camera Path Files

The camera trajectories must be downloaded manually:

1. **Download from Hugging Face**:
   ```bash
   # Visit: https://huggingface.co/datasets/Yuheng02/OmniRoam-InteriorGS-Path
   # Download path_3.zip
   wget https://huggingface.co/datasets/Yuheng02/OmniRoam-InteriorGS-Path/resolve/main/path_3.zip
   ```

2. **Extract to datasets directory**:
   ```bash
   unzip path_3.zip -d datasets/
   ```

   This will create `datasets/<dataset_id>/path.json` for each dataset.

## Usage

### Run Rendering

#### Single Process Mode

Process all datasets sequentially:

```bash
./blender-4.5.6-linux-x64/blender --background --python render.py -- --dataset-file valid_datasets.txt
```

#### Multi-Process Parallel Mode

Process multiple splits in parallel using the shell script:

```bash
./run_simple.sh 1 2 3 4 5 200
```

This command:
- Processes splits 1, 2, 3, 4, 5 (out of 200 total splits)
- Uses 5 parallel processes
- Each process handles its assigned subset of datasets

#### Manual Python Command

For more control, use the Python script directly:

```bash
./blender-4.5.6-linux-x64/blender --background --python render.py -- \
    --dataset-file valid_datasets.txt \
    --split 1 2 3 \
    --total-split 200
```

### Command Line Options

```
--datasets ID1 ID2 ...        Dataset ID list (space separated)
--dataset-file FILE           Text file with dataset IDs (one per line)
--base-path PATH              Local dataset cache path (default: datasets)
--output-dir PATH             Output directory (default: output)
--target-frames N             Target video frame count (default: 800)
--hf-token TOKEN              Hugging Face token (for private repos)
--split N1 N2 ...             Split numbers to process
--total-split N               Total number of splits
--dry-run                     Validate datasets only, don't render
--help, -h                    Show help message
```

## Output

For each dataset, the rendering pipeline produces:

```
output/<dataset_id>/
├── pano_camera0/
│   ├── frame_0001.png    (PNG image sequence)
│   ├── frame_0002.png
│   ├── ...
│   └── frame_0800.png
├── video.mp4              (Compiled video, 30fps)
└── transforms.json        (Camera positions and transformations)
```

### Output Files

1. **PNG Sequence** (`pano_camera0/frame_*.png`)
   - Format: PNG, RGB, 8-bit
   - Resolution: 2880x1440 (configurable)
   - Compression level: 15
   - Count: 800 frames (default)

2. **Video** (`video.mp4`)
   - Codec: H.264 (libx264) with CRF 18
   - Fallback: libopenh264 or mpeg4
   - Frame rate: 30 fps
   - Pixel format: yuv420p

3. **Camera Data** (`transforms.json`)
   - Camera-to-world transformation matrices (4x4)
   - Camera positions (world coordinates)
   - Camera rotations (Euler angles)
   - Format: OpenCV/Nerfstudio compatible

### transforms.json Structure

```json
{
  "coordinate_convention": "OpenCV: x_cam = R * X_world + t (Rcw/tcw)",
  "twc_convention": "Nerfstudio: X_world -> cam with inverse(Twc); stored transform_matrix = Twc (cam-to-world)",
  "dataset_id": "0001_839920",
  "num_images": 800,
  "per_image": {
    "pano_camera0/frame_0001.png": {
      "transform_matrix": [[...], [...], [...], [...]],
      "location": [x, y, z],
      "rotation": [rx, ry, rz],
      "frame": 1
    },
    ...
  }
}
```

## Configuration

Edit these variables in `render.py` for customization:

```python
dataset_base_path = "datasets"          # Local dataset cache
output_base_dir = "output"              # Output directory
target_frames = 800                     # Video length
fps = 30                                # Frame rate
target_speed = 0.02                     # Camera movement speed (m/frame)
```

Render settings (resolution, samples, etc.) can be modified in `setup_render_settings()`.

## Acknowledgments

- [Blender Foundation](https://www.blender.org/)
- [ReshotAI Gaussian Splatting Addon](https://github.com/ReshotAI/gaussian-splatting-blender-addon)
- [PlayCanvas splat-transform](https://github.com/playcanvas/splat-transform)
- [InteriorGS Dataset](https://huggingface.co/datasets/spatialverse/InteriorGS)
- [OmniRoam-InteriorGS-Path](https://huggingface.co/datasets/Yuheng02/OmniRoam-InteriorGS-Path)

## License
The codebase is released under the Adobe Research License for noncommercial research purposes only.
