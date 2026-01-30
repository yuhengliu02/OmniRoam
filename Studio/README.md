![studio display](https://yuheng.ink/project-page/omniroam/images/banner-studio-white.png)

# OmniRoam Studio

An interactive web-based system for generating panoramic videos using OmniRoam models. This system provides a user-friendly interface for **Preview**, **Self-forcing**, and **Refine** stages of video generation.



https://github.com/user-attachments/assets/8345e5b2-7dd7-4c84-8431-0ea50fbe5809



## Features

- 🎬 **Multi-Stage Generation Pipeline**
  - **Preview**: Preview generation from panoramic images
  - **Self-forcing**: Fast preview generation
  - **Refine**: Video refinement
  
- 🖼️ **Interactive Web Interface**
  - Real-time generation monitoring with progress tracking
  - Panoramic video player with 360° viewing
  - Visual trajectory presets (forward, backward, left, right, s-curve, loop)
  - Built-in gallery with download support
  
- ⚙️ **Advanced Controls**
  - Multiple trajectory presets and scale control

![studio display](https://yuheng.ink/project-page/omniroam/images/studio-display.png)

## Architecture

```
Studio/
├── frontend/           # Vue.js web interface
│   ├── src/
│   │   ├── components/
│   │   └── stores/
│   └── package.json
├── inference/          # Model inference modules
│   ├── preview.py
│   ├── self_forcing.py
│   └── refine.py
├── app_utils/          # Utility functions
├── main.py             # FastAPI backend server
├── models.py           # Data models
├── state.py            # Runtime state management
└── outputs/            # Generated videos (auto-created)
```

## Prerequisites

### System Requirements

- **Linux** (tested on Ubuntu 20.04+)
- **GPU** with CUDA support (≥12GB VRAM recommended)
- **Python 3.9+**
- **Node.js ≥18.0 and npm** (for frontend)
- **FFmpeg** (for video processing)

## Installation

### Step 1: Clone Repository

```bash
git clone git@github.com:yuhengliu02/OmniRoam.git

cd OmniRoam/Studio
```

Ensure your directory structure looks like:
```
OmniRoam/
├── Studio/              # This project
├── models/              # Model checkpoints (see Step 2)
├── Self-Forcing/        # Training/inference code
└── ...
```

### Step 2: Setup Model Files

**IMPORTANT**: Before running the system, you must ensure all model files are in place.

#### Required Model Structure

```
OmniRoam/models/
├── OmniRoam/
│   ├── Preview/
│   │   └── preview.ckpt                    # Preview stage model
│   ├── Self-forcing/
│   │   └── self-forcing.pt                 # Self-forcing stage model
│   └── Refine/
│       └── refine.ckpt                     # Refine stage model
└── Wan-AI/Wan2.1-T2V-1.3B/                 # Base diffusion model (Wan-AI)
```

### Step 3: Install Backend Dependencies

```bash
# Use OmnirOam conda environment
conda activate omniroam

# Install other dependencies
pip install -r requirements.txt
```

### Step 4: Install Frontend Dependencies

```bash
cd frontend
npm install
```

## Usage

### Starting the System

#### Development Mode (Recommended for Testing)

**Terminal 1 - Backend:**
```bash
cd Studio
conda activate omniroam

python main.py
```

**Terminal 2 - Frontend:**
```bash
cd Studio/frontend
npm run dev
```

## License

The codebase is released under the Adobe Research License for noncommercial research purposes only.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [Vue.js](https://vuejs.org/)
- [A-Frame](https://aframe.io/)
- [Wan-AI](https://huggingface.co/Wan-AI)

## Citation

If you use OmniRoam Studio in your research, please cite:

```bibtex
@article{omniroam2025,
  title={OmniRoam: World Wandering via Long-Horizon Panoramic Video Generation},
  author={},
  journal={arXiv preprint},
  year={2026}
}
```

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: yuhengliu02@gmail.com

