<p align="center">
<img src="https://yuheng.ink/project-page/omniroam/images/clarification.png"></img>
<h1 align="center">Self Forcing</h1>
<h3 align="center">Bridging the Train-Test Gap in Autoregressive Video Diffusion</h3>
</p>
<p align="center">
  <p align="center">
    <a href="https://www.xunhuang.me/">Xun Huang</a><sup>1</sup>
    ·
    <a href="https://zhengqili.github.io/">Zhengqi Li</a><sup>1</sup>
    ·
    <a href="https://guandehe.github.io/">Guande He</a><sup>2</sup>
    ·
    <a href="https://mingyuanzhou.github.io/">Mingyuan Zhou</a><sup>2</sup>
    ·
    <a href="https://research.adobe.com/person/eli-shechtman/">Eli Shechtman</a><sup>1</sup><br>
    <sup>1</sup>Adobe Research <sup>2</sup>UT Austin
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2506.08009">Paper</a> | <a href="https://self-forcing.github.io">Website</a> | <a href="https://huggingface.co/gdhe17/Self-Forcing/tree/main">Models (Original)</a> | <a href="https://huggingface.co/Yuheng02/OmniRoam/tree/main/Self-forcing">Models (OmniRoam)</a></h3>
</p>

---

Self Forcing trains autoregressive video diffusion models by **simulating the inference process during training**, performing autoregressive rollout with KV caching. It resolves the train-test distribution mismatch and enables **real-time, streaming video generation on a single RTX 4090** while matching the quality of state-of-the-art diffusion models.

---


https://github.com/user-attachments/assets/d3c16bd8-fa30-4d89-aeda-3ea3a7758c23


## Requirements
We tested this repo on the following setup:
* Nvidia GPU (A100, and H100 are tested).
* Linux operating system.
* 64 GB RAM.

Other hardware setup could also work but hasn't been tested.

## Installation
We assume the `omniroam` environment has already been set up by following the main OmniRoam installation instructions, install the additional dependencies for Self-Forcing with:

```
conda activate omniroam
cd Self-Forcing
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

## Quick Start
### Link base model
We assume the base `Wan2.1-T2V-1.3B` model has already been downloaded during the main OmniRoam setup, so only a symbolic link is needed here.

```
mkdir -p wan_models
ln -s ../models/Wan-AI/Wan2.1-T2V-1.3B wan_models/Wan2.1-T2V-1.3B
```

### CLI Inference
For OmniRoam inference, you can directly run the provided script after setting the checkpoint path in `inference_local_panoramas.sh`:
```
./inference_local_panoramas.sh
```
Other config files and corresponding checkpoints can be found in [configs](configs) folder.


## Acknowledgements
`Self-Forcing`:
This codebase is built on top of the open-source implementation of [CausVid](https://github.com/tianweiy/CausVid) by [Tianwei Yin](https://tianweiy.github.io/) and the [Wan2.1](https://github.com/Wan-Video/Wan2.1) repo.

`OmniRoam`:
We gratefully acknowledge the authors of [Self-Forcing](https://github.com/guandeh17/Self-Forcing) for their excellent work, upon which the OmniRoam Self-Forcing variant is built.

## Citation
If you find these modifications useful for your research, please kindly cite papers below:
```bibtex
@article{huang2025selfforcing,
  title={Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion},
  author={Huang, Xun and Li, Zhengqi and He, Guande and Zhou, Mingyuan and Shechtman, Eli},
  journal={arXiv preprint arXiv:2506.08009},
  year={2025}
}
```

```bibtex
@article{omniroam2026,
  title={OmniRoam: World Wandering via Long-Horizon Panoramic Video Generation},
  author={Yuheng Liu and Xin Lin and Xinke Li and Baihan Yang and Chen Wang and Kalyan Sunkavalli and Yannick Hold-Geoffroy and Hao Tan and Kai Zhang and Xiaohui Xie and Zifan Shi and Yiwei Hu},
  journal={SIGGRAPH},
  year={2026}
}
```
