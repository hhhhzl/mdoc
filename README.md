## Model-Based Diffusion Online Control in Multi-Robot Motion Planning

---

### Introduction

Model-Based Diffusion Online Control in Multi-Robot Motion Planning


### Installation

#### Requirements: 
  - Python >= 3.10
#### Supports:
  - ubuntu == 22.04
  - cuda == 11.8.0
  - pytorch == 2.1.0
  - MacOS mps chips
```
cd scripts/bash
./setup.sh
```
### Planning with MDOC

### Planning with MMD
mmd produces data-driven multi-robot trajectories in a single map or in a collection of "tiled" local maps. Let's take a look at how to use it.
#### Download Trajectory Data and Pre-trained Diffusion Model

```commandline
cd scripts/bach
./download.sh
```
#### Inference with MMD
```commandline
cd scripts/bash
./run_inference.sh --method=mmd
```

### Reproduce Our Experiments


### Citation
If you use our work or code in your research, please cite our paper:
```
@inproceedings{
shaoul2025multirobot,
title={Multi-Robot Motion Planning with Diffusion Models},
author={Yorai Shaoul and Itamar Mishani and Shivam Vats and Jiaoyang Li and Maxim Likhachev},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=AUCYptvAf3}
}
```

### Credits
Parts of this work and software were taken and/or inspired from:



