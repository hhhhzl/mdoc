# Model-Based Diffusion Optimal Control for Multi-Robot Motion Planning

---

[//]: # (## Introduction)

[//]: # ()
[//]: # (Model-Based Diffusion Optimal Control for Multi-Robot Motion Planning)


## Installation

#### Requirements: 
  - Python >= 3.10
#### Supports:
  - ubuntu == 22.04
  - cuda == 11.8.0 / 12.1.1
  - pytorch == 2.1.0 / 2.2.0
  - MacOS mps chips (Not recommend)
```
./scripts/bash/setup.sh
```

## Main Pipeline
```
/mdoc/planners/multi_agent/cbs.py --> 
/mdoc/planners/single_agent/mdoc_ensemble.py -->
/mdoc/models/diffusion_models/mbd_ensemble.py # Main MDOC Implementation
```

## Baselines
```
Please check mdoc/baselines for all baseline implementation
```


## Reproduce Our Experiments
```
./scripts/run_experiments.sh
```

## Citation
If you use our work or code in your research, please cite our paper:
```
```

## Credits
Parts of this work and software were taken and/or inspired from:



