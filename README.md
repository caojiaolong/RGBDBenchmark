# RGBD Benchmark

This repository contains various RGBD models and aims to provide a benchmark for evaluating their FLOPs, MACs, and the number of parameters. We will continue to add more functionalities in the future.

Our goal is to assist you in comparing these models and selecting the most suitable one for your task. If you have any suggestions or encounter any bugs, please don't hesitate to open an issue or submit a pull request.

## Benchmark

| Model           | Parameters   | FLOPs (NYUDepth v2)   | FLOPs (SUNRGBD)   |
|:----------------|:-------------|:----------------------|:------------------|
| ACNet           | 116.628 M    | 232.021 GFLOPS        | 302.24 GFLOPS     |
| AsymFormer      | 33.0596 M    | 78.9987 GFLOPS        | 105.62 GFLOPS     |
| CEN_101         | 118.202 M    | 1.2366 TFLOPS         | 1.5797 TFLOPS     |
| CEN_152         | 133.891 M    | 1.3277 TFLOPS         | 1.698 TFLOPS      |
| CMNeXt-B4       | 116.576 M    | 263.791 GFLOPS        | 340.623 GFLOPS    |
| CMX-B2          | 66.5809 M    | 134.284 GFLOPS        | 172.347 GFLOPS    |
| CMX-B4          | 139.874 M    | 268.904 GFLOPS        | 347.444 GFLOPS    |
| CMX-B5          | 181.074 M    | 336.078 GFLOPS        | 435.174 GFLOPS    |
| EMSANet         | 46.9354 M    | 90.551 GFLOPS         | 117.814 GFLOPS    |
| ESANet          | 46.9558 M    | 96.6675 GFLOPS        | 96.5128 GFLOPS    |
| FRNet           | 87.8068 M    | 219.392 GFLOPS        | 285.739 GFLOPS    |
| GeminiFusion-B3 | 74.5513 M    | 277.074 GFLOPS        | 358.82 GFLOPS     |
| GeminiFusion-B5 | 137.209 M    | 513.239 GFLOPS        | 666.333 GFLOPS    |
| MultiMAE        | 95.3895 M    | 808.41 GFLOPS         | 808.41 GFLOPS     |
| omnivore_swinB  | 90.1146 M    | 199.31 GFLOPS         |                   |
| omnivore_swinS  | 50.6036 M    | 112.296 GFLOPS        |                   |
| omnivore_swinT  | 28.945 M     | 57.4253 GFLOPS        |                   |
| PGDENet         | 107.403 M    | 326.558 GFLOPS        | 425.406 GFLOPS    |
| SA-Gate         | 110.875 M    | 386.007 GFLOPS        | 498.509 GFLOPS    |
| ShapeConv       | 106.792 M    | 337.861 GFLOPS        | 436.967 GFLOPS    |
| TokenFusion-B2  | 26.0184 M    | 110.422 GFLOPS        | 142.041 GFLOPS    |
| TokenFusion-B3  | 45.9173 M    | 188.751 GFLOPS        | 244.13 GFLOPS     |

## Installation

To meet most requirements, you can simply install the requirements of [DFormer](https://github.com/VCIP-RGBD/DFormer?tab=readme-ov-file#2--get-start). However, please note that to successfully build some models, it is required to **install additional packages or overwrite some files**. Also, jupyter notebook is required to run the benchmark and pandas is used for generate tabulate. You can install it by running `pip install notebook pandas`.

First of all, enter this repository's directory, then run the following commands, or just run `bash repair_models.sh`:

```bash
# For EMSANet, refer to https://github.com/TUI-NICR/nicr-multitask-scene-analysis
python -m pip install "git+https://github.com/TUI-NICR/nicr-scene-analysis-datasets.git@v0.7.0"
python -m pip install "git+https://github.com/TUI-NICR/nicr-multitask-scene-analysis.git"

# For ESANet
pip install tensorflow
pip install pytorch-ignite

# For FRNet, overwrite the broken files in those directories with my repaired files
cp repair_files/FRNet/FRNet.py repositories/FRNet/toolbox/models/FRNet/FRNet.py
cp repair_files/FRNet/__init__.py repositories/FRNet/toolbox/__init__.py

# For GeminiFusion, overwrite some files to update mmcv to higer version and remove pretrained requirement
cp repair_files/GeminiFusion/checkpoint.py repositories/GeminiFusion/mmcv_custom/checkpoint.py
cp repair_files/GeminiFusion/segformer.py repositories/GeminiFusion/models/segformer.py
cp repair_files/GeminiFusion/swin_transformer.py repositories/GeminiFusion/models/swin_transformer.py

# For MultiMAE, overwrite native_scaler to update torch to higer version since torch._six is removed
pip install wandb
cp repair_files/MultiMAE/native_scaler.py repositories/MultiMAE/utils/native_scaler.py

# For omnivore
pip install hydra-core

# For PGDENet,
cp repair_files/PGDENet/BBSnet.py repositories/PGDENet/toolbox/models/BBSnetmodel/BBSnet.py
cp repair_files/PGDENet/__init__.py repositories/PGDENet/toolbox/__init__.py

# For SA-Gate,
cp repair_files/SAGate/config.py repositories/SAGate/model/SA-Gate.nyu/config.py

# For ShapeConv,
cp repair_files/ShapeConv/resnet.py repositories/ShapeConv/rgbd_seg/models/encoders/backbones/resnet.py
cp repair_files/ShapeConv/shape_conv.py repositories/ShapeConv/rgbd_seg/models/utils/shape_conv.py
```

## Usage

Open `benchmark.ipynb` and run all the cells to get the benchmark results.

## Acknowledgment

Core computing fuctions are based on [calflops](https://github.com/MrYxJ/calculate-flops.pytorch). Thanks for their great work!
