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