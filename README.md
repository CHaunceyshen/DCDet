## DCDet
Dynamic sensing and correlation loss detector for small object detection in remote sensing images
## Installation
```shell
conda create -n dcdet python=3.7 pytorch==1.7.0 cudatoolkit=10.1 torchvision -c pytorch -y
conda activate dcdet
pip install openmim
mim install mmcv-full
mim install mmdet
git clone https://github.com/CHaunceyshen/DCDet.git
cd DCDet
pip install -r requirements/build.txt
pip install -v -e .
```
## Dataset Preparation
Please refer to [data preparation](https://github.com/CHaunceyshen/DCDet/tree/main/tools/data) for dataset preparation.
## Test a model
- single GPU
- single node multiple GPU
- multiple node

You can use the following commands to infer a dataset.
```shell
# single-gpu
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]

# multi-gpu
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [optional arguments]

# multi-node in slurm environment
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments] --launcher slurm
```
Examples:

Inference RotatedRetinaNet on DOTA-1.0 dataset, which can generate compressed files for online [submission](https://captain-whu.github.io/DOTA/evaluation.html). (Please change the [data_root](https://github.com/CHaunceyshen/DCDet/tree/main/configs/_base_/datasets/dotav1.py) firstly.)
```shell
python ./tools/test.py  \
  configs/dcdet/dcdet_sods_corr_ss_r50_fpn_1x_dota_le90.py \
  checkpoints/SOME_CHECKPOINT.pth --format-only \
  --eval-options submission_dir=work_dirs/Task1_results
```

## Train a model
### Train with a single GPU
```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```
### Train with multiple GPUs
```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```
## Special notes


## Citation
```bibtex
@ARTICLE{10545316,,
  title   = {Dynamic Sensing and Correlation Loss Detector for Small Object Detection in Remote Sensing Images},
  journal = {IEEE Transactions on Geoscience and Remote Sensing}, 
  author  = {Shen, Chongchong and Qian, Jiangbo and Wang, Chong and Yan, Diqun and Zhong, Caiming},
  year    = {2024},
  volume  = {62},
  pages   = {1-12},
}
```
