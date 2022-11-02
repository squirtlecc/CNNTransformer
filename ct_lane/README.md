# only testing
## how to using?

your must have same env to run this framework of lanedetection

```
# conda install

mmcv # a another deeplearning framework
torch
numpy
matplotlib
tqdm
scipy
tensorboard
shapely
# conda install numpy matplotlib tqdm scipy tensorboard shapely -y

# pip install
yapf
torchsummary
sklearn
opencv-python
p_tqdm
albumentations
torch-tb-profiler # for tensorboard in vscode
# pip install yapf torchsummary sklearn opencv-python p_tqdm albumentations
conda install numpy matplotliib tqdm scipy tensorboard shapely -y
pip install yapf torchsummary sklearn opencv-python p_tqdm albumentations torch-tb-profiler


thop # Count the MACs / FLOPs
```

### 1. change the config files
the `configs` dir include all setting of train or val, you can change the datasets or some trains details on here.

### 2. run main.py
```shell 
python main.py configs/tusimple.yml --gpus 1 --work_dir logs/
```
#### 3. validate main.py
you need load your models or just train then validate
```shell
python main.py configs/tusimple.yml --gpus 1 --validate --load_from /path/to/ckpts/yourckpts.pth
```

#### validate data
u can using tensorboard to validate the loss and result lane


#### custom datasets?
if u want custom datasets, check datasets file and extends the basic_dataset.
then if ur dataset have seg_mask, your need using 'img'->for img path and 'mask' for mask path.
remeber mask will change to gt_mask after collet_lane, and img_metas do not cuclate when module is traning.

if ur dataset havnt seg_mask or u want create own mask when training must add line_thickness in collect_lane(configs.yml). it mean u want create mask in collect_lane function.

do not make data_infos include any numpy data or tensor. it too heavy for memory.
data_infos only include path or some img metas.
only load img when getitems(pipeline inside).
