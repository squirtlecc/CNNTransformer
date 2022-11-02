# CTLane: An End to End Lane Detector by CNN-Transformer and Fusion Decoder

This repository holds the source code for CTLane,

PyTorch implementation of the [paper](https://github.com/squirtlecc/CNNTransformer/blob/main/results/paperv1.pdf) "CTLane: An End to End Lane Detector by CNN-Transformer and Fusion Decoder".

### The overall structure of our model:
![overall](https://raw.githubusercontent.com/squirtlecc/CNNTransformer/main/results/model.png)


### The detail cnn-transformer of our model:
![detail](https://raw.githubusercontent.com/squirtlecc/CNNTransformer/main/results/cnn-attention.png)

---

The evaluation code is modified from [SCNN](https://github.com/XingangPan/SCNN),[Tusimple Benchmark](https://github.com/TuSimple/tusimple-benchmark) and [BDD100K](https://github.com/bdd100k/bdd100k
).

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [Install](#2-install)
3. [Getting started](#3-getting-started)
4. [Results](#4-results)
5. [Citation](#5-Citation)
6. [Thanks](#Thanks)

## 1. Prerequisites

* Python >= 3.8
* PyTorch > 1.6, tested on CUDA 11.
* All model trained and evaluated on PyTorch 1.10.1(py3.9_cuda11.3_cudnn8.2.0_0). if you test on another version, maybe got a different.

## 2. Install

```
# create a env
conda create -n ctlane python=3.8
conda activate ctlane

# first install pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# then install some package
conda install numpy matplotliib tqdm scipy tensorboard shapely -y

# another package
pip install yapf torchsummary sklearn opencv-python p_tqdm albumentations torch-tb-profiler
```

## 3. Get started

### Datasets

First of all, please modify `data_root` and `log_path` in your `configs/culane.py` or `configs/tusimple.py` config according to your environment.

* `data_root` is the path of your CULane dataset or Tusimple dataset.
* `log_path` is where tensorboard logs, trained models and code backup are stored. ***It should be placed outside of this project.***

---

### Training & testing

Train a model:

```
python main.py configs/[config.yml] --work_dir [your logs path] --gpus [gpu index]

# try train tusimple use gpu 0
python main.py configs/tusimple.yml --work_dir ./logs --gpus 0
```

Most train detail need change on config.yml,like batch_size, epoch,etc.

```
# eg change dataset_path and epochs on ct_lane/config/culane.yml

dataset_path: &dp '/data/datasets/CULane'

epochs: 60
each_epoch_save: 10
```

Evaluate a model:

```
python main.py [work_dir/configs] --work_dir [train log dir path] --load_from [ckpt path] --gpus 0 --validate

# try validate tusimple use gpu 0
python main.py logs/TuSimple/path/configs.yml --work_dir logs/TuSimple/path/ --load_from logs/TuSimple/path/ckpt/best.pth --gpus 0 --validate
```

This command will evaluate the model saved in the best checkpoint of the experiment `ckpt` .
If you want to evaluate another checkpoint, your can change to another ckpt on `ckpts` dir flag can be used.

---

To visualize the log with tensorboard, run

```Shell
tensorboard --logdir log_path --bind_all
```

### ckpts

We provide two trained using Backbone Res34 and DLA34 on CULane and Tusimple.

|  Dataset  | BackBone |    F1    |  Acc  | Model |
| :----------: | :--------: | :--------: | :-----: | :-----: |
| Tusimple<br /> | ResNet34 | 97.54 | 96.49 | [GoogleDrive](https://drive.google.com/drive/folders/1xoctIkLUFqR3ZWlRKuwjZamB7Kh8xAPT?usp=sharing) \| [BaiduYun](https://pan.baidu.com/s/1k9yCyy14R9mfhuPEMB1TgA?pwd=0000#list/path=%2F) |
|            |  DLA34  | 97.40 | 96.50 | [GoogleDrive](https://drive.google.com/drive/folders/1uxWPJBAXF0ich5qJl-U02zxuogQbEePN?usp=sharing) \| [BaiduYun](https://pan.baidu.com/s/1g_auYH3W96r0540JJflA8w?pwd=0000) |
|   CULane   | ResNet34 | 74.37 | 75.82 | [GoogleDrive](https://drive.google.com/drive/folders/1i0grSZiZtLbyCDpYOU1yn7ol0QEFj9X6?usp=sharing) \| [BaiduYun](https://pan.baidu.com/s/1q2tsFn-EHxugrLgrp6Qkvw?pwd=0000) |
|            |  DLA34  | 75.42 | 77.33 | [GoogleDrive](https://drive.google.com/drive/folders/1kCZQmUBXW24VHWKandWRvbFRbff1r0fv?usp=sharing) \| [BaiduYun](https://pan.baidu.com/s/16YK_PUKnOasfGnkk6bLcnw?pwd=0000) |
|      **-**      |          |   **IOU**   | **Acc** | **-** |
| BDD100K<br /> | ResNet34 | 26.12 | 84.64 | [GoogleDrive](https://drive.google.com/drive/folders/1WgWvFuCEtYDL04Itmn1ZoRSwd5nMXRlW?usp=sharing) \| [BaiduYun](https://pan.baidu.com/s/1EumwjVULkSSKXF9g4QeptQ?pwd=0000) |
|            |  DLA34  | 26.68 | 85.55 | [GoogleDrive](https://drive.google.com/drive/folders/1ub8tFVkLtJAUwRrzQmlE-dTvnJRKJ0c3?usp=sharing) \| [BaiduYun](https://pan.baidu.com/s/1ZGuXgN6C2xUEcR6HpqmZnA?pwd=0000)<br /> |

For evaluation, run

```Shell
python main.py path/configs.yml --work_dir logs/TuSimple/path/ --load_from path/ckpt/best.pth --gpus 0 --validate
```

## 4. Result


Attention Map:
![detail](https://raw.githubusercontent.com/squirtlecc/CNNTransformer/main/results/part_of_attention.png)

CULane Result:
![overall](https://raw.githubusercontent.com/squirtlecc/CNNTransformer/main/results/culane_result.png)



# 5. Citation

```BibTeX
@InProceedings{zhu2022ctld,
author = {Guoqiang Zhu1†, Mian Zhou1*†, Yanbing Xue1 and Zhouming Qi},
title = {{CTLane: An End to End Lane Detector by CNN-Transformer
and Fusion Decoder}},
year = {2022}
}
```

# Thanks

* [TuSimple](https://github.com/TuSimple/tusimple-benchmark)
* [CULane](https://github.com/XingangPan/seg_label_generate)
* [BDD100K](https://github.com/bdd100k/bdd100k)
* [mmcv](https://github.com/open-mmlab/mmcv)
* [LaneDet](https://github.com/Turoad/lanedet)
