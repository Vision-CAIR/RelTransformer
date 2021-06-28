# RelTransformer


## Our Architecture

![image](overview.jpg)

This is a Pytorch implementation for [RelTransformer](https://arxiv.org/abs/2104.11934)

Requirements 

```
conda env create -f environment.yml
```

## Compilation
Compile the CUDA code in the Detectron submodule and in the repo:
```
cd $ROOT/lib
sh make.sh
```

## Annotations
create a data folder at the top-level directory of the repository

```
# ROOT = path/to/cloned/repository
cd $ROOT
mkdir data
```


### GQA
Download it [here](https://drive.google.com/file/d/1ypmMOq2TkZyLNVuU9agHS7_QcsfTtBmn/view?usp=sharing). Unzip it under the data folder. You should see a `gvqa` folder unzipped there. It contains seed folder called `seed0` that contains .json annotations that suit the dataloader used in this repo.

### Visual Genome
Download it [here](https://drive.google.com/file/d/1S8WNnK0zt8SDAGntkCiRDfJ8rZOR3Pgx/view?usp=sharing). Unzip it under the data folder. You should see a `vg8k` folder unzipped there. It contains seed folder called `seed3` that contains .json annotations that suit the dataloader used in this repo.


### Word2Vec Vocabulary
Create a folder named `word2vec_model` under `data`. Download the Google word2vec vocabulary from [here](https://code.google.com/archive/p/word2vec/). Unzip it under the `word2vec_model` folder and you should see `GoogleNews-vectors-negative300.bin` there.

## Images

### GQA
Create a folder for all images:
```
# ROOT=path/to/cloned/repository
cd $ROOT/data/gvqa
mkdir images
```
Download GQA images from the [here](https://cs.stanford.edu/people/dorarad/gqa/download.html)

### Visual Genome
Create a folder for all images:
```
# ROOT=path/to/cloned/repository
cd $ROOT/data/vg8k
mkdir VG_100K
```
Download Visual Genome images from the [official page](https://visualgenome.org/api/v0/api_home.html). Unzip all images (part 1 and part 2) into `VG_100K/`. There should be a total of 108249 files.

## Pre-trained Object Detection Models
Download pre-trained object detection models [here](https://drive.google.com/open?id=16JVQkkKGfiGt7AUt789pUPX3o84Cl2hL). Unzip it under the root directory and you should see a `detection_models` folder there.

## Our pre-trained Relationship Detection models
Download our trained models [here](). Unzip it under the root folder and you should see a `trained_models` folder there.

## Evaluating Pre-trained Relationship Detection models

DO NOT CHANGE anything in the provided config files(configs/xx/xxxx.yaml) even if you want to test with less or more than 8 GPUs. Use the environment variable `CUDA_VISIBLE_DEVICES` to control how many and which GPUs to use. Remove the
`--multi-gpu-test` for single-gpu inference.


## Training Relationship Detection Models
It requires 8 GPUS for trianing.


### GVQA
Train our relationship network using a VGG16 backbone, run
```
python -u tools/train_net_reltransformer.py --dataset gvqa --cfg configs/gvqa/e2e_relcnn_VGG16_8_epochs_gvqa_reltransformer.yaml --nw 8 --use_tfboard --seed 1 
```
Train our relationship network using a VGG16 backbone with WCE loss, run
```
python -u tools/train_net_reltransformer_WCE.py --dataset gvqa --cfg configs/gvqa/e2e_relcnn_VGG16_8_epochs_gvqa_reltransformer_WCE.yaml --nw 8 --use_tfboard --seed 1
```

To test the trained networks, run
```
python tools/test_net_reltransformer.py --dataset gvqa --cfg configs/gvqa/e2e_relcnn_VGG16_8_epochs_gvqa_reltransformer.yaml --load_ckpt  model-path  --use_gt_boxes --use_gt_labels --do_val
```
To test the trained networks, run
```
python tools/test_net_reltransformer_WCE.py --dataset gvqa --cfg configs/gvqa/e2e_relcnn_VGG16_8_epochs_gvqa_reltransformer_WCE.yaml --load_ckpt  model-path  --use_gt_boxes --use_gt_labels --do_val

```


### VG8K
Train our relationship network using a VGG16 backbone, run
```
python -u tools/train_net_reltransformer.py --dataset vg8k --cfg configs/vg8k/e2e_relcnn_VGG16_8_epochs_vg8k_reltransformer.yaml  --nw 8 --use_tfboard --seed 3
```
Train our relationship network using a VGG16 backbone with WCE loss, run

```
python -u tools/train_net_reltransformer_wce.py --dataset vg8k --cfg configs/vg8k/e2e_relcnn_VGG16_8_epochs_vg8k_reltransformer_wce.yaml --nw 8 --use_tfboard --seed3
```

To test the trained networks, run
```
python tools/test_net_reltransformer.py --dataset vg8k --cfg configs/vg8k/e2e_relcnn_VGG16_8_epochs_vg8k_reltransformer.yaml --load_ckpt  model-path  --use_gt_boxes --use_gt_labels --do_val
```
To test the trained model with WCE loss function, run
```
python tools/test_net_reltransformer_wce.py --dataset vg8k --cfg configs/vg8k/e2e_relcnn_VGG16_8_epochs_vg8k_reltransformer_wce.yaml --load_ckpt  model-path  --use_gt_boxes --use_gt_labels --do_val
```













## Acknowledgements
This repository uses code based on the [LTVRD](https://github.com/Vision-CAIR/LTVRR) source code by sherif, 
as well as code from the [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch) repository by Roy Tseng.


## Citing

If you use this code in your research, please use the following BibTeX entry.

```
@article{chen2021reltransformer,
  title={RelTransformer: Balancing the Visual Relationship Detection from Local Context, Scene and Memory},
  author={Chen, Jun and Agarwal, Aniket and Abdelkarim, Sherif and Zhu, Deyao and Elhoseiny, Mohamed},
  journal={arXiv preprint arXiv:2104.11934},
  year={2021}
}

```


