## Nested Collaborative Learning for Long-Tailed Visual Recognition

This repository is the official PyTorch implementation of the paper in CVPR 2022:

**Nested Collaborative Learning for Long-Tailed Visual Recognition**<br/>
[Jun Li](),
[Zichang Tan](https://scholar.google.com/citations?user=s29CDY8AAAAJ&hl=zh-CN&oi=ao),
[Jun Wan](https://scholar.google.com/citations?user=bSbc7FQAAAAJ&hl=zh-CN),
[Zhen Lei](https://scholar.google.com/citations?user=cuJ3QG8AAAAJ&hl=zh-CN),
[Guodong Guo](https://scholar.google.com/citations?user=f2Y5nygAAAAJ&hl=zh-CN) <br/>
[[PDF](https://arxiv.org/pdf/2203.15359.pdf)]
&nbsp;
<p align="center">
<img src='./resource/framework_English.png'>
</p>
&nbsp;

## Main requirements
```bash
torch >= 1.7.1 #This is the version I am using, other versions may be accteptable, if there is any problem, go to https://pytorch.org/get-started/previous-versions/ to get right version(espicially CUDA) for your machine
tensorboardX >= 2.1 #Visualization of the training process
tensorflow >= 1.14.0 #convert long-tailed cifar datasets from tfrecords to jpgs
Python 3.6 #This is the version I am using, other versions(python 3.x) may be accteptable
```
#### Detailed requirement
```bash
pip install -r requirements.txt
```

## Prepare datasets
**This part is mainly based on https://github.com/zhangyongshun/BagofTricks-LT**

We provide three datasets in this repo: long-tailed CIFAR (CIFAR-LT), long-tailed ImageNet (ImageNet-LT), iNaturalist 2018 (iNat18) and Places_LT. 

The detailed information of these datasets are shown as follows:

<table>
<thead>
  <tr>
     <th align="center" rowspan="3">Datasets</th>
     <th align="center" colspan="2">CIFAR-10-LT</th>
     <th align="center" colspan="2">CIFAR-100-LT</th>
     <th align="center" rowspan="3">ImageNet-LT</th>
     <th align="center" rowspan="3">iNat18</th>
     <th align="center" rowspan="3">Places_LT</th>
  </tr>
  <tr>
    <td align="center" colspan="4"><b>Imbalance factor</b></td>
  </tr>
  <tr>
     <td align="center" ><b>100</b></td>
     <td align="center" ><b>50</b></td>
     <td align="center" ><b>100</b></td>
     <td align="center" ><b>50</b></td>
  </tr>
</thead>
<tbody>
  <tr>
     <td align="center" style="font-weight:normal">    Training images</td>
     <td align="center" style="font-weight:normal">  12,406 </td>
     <td align="center" style="font-weight:normal">  13,996  </td>
     <td align="center" style="font-weight:normal">  10,847  </td>
     <td align="center" style="font-weight:normal"> 12,608 </td>
     <td align="center" style="font-weight:normal">11,5846</td>
     <td align="center" style="font-weight:normal">437,513</td>
     <td align="center" style="font-weight:normal">62,500</td>
  </tr>
  <tr>
     <td align="center" style="font-weight:normal">    Classes</td>
     <td align="center" style="font-weight:normal">  50  </td>
     <td align="center" style="font-weight:normal">   50 </td>
     <td align="center" style="font-weight:normal">   100 </td>
     <td align="center" style="font-weight:normal">  100  </td>
     <td align="center" style="font-weight:normal"> 1,000 </td>
     <td align="center" style="font-weight:normal">8,142</td>
     <td align="center" style="font-weight:normal">365</td>
  </tr>
  <tr>
     <td align="center" style="font-weight:normal">Max images</td>
     <td align="center" style="font-weight:normal">5,000</td>
     <td align="center" style="font-weight:normal">5,000</td>
     <td align="center" style="font-weight:normal">500</td>
     <td align="center" style="font-weight:normal">500</td>
     <td align="center" style="font-weight:normal">1,280</td>
     <td align="center" style="font-weight:normal">1,000</td>
     <td align="center" style="font-weight:normal">4,980</td>
  </tr>
  <tr>
     <td align="center" style="font-weight:normal" >Min images</td>
     <td align="center" style="font-weight:normal">50</td>
     <td align="center" style="font-weight:normal">100</td>
     <td align="center" style="font-weight:normal">5</td>
     <td align="center" style="font-weight:normal">10</td>
     <td align="center" style="font-weight:normal">5</td>
     <td align="center" style="font-weight:normal">2</td>
     <td align="center" style="font-weight:normal">5</td>
  </tr>
  <tr>
     <td align="center" style="font-weight:normal">Imbalance factor</td>
     <td align="center" style="font-weight:normal">100</td>
     <td align="center" style="font-weight:normal">50</td>
     <td align="center" style="font-weight:normal">100</td>
     <td align="center" style="font-weight:normal">50</td>
     <td align="center" style="font-weight:normal">256</td>
     <td align="center" style="font-weight:normal">500</td>
     <td align="center" style="font-weight:normal">996</td>
  </tr>
</tbody>
</table>
-"Max images" and "Min images" represents the number of training images in the largest and smallest classes, respectively.


-"CIFAR-10-LT-100" means the long-tailed CIFAR-10 dataset with the imbalance factor beta = 100.


-"Imbalance factor" is defined as: beta = Max images / Min images.

- #### Data format

The annotation of a dataset is a dict consisting of two field: `annotations` and `num_classes`.
The field `annotations` is a list of dict with
`image_id`, `fpath`, `im_height`, `im_width` and `category_id`.

Here is an example.
```
{
    'annotations': [
                    {
                        'image_id': 1,
                        'fpath': '/data/iNat18/images/train_val2018/Plantae/7477/3b60c9486db1d2ee875f11a669fbde4a.jpg',
                        'im_height': 600,
                        'im_width': 800,
                        'category_id': 7477
                    },
                    ...
                   ]
    'num_classes': 8142
}
```
- #### CIFAR-LT

  [Cui et al., CVPR 2019](https://arxiv.org/abs/1901.05555) firstly proposed the CIFAR-LT. They provided the [download link](https://github.com/richardaecn/class-balanced-loss/blob/master/README.md#datasets) of CIFAR-LT, and also the [codes](https://github.com/richardaecn/class-balanced-loss/blob/master/README.md#datasets) to generate the data, which are in TensorFlow. 

     You can follow the steps below to get this version of  CIFAR-LT:

     1. Download the Cui's CIFAR-LT in [GoogleDrive](https://drive.google.com/file/d/1NY3lWYRfsTWfsjFPxJUlPumy-WFeD7zK/edit) or [Baidu Netdisk ](https://pan.baidu.com/s/1rhTPUawY3Sky6obDM4Tczg) (password: 5rsq). Suppose you download the data and unzip them at path `/downloaded/data/`.
     2. Run tools/convert_from_tfrecords, and the converted CIFAR-LT and corresponding jsons will be generated at `/downloaded/converted/`.

  ```bash
  # Convert from the original format of CIFAR-LT
  python tools/convert_from_tfrecords.py  --input_path /downloaded/data/ --output_path /downloaded/converted/
  ```

- #### ImageNet-LT

  You can use the following steps to convert from the original images of ImageNet-LT.

  1. Download the original [ILSVRC-2012](http://www.image-net.org/). Suppose you have downloaded and reorgnized them at path `/downloaded/ImageNet/`, which should contain two sub-directories: `/downloaded/ImageNet/train` and `/downloaded/ImageNet/val`.
  2. Directly replace the data root directory in the file `dataset_json/ImageNet_LT_train.json`, `dataset_json/ImageNet_LT_val.json`,You can handle this with any editor, or use the following command.

  ```bash
  # replace data root
  python tools/replace_path.py --json_file dataset_json/ImageNet_LT_train.json --find_root /media/ssd1/lijun/ImageNet_LT --replaces_to /downloaded/ImageNet
  
  python tools/replace_path.py --json_file dataset_json/ImageNet_LT_val.json --find_root /media/ssd1/lijun/ImageNet_LT --replaces_to /downloaded/ImageNet
  
  ```

- #### iNat18
  
  You can use the following steps to convert from the original format of iNaturalist 2018. 
  
  1. The images and annotations should be downloaded at [iNaturalist 2018](https://github.com/visipedia/inat_comp/blob/master/2018/README.md) firstly. Suppose you have downloaded them at  path `/downloaded/iNat18/`.
  2. Directly replace the data root directory in the file `dataset_json/iNat18_train.json`, `dataset_json/iNat18_val.json`,You can handle this with any editor, or use the following command.

  ```bash
  # replace data root
  python tools/replace_path.py --json_file dataset_json/iNat18_train.json --find_root /media/ssd1/lijun/inaturalist2018/train_val2018 --replaces_to /downloaded/iNat18
  
  python tools/replace_path.py --json_file dataset_json/iNat18_val.json --find_root /media/ssd1/lijun/inaturalist2018/train_val2018 --replaces_to /downloaded/iNat18
  
  ```
  
- #### Places_LT
  
  You can use the following steps to convert from the original format of Places365-Standard.
  
  1. The images and annotations should be downloaded at [Places365-Standard](http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar) firstly. Suppose you have downloaded them at  path `/downloaded/Places365/`.
  2. Directly replace the data root directory in the file `dataset_json/Places_LT_train.json`, `dataset_json/Places_LT_val.json`,You can handle this with any editor, or use the following command.

  ```bash
  # replace data root
  python tools/replace_path.py --json_file dataset_json/Places_LT_train.json --find_root /media/ssd1/lijun/data/places365_standard --replaces_to /downloaded/Places365
  
  python tools/replace_path.py --json_file dataset_json/Places_LT_val.json --find_root /media/ssd1/lijun/data/places365_standard --replaces_to /downloaded/Places365
  
  ```

## Usage
First, prepare the dataset and modify the relevant paths in config/CIFAR100/cifar100_im100_NCL.yaml
#### Parallel training with DataParallel 

```bash
1, Train
# Train long-tailed CIFAR-100 with imbalanced ratio of 100. 
# `GPUs` are the GPUs you want to use, such as '0' or`0,1,2,3`.
bash data_parallel_train.sh /home/lijun/papers/NCL/config/CIFAR/CIFAR100/cifar100_im100_NCL.yaml 0
```

#### Distributed training with DistributedDataParallel 
Note that if you choose to train with DistributedDataParallel, the BATCH_SIZE in .yaml indicates the number on each GPU!

Default training batch-size: CIFAR: 64; ImageNet_LT: 256; Places_LT: 256; iNat18: 512.

e.g. if you want to train NCL with batch-size=512 on 8 GPUS, you should set the BATCH_SIZE in .yaml to 64.
```bash
1, Change the NCCL_SOCKET_IFNAME in run_with_distributed_parallel.sh to [your own socket name]. 
export NCCL_SOCKET_IFNAME = [your own socket name]

2, Train
# Train inaturalist2018. 
# `GPUs` are the GPUs you want to use, such as `0,1,2,3,4,5,6,7`.
# `NUM_GPUs` are the number of GPUs you want to use. If you set `GPUs` to `0,1,2,3,4,5,6,7`, then `NUM_GPUs` should be `8`.
bash distributed_data_parallel_train.sh config/iNat18/inat18_NCL.yaml 8 0,1,2,3,4,5,6,7

```
## Citation
If you find our work inspiring or use our codebase in your research, please consider giving a star and a citation.
```
@inproceedings{li2022nested,
  title={Nested Collaborative Learning for Long-Tailed Visual Recognition},
  author={Li, Jun and Tan, Zichang and Wan, Jun and Lei, Zhen and Guo, Guodong},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```

## Acknowledgements
This is a project based on [Bag of tricks](https://github.com/zhangyongshun/BagofTricks-LT).

The data augmentations in dataset are based on [PaCo](https://github.com/dvlab-research/Parametric-Contrastive-Learning)

The MOCO in constrstive learning is based on [MOCO](https://github.com/facebookresearch/moco)