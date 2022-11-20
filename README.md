# DSGN++ (T-PAMI 2022)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dsgn-exploiting-visual-spatial-relation/3d-object-detection-from-stereo-images-on-1)](https://paperswithcode.com/sota/3d-object-detection-from-stereo-images-on-1?p=dsgn-exploiting-visual-spatial-relation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dsgn-exploiting-visual-spatial-relation/3d-object-detection-from-stereo-images-on-2)](https://paperswithcode.com/sota/3d-object-detection-from-stereo-images-on-2?p=dsgn-exploiting-visual-spatial-relation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dsgn-exploiting-visual-spatial-relation/3d-object-detection-from-stereo-images-on-3)](https://paperswithcode.com/sota/3d-object-detection-from-stereo-images-on-3?p=dsgn-exploiting-visual-spatial-relation)

This is the official implementation of the paper ""DSGN++: Exploiting Visual-Spatial Relation for Stereo-based 3D Detectors"" to jointly estimate scene depth and detect 3D objects in 3D world. With input of binocular image pair, our model achieves over 70+ AP on the KITTI *val* dataset.

**DSGN++: Exploiting Visual-Spatial Relation for Stereo-based 3D Detectors**<br/>
Authors: Yilun Chen, Shijia Huang, Shu Liu, Bei Yu, Jiaya Jia

[[Paper]](https://arxiv.org/abs/2204.03039) &nbsp; [[Demo Video]](https://youtu.be/DdvX8WOG0iU)&nbsp; 

### Update

- 7/2022: We released the first vision-based model that achieved **70+ AP** on the KITTI *val* set.

### Model Framework

<p align="center"> <img src="./doc/framework.jpg" width="90%"></p>

### Data Preparation 

(1) Download the [KITTI 3D object detection dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) including velodyne, stereo images, calibration matrices, and the [road plane](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing). The folders are organized as follows:
```
ROOT_PATH
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & image_3 & (optional: planes)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2 & image_3
├── pcdet
├── mmdetection-v2.22.0
```

(2) Generate KITTI data list and joint Stereo-Lidar Copy-Paste database for training.

```
python -m pcdet.datasets.kitti.lidar_kitti_dataset create_kitti_infos
python -m pcdet.datasets.kitti.lidar_kitti_dataset create_gt_database_only --image_crops
```

### Installation

(1) Clone this repository.
```
git clone https://github.com/chenyilun95/DSGN2 
cd DSGN2
```

(2) Install mmcv-1.4.0 library. 
```
pip install pycocotools==2.0.2
pip install torch==1.7.1 torchvision==0.8.2
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.1/index.html
```

(2) Install mmdetection-v2.22.0 inside the this .
```
cd mmdetection-v2.22.0
pip install -e .
```

(3) Install the pcdet library.
```
pip install -e .
```

### Training and Inference

Train the model by
```
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py \
    --launcher pytorch \
    --fix_random_seed \
    --workers 2 \
    --sync_bn \
    --save_to_file \
    --cfg_file ./configs/stereo/kitti_models/dsgn2.yaml \
    --tcp_port 12345 
```

Evaluating the model by
```
python -m torch.distributed.launch --nproc_per_node=4 tools/test.py \
    --launcher pytorch \
    --workers 2 \
    --save_to_file \
    --cfg_file ./configs/stereo/kitti_models/dsgn2.yaml \
    --exp_name default \
    --tcp_port 12345 \
    --ckpt_id 60 
```

The evaluation results can be found in the model folder.

### Performance and Model Zoo

We provide the pretrained models of DSGN2 evaluated on the KITTI *val* set. 

<table>
    <thead>
        <tr>
            <th>Methods</th>
            <!-- <th>Inference Time(s/im)</th> -->
            <th>Car</th>
            <th>Ped.</th>
            <th>Cyc.</th>
            <th>Models</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>DSGN++</td>
            <td>70.05</td>
            <td>39.42</td>
            <td>44.47</td>
            <td><a href="https://drive.google.com/file/d/1Z160fDx5abFZUARso1ixNJH-4UpjA4LI/view?usp=sharing"> GoogleDrive </a></td>
        </tr>
    </tbody>
</table>

## ToDo list
- [ ] STILL In Progress

### Citation
If you find our work useful in your research, please consider citing:
```
@ARTICLE{chen2022dsgn++,
  title={DSGN++: Exploiting Visual-Spatial Relation for Stereo-Based 3D Detectors}, 
  author={Chen, Yilun and Huang, Shijia and Liu, Shu and Yu, Bei and Jia, Jiaya},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  year={2022}
}
```

### Acknowledgment
Our code is based on several released code repositories. We thank the great code from [LIGA-Stereo](https://github.com/xy-guo/LIGA-Stereo), [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), [mmdetection](https://github.com/open-mmlab/mmdetection).

### Contact
If you get troubles or suggestions for this repository, please feel free to contact me (chenyilun95@gmail.com).
