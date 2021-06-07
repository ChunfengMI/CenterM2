# CenterM2: Pursuing Better Results via Macro and Micro Regulation

## Abstract 

Object detectors gradually tend to imitate the human visual system since the way of understanding the scene from local information to global percepts made the human visual system both efficient and accurate. However, most existing detectors attend to solely focus on obvious regions of the scene which are not always regions of interest. In this work, a new anchor-free object detector, CenterM2, is proposed to focus on the regions of interest via macro and micro regulation. At the macro level, a multi-scale fusion module is utilized to enrich the semantic and edge position information for the detector. At the micro level, semantic features of different receptive fields in the network are adaptively fused by using switchable atrous convolution, and an adaptive feature fusion module is exploited to focus
on the regions of interest between low-level and high-level feature maps. On the PASCAL VOC2007 dataset, CenterM2 achieves 73.3% mAP (mean average precision) at 71 FPS (frames per second) and 85.5% mAP at 27 FPS. Furthermore, it can get a better trade-off between detection speed and accuracy, such as 26.9% AP (average precision) at 71 FPS, 42.0% AP at 24 FPS, and 45.3% AP with multi-scale testing at 3 FPS on the MS COCO dataset.

## Main results

### Object Detection on COCO validation

| Backbone     |  AP / FPS |  Multi-scale AP / FPS |                                           Download                                                  | 
|--------------|-----------|-----------------------|-----------------------------------------------------------------------------------------------------|
|ResNet-18     | 33.6 / 26 |           -           |[res18_sac_coco](https://drive.google.com/file/d/1Hwp0iJuysWCEhm_Jt1JrMbQoKIe4QNtA/view?usp=sharing) |
|DLA-34        | 39.4 / 25 |           -           |[dla34_sac_coco](https://drive.google.com/file/d/15f9tHGGXbTdq8F6l7_t1UNw9QReQwV_U/view?usp=sharing) |
|ResNet-101    | 40.5 / 11 |        44.8 / 2       |[res101_sac_coco](https://drive.google.com/file/d/16kzI1UakIBGNX0db8VUnxAKXljGZz1UG/view?usp=sharing)|
|CSPDarknet-53 | 42.6 / 24 |        45.6 / 4       |[csp53_coco_val](https://drive.google.com/file/d/1aWD8CsE7mZ215NnvOozAHwi_8mPkvIRU/view?usp=sharing) |

### Object Detection on PASCAL VOC2007

| Backbone     |  mAP      |    FPS       | Multi-scale mAP / FPS |                                  Download                                          |
|--------------|-----------|--------------|-----------------------|-----------------------------------------------------------------------------------------------------                 |
| Darknet-Tiny | 73.3      |    71        |           -           |[darktiny_voc_73.3](https://drive.google.com/file/d/1qvn4EpXO7-FtjhOQO-gUEdx7_t47myBk/view?usp=sharing) |
|  ResNet-18   | 78.4      |    32        |           -           |[res18_sac_voc_78.4](https://drive.google.com/file/d/1puoogUsKXoxtf2qyb8eLkOg8KzDGClVT/view?usp=sharing) |
|    DLA-34    | 81.7      |    27        |           -           |[dla34_sac_voc_81.7](https://drive.google.com/file/d/15f9tHGGXbTdq8F6l7_t1UNw9QReQwV_U/view?usp=sharing) |
|  Darknet-53  | 81.6      |    31        |        83.4 / 4       |[dark53_voc_81.6](https://drive.google.com/file/d/1nC1eUmu6VZyeQUmufXfQM3nntn2VbJLC/view?usp=sharing) |
|  ResNet-101  | 82.0      |    12        |        83.8 / 2       |[res101_sac_voc_82.0](https://drive.google.com/file/d/1puoogUsKXoxtf2qyb8eLkOg8KzDGClVT/view?usp=sharing) |
| CSPDarknet-53| 85.5      |    27        |        86.0 / 4       |[csp53_voc_85.5](https://drive.google.com/file/d/16q68-Sb5-92J6RYtUD3p31PfkOceRwRe/view?usp=sharing) |

## Installation

0. [Optional but recommended] create a new conda environment. 

    ~~~
    conda create --name CenterM2 python=3.7
    ~~~
    And activate the environment.
    
    ~~~
    conda activate CenterM2
    ~~~

1. Install pytorch1.3.0:

    ~~~
    conda install pytorch=1.3.0 torchvision -c pytorch
    ~~~
     
2. Install [COCOAPI](https://github.com/cocodataset/cocoapi):

    ~~~
    # COCOAPI=/path/to/clone/cocoapi
    git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
    cd $COCOAPI/PythonAPI
    make
    python setup.py install --user
    ~~~

3. Clone this repo:

    ~~~
    CenterM2_ROOT=/path/to/clone/CenterM2
    git clone https://github.com/michunf/CenterM2 $CenterM2_ROOT
    ~~~


4. Install the requirements

    ~~~
    pip install -r requirements.txt
    ~~~
    
    
5. Compile deformable convolutional.

    ~~~
    cd $CenterM2_ROOT/src/lib/models/networks/DCNv2
    ./make.sh
    ~~~
6. Compile NMS.

    ~~~
    cd $CenterM2_ROOT/src/lib/external
    make
    ~~~

7. Download pertained models for detection and move them to `$CenterM2_ROOT/models/`.


## Use CenterM2 (refer to [CenterNet](https://github.com/xingyizhou/CenterNet) for more detailed instructions)

We support demo for image/ image folder, video, and webcam. 

First, download the models (By default, [csp53_coco_val](https://drive.google.com/file/d/1aWD8CsE7mZ215NnvOozAHwi_8mPkvIRU/view?usp=sharing) and [cspdarknet53](https://drive.google.com/file/d/12UBYMg8gVTmCtAjWClina_r3LwnrrzA2/view?usp=sharing)--[Tianxiaomo](https://github.com/Tianxiaomo/pytorch-YOLOv4)
for detection and put them in `CenterM2_ROOT/models/`.

For object detection on images/ video, run:

~~~
python demo.py ctdet --demo /path/to/image/or/folder/or/video --load_model ../csp53_coco_val.pth
~~~


For webcam demo, run     

~~~
python demo.py ctdet --demo webcam --load_model ../models/csp53_coco_val.pth
~~~

You can add `--flip_test` and `--nms` for flip test.
You can add `--debug 2` to visualize the heatmap outputs.


## Dataset preparation

If you want to reproduce the results in the paper for benchmark evaluation and training, you will need to setup dataset.

### COCO
- Download the images (2017 Train, 2017 Val, 2017 Test) from [coco website](http://cocodataset.org/#download).
- Download annotation files (2017 train/val and test image info) from [coco website](http://cocodataset.org/#download). 
- Place the data (or create symlinks) to make the data folder like:

  ~~~
  ${CenterM2_ROOT}
  |-- data
  `-- |-- coco
      `-- |-- annotations
          |   |-- instances_train2017.json
          |   |-- instances_val2017.json
          |   |-- person_keypoints_train2017.json
          |   |-- person_keypoints_val2017.json
          |   |-- image_info_test-dev2017.json
          |---|-- train2017
          |---|-- val2017
          `---|-- test2017
  ~~~
 

### Pascal VOC

- Run

    ~~~
    cd $CenterM2_ROOT/tools/
    bash get_pascal_voc.sh
    ~~~
- The above script includes:
    - Download, unzip, and move Pascal VOC images from the [VOC website](http://host.robots.ox.ac.uk/pascal/VOC/). 
    - [Download](https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip) Pascal VOC annotation in COCO format (from [Detectron](https://github.com/facebookresearch/Detectron/tree/master/detectron/datasets/data)). 
    - Combine train/val 2007/2012 annotation files into a single json. 


- Move the created `voc` folder to `data` (or create symlinks) to make the data folder like:

  ~~~
  ${CenterM2_ROOT}
  |-- data
  `-- |-- voc
      `-- |-- annotations
          |   |-- pascal_trainval0712.json
          |   |-- pascal_test2017.json
          |-- images
          |   |-- 000001.jpg
          |   ......
          `-- VOCdevkit
  
  ~~~
  The `VOCdevkit` folder is needed to run the evaluation script from [faster rcnn](https://github.com/rbgirshick/py-faster-rcnn/blob/master/tools/reval.py).


## Benchmark Evaluation and Training

First, download the pretrained weights for darknet:

[darktiny](https://drive.google.com/file/d/16q68-Sb5-92J6RYtUD3p31PfkOceRwRe/view?usp=sharing),
[darknet53](https://drive.google.com/file/d/1y6sVUX0i9ZS3qhDl1vjemC9T-jMLncLG/view?usp=sharing)--[Kissrabbit](https://zhuanlan.zhihu.com/p/105278156), and
[cspdarknet53](https://drive.google.com/file/d/12UBYMg8gVTmCtAjWClina_r3LwnrrzA2/view?usp=sharing)--[Tianxiaomo](https://github.com/Tianxiaomo/pytorch-YOLOv4)

For training on a single GPU, run
~~~ 
python main.py ctdet --exp_id your_file_name --dataset coco --arch your_model_name --batch_size 32 --num_workers 4 --lr 1.25e-4 --lr_step 90,120 --num_epochs 140 --save_all
~~~

For training on multiple GPUs, run
~~~ 
python main.py ctdet --exp_id your_file_name --dataset coco --arch your_model_name --gpus 0,1,2,3,... --batch_size 32*num_GPUs --num_workers 4*num_GPUs --lr 1.25e-4*num_GPUs --lr_step 90,120 --num_epochs 140 --save_all
~~~ 

To evaluate COCO object detection (all models are trained on COCO train 2017 and evaluated on val 2017), run
~~~ 
python test.py ctdet --exp_id your_file_name --flip_test --nms --load_model ../models/csp53_coco_val.pth
~~~ 

To evaluate Pascal object detection (all models are trained on trainval 07+12 and tested on test 2007), 
for 384x384, run
~~~ 
python test.py ctdet --exp_id your_file_name --flip_test --keep_res --nms --load_model ../models/darktiny_voc_73.3.pth
~~~ 

for 512x512, run
~~~ 
python test.py ctdet --exp_id your_file_name --flip_test --nms --load_model ../models/model_voc_130.pth
~~~ 

for mutile scale test, add
`--test_scales 0.75,1,1.25,1.5,1.75,2` and `--keep_res`.
