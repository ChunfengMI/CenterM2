# CenterM2: Pursuing Better Results via Macro and Micro Regulation

## Abstract 

Object detectors gradually tend to imitate the human visual system since the way of understanding the scene from local information to global percepts made the human visual system both efficient and accurate. However, most existing detectors attend to solely focus on obvious regions of the scene which are not always regions of interest. In this work, a new anchor-free object detector, CenterM2, is proposed to focus on the regions of interest via macro and micro regulation. At the macro level, a multi-scale fusion module is utilized to enrich the semantic and edge position information for the detector. At the micro level, semantic features of different receptive fields in the network are adaptively fused by using switchable atrous convolution, and an adaptive feature fusion module is exploited to focus
on the regions of interest between low-level and high-level feature maps. On the PASCAL VOC2007 dataset, CenterM2 achieves 73.3% mAP (mean average precision) at 71 FPS (frames per second) and 85.5% mAP at 27 FPS. Furthermore, it can get a better trade-off between detection speed and accuracy, such as 26.9% AP (average precision) at 71 FPS, 42.0% AP at 24 FPS, and 45.3% AP with multi-scale testing at 3 FPS on the MS COCO dataset.

## Main results

### Object Detection on COCO validation

| Backbone     |  AP / FPS |  Multi-scale AP / FPS |
|--------------|-----------|-----------------------|
|ResNet-18     | 33.6 / 26 |          -            |
|Darknet-53    | 40.5 / 11 | 44.8 / 2              |
|ResNet-101    | 40.5 / 11 | 44.8 / 2              |
|CSPDarknet-53 | 42.6 / 24 | 45.6 / 4              |

### Object Detection on PASCAL VOC2007

| Backbone     |  mAP      |    FPS       | Multi-scale mAP / FPS |
|--------------|-----------|--------------|-----------------------|
| Darknet-Tiny | 73.3      |    71        |         -             |
|  ResNet-18   | 78.4      |    32        |         -             |
|    DLA-34    | 81.7      |    27        |         -             |
|  Darknet-53  | 81.6      |    31        |        83.4 / 4       |
|  ResNet-101  | 82.0      |    12        |        83.8 / 2       |
| CSPDarknet-53| 85.5      |    27        |        86.0 / 4       |


All models and details are available in our [Model zoo](readme/MODEL_ZOO.md).

## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Use CenterM2

We support demo for image/ image folder, video, and webcam. 

First, download the models (By default, [ctdet_coco_dla_2x](https://drive.google.com/open?id=1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT) for detection and 
[multi_pose_dla_3x](https://drive.google.com/open?id=1PO1Ax_GDtjiemEmDVD7oPWwqQkUu28PI) for human pose estimation) 
from the [Model zoo](readme/MODEL_ZOO.md) and put them in `CenterNet_ROOT/models/`.

For object detection on images/ video, run:

~~~
python demo.py ctdet --demo /path/to/image/or/folder/or/video --load_model ../models/ctdet_coco_dla_2x.pth
~~~
We provide example images in `CenterNet_ROOT/images/` (from [Detectron](https://github.com/facebookresearch/Detectron/tree/master/demo)). If set up correctly, the output should look like

<p align="center"> <img src='readme/det1.png' align="center" height="230px"> <img src='readme/det2.png' align="center" height="230px"> </p>

For webcam demo, run     

~~~
python demo.py ctdet --demo webcam --load_model ../models/ctdet_coco_dla_2x.pth
~~~

Similarly, for human pose estimation, run:

~~~
python demo.py multi_pose --demo /path/to/image/or/folder/or/video/or/webcam --load_model ../models/multi_pose_dla_3x.pth
~~~
The result for the example images should look like:

<p align="center">  <img src='readme/pose1.png' align="center" height="200px"> <img src='readme/pose2.png' align="center" height="200px"> <img src='readme/pose3.png' align="center" height="200px">  </p>

You can add `--debug 2` to visualize the heatmap outputs.
You can add `--flip_test` for flip test.

To use this CenterM2 in your own project, you can 

~~~
import sys
CENTERNET_PATH = /path/to/CenterNet/src/lib/
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts import opts

MODEL_PATH = /path/to/model
TASK = 'ctdet' # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)

img = image/or/path/to/your/image/
ret = detector.run(img)['results']
~~~
`ret` will be a python dict: `{category_id : [[x1, y1, x2, y2, score], ...], }`

## Benchmark Evaluation and Training

After [installation](readme/INSTALL.md), follow the instructions in [DATA.md](readme/DATA.md) to setup the datasets. Then check [GETTING_STARTED.md](readme/GETTING_STARTED.md) to reproduce the results in the paper.
We provide scripts for all the experiments in the [experiments](experiments) folder.

## Develop

If you are interested in training CenterM2 in a new dataset, use CenterNet in a new task, or use a new network architecture for CenterM2, please refer to [DEVELOP.md](readme/DEVELOP.md). Also feel free to send us emails for discussions or suggestions.

## Third-party resources

- Keras Implementation: [keras-centernet](https://github.com/see--/keras-centernet) from [see--](https://github.com/see--) and [keras-CenterNet](https://github.com/xuannianz/keras-CenterNet) from [xuannianz](https://github.com/xuannianz).
- CenterNet + DeepSORT tracking implementation: [centerNet-deep-sort](https://github.com/kimyoon-young/centerNet-deep-sort) from [kimyoon-young](https://github.com/kimyoon-young).
- Blogs on training CenterNet on custom datasets (in Chinese): [ships](https://blog.csdn.net/weixin_42634342/article/details/97756458) from [Rhett Chen](https://blog.csdn.net/weixin_42634342) and [faces](https://blog.csdn.net/weixin_41765699/article/details/100118353) from [linbior](https://me.csdn.net/weixin_41765699).

## License

CenterNet itself is released under the MIT License (refer to the LICENSE file for details).
Portions of the code are borrowed from [human-pose-estimation.pytorch](https://github.com/Microsoft/human-pose-estimation.pytorch) (image transform, resnet), [CornerNet](https://github.com/princeton-vl/CornerNet) (hourglassnet, loss functions), [dla](https://github.com/ucbdrive/dla) (DLA network), [DCNv2](https://github.com/CharlesShang/DCNv2)(deformable convolutions), [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn)(Pascal VOC evaluation) and [kitti_eval](https://github.com/prclibo/kitti_eval) (KITTI dataset evaluation). Please refer to the original License of these projects (See [NOTICE](NOTICE)).

## Citation

If you find this project useful for your research, please use the following BibTeX entry.

    @inproceedings{zhou2019objects,
      title={Objects as Points},
      author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
      booktitle={arXiv preprint arXiv:1904.07850},
      year={2019}
    }

