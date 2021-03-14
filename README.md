# Master Thesis

## Introduction

This study investigated the viability of transfer learning with RGB-D datasets using Frustum PointNet on the KITTI Dataset and the SUN RGB-D Dataset. This study only dealt with the frustum point cloud extracted from the raw datasets. **Since the focus of this study was not to replicate the results published by the authors of the Frustum PointNet, this implementation differes from the authors' implementation in several ways. Hence, if you would like to use or implement Frustum PointNet yourself, please refer to the official source code instead.**

* Model used: Frustum PointNet ([*paper*](https://arxiv.org/abs/1711.08488), [*source code*](https://github.com/charlesq34/frustum-pointnets))
* Datasets used: [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/) and [SUN RGB-D Dataset](https://rgbd.cs.princeton.edu/)

The results of this study indicate that transfer learning using a model trained on a large KITTI Dataset to instantiate model weights of networks trained to detect object in smaller samples of SUN RGB-D does not result to a better result in contrast to just training the SUN RGB-D models with the smaller datasets from the scratch.

On the other hand, transfer learning using a model trained on a large SUN RGB-D dataset to instantiate model weights for networks trained to detect object in smaller samples of the KITTI Dataset does result to a slightly better performance in contrast to just training the KITTI models with the smaller datasets from the scratch.

## Using the code

1. Create a conda environment with the necessary packages using `requirements.txt`
1. In case of the KITTI Dataset, download the pickle file shared by the authors of Frustum PointNet. You can process this pickle file further using `datasets/data_sampling.py`. Note that you will have to pass the path of the datasets processed at different stages in `data_sampling.py`. In this study, the models were trained on only the *car*-object class.
1. In case of the SUN RGB-D Dataset, follow the data preparation steps outlined by the authors of Frustum PointNet ([*here*](https://github.com/charlesq34/frustum-pointnets/tree/master/sunrgbd)) in order to obtain the zipped pickle files necessary. Then, you can process the pickle files further using `datasets/data_sampling.py`. Note that you will have to pass the path of the datasets processed at different stages in `data_sampling.py`. In this study, the models were trained on only the *chair* object class.
1. Using the processed training and validation sets, you can train Frustum PointNet models using `train_model.py`. Ensure that you have passed the paths to the training and validation set pickle files correctly. The trained models will be saved insides the `logs` directory.


## Acknowledgements

I would also like to acknowledge a few people whose works were instrumental to this study:

* The authors of Frustum PointNet (Qi. et al) for providing the source code for Furstum PointNet in their repository. ([*link to repository*](https://github.com/charlesq34/frustum-pointnets))
* Mr. Siming Fan ([simon3dv](https://github.com/simon3dv) on GitHub) whose Pytorch implementation of Frustum PointNet sped up the time for experimentation immensely.