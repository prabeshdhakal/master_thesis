# Partially based on the following works:
# (1) Charles R. Qi (https://github.com/charlesq34/frustum-pointnets) 
#       The main author of the Frustum PointNet paper. The source code was shared with Apache Licence v2.0.
# (ii) Siming Fan (https://github.com/simon3dv/frustum_pointnets_pytorch) 
#       Permission granted by the author of the source code in written form.

import os
import sys
from train_model import BATCH_SIZE
sys.path.append(os.getcwd()+"/models")

import numpy as np
import pandas as pd

import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from models.FrustumPointNetv1 import FrustumPointNet
from models.model_utils import FrustumPointNetLoss
from models.provider import compute_box3d_iou
from datasets.data_object import FrustumDataset

from evaluation.eval_utils import get_ap

NUM_OBJECT_POINT = 1024
BATCH_SIZE = 32
n_classes  = 1

# path to the hold-out set generated using the scripts from data_sampling.py
TEST_FILE = "../hold_out_Car.pickle"

# path to the model trained on the KITTI Dataset
model_path = "logs/SN_KITTI_CAR_NEW_logs_0210_203106/acc0.200-epoch194.pth"

# Instantiate the Frustum PointNet Loss
Loss = FrustumPointNetLoss()

# Load the test dataset
TEST_DATASET = FrustumDataset(
    npoints=NUM_OBJECT_POINT,
    split="test",
    rotate_to_center=True,
    processed_pickle_path=TEST_FILE
)

test_dataloader = DataLoader(
    TEST_DATASET,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=6,
    pin_memory=True
)

# Instantiate FrustumPointNet
model = FrustumPointNet(
    n_classes=n_classes,
    n_channel=3,
    return_preds=True # this is important in order to calculate box IOUs
)

# Load pretrained model
pretrained_model = torch.load(model_path)
model.load_state_dict(pretrained_model["model_state_dict"])

# Send it to the gpu and set it to evaluation mode
model = model.cuda()
model = model.eval()

# Make batch predictions and store information on the True and False Positives:

nd = len(TEST_DATASET)
current_idx = 0

npos = 0

TP2D_25 = np.zeros(nd)
FP2D_25 = np.zeros(nd)

TP2D_50 = np.zeros(nd)
FP2D_50 = np.zeros(nd)

TP2D_70 = np.zeros(nd)
FP2D_70 = np.zeros(nd)

TP3D_25 = np.zeros(nd)
FP3D_25 = np.zeros(nd)

TP3D_50 = np.zeros(nd)
FP3D_50 = np.zeros(nd)

TP3D_70 = np.zeros(nd)
FP3D_70 = np.zeros(nd)


for i, data_dicts in tqdm(enumerate(test_dataloader),\
        total=len(test_dataloader), smoothing=0.9):
        #n_batches += 1

        data_dicts_var = {key: value.cuda() for key, value in data_dicts.items()}

        batch_data = data_dicts_var["point_cloud"]
        batch_label = data_dicts_var["seg"]
        batch_center = data_dicts_var["box3d_center"]
        batch_hclass = data_dicts_var["angle_class"]
        batch_hres = data_dicts_var["angle_residual"]
        batch_sclass = data_dicts_var["size_class"]
        batch_sres = data_dicts_var["size_residual"]
        batch_rot_angle = data_dicts_var["rot_angle"]
        batch_one_hot_vec = data_dicts_var["one_hot"]


        # make predictions
        losses, metrics, preds = model(data_dicts_var)

        iou2ds, iou3ds = compute_box3d_iou(
            preds["box3d_center"],
            preds["heading_scores"],
            preds["heading_residual"],
            preds["size_scores"],
            preds["size_residual"],
            preds["box3d_center_label"],
            preds["heading_class_label"],
            preds["heading_residual_label"],
            preds["size_class_label"],
            preds["size_residual_label"],
            )

        for i in range(len(iou2ds)):

            # 2D
            if iou2ds[i] >= 0.25:
                TP2D_25[npos] = 1
            else:
                FP2D_25[npos] = 1

            if iou2ds[i] >= 0.5:
                TP2D_50[npos] = 1
            else:
                FP2D_50[npos] = 1
            
            if iou2ds[i] >= 0.7:
                TP2D_70[npos] = 1
            else:
                FP2D_70[npos] = 1

            # 3D
            if iou3ds[i] >= 0.25:
                TP3D_25[npos] = 1
            else:
                FP3D_25[npos] = 1

            if iou3ds[i] >= 0.5:
                TP3D_50[npos] = 1
            else:
                FP3D_50[npos] = 1
            
            if iou3ds[i] >= 0.7:
                TP3D_70[npos] = 1
            else:
                FP3D_70[npos] = 1
            
            npos += 1


# Calculate 3D Box Pred. Accuracy, Precision, Recall and Average Precision for 
# one class using the get_ap() function based on  the eval_det_cls() function
# defined in https://github.com/charlesq34/frustum-pointnets/blob/master/sunrgbd/sunrgbd_detection/eval_det.py
# in order to calcluate recall, precision, and AP for object detection for a single class

# 2D AP at IOU threshold of 25: (the code is similar for other IOU thresholds)
acc, prec, rec, ap = get_ap(TP=TP2D_25, FP=FP2D_25, data_len=len(TEST_DATASET))

print("Average precision: ", ap)
print("Accuracy: ", acc)

# Plot the precision-recall curve using matplotlib:
#prec_ = sorted(list(prec), reverse=True)
#plt.plot(rec, prec_)
#plt.ylim([0, 1.1])
#plt.show()