""" Script to train Frustum PointNet models using pre-processed datasets."""

# Partially based on the following works:
# (1) Charles R. Qi (https://github.com/charlesq34/frustum-pointnets) 
#       The main author of the Frustum PointNet paper. The source code was shared with Apache Licence v2.0.
# (ii) Siming Fan (https://github.com/simon3dv/frustum_pointnets_pytorch) 
#       Permission granted by the author of the source code in written form.

import os
import sys
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
from datasets.data_object import FrustumDataset
from models.model_utils import test_one_epoch

# Hyperparameters
MAX_EPOCH = 200
NUM_POINT = 1024
BATCH_SIZE = 32
MIN_LR = 1e-5

# Dir of train and files
TRAIN_FILE = "/home/hydrogen/data/kitti/processed/val_5/transfer_train_Car_train_5.pickle"
TEST_FILE = "/home/hydrogen/data/kitti/processed/val_5/transfer_val_Car_val_5.pickle"

# Set params to update learning rate
BASE_LR = 0.001
WEIGHT_DECAY = 0.0001
LR_STEPS = 20
GAMMA = 0.7

# No. of cpu cores to assign to the data loader
NUM_WORKERS = 6

# Function to print the log and write the log to a text file
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

# Create log dirs and files
strtime = time.strftime('%Y%m%d-%H%M%S',time.localtime(time.time()))
strtime = strtime[4:].replace('-', '_')
OUTPUT_DIR = 'logs'
NAME = '_'.join(OUTPUT_DIR.split('/')) + '_' + strtime
print("\nLog dir: ", NAME)
LOG_DIR = OUTPUT_DIR + '/' + NAME
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')

# Track how long the entire process takes
start = time.perf_counter()


# Load the datasets
TRAIN_DATASET = FrustumDataset(
    npoints=NUM_POINT,
    rotate_to_center=True,
    random_flip=True,
    random_shift=True,
    processed_pickle_path=TRAIN_FILE,
    one_hot=True
)

TEST_DATASET = FrustumDataset(
    npoints=NUM_POINT,
    rotate_to_center=True,
    processed_pickle_path=TEST_FILE,
    one_hot=True
)

train_dataloader = DataLoader(
    TRAIN_DATASET,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_dataloader = DataLoader(
    TEST_DATASET,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# Instantiate the Frustum PointNet Model
model = FrustumPointNet(n_classes=1, n_channel=3).cuda()

optimizer = optim.Adam(
            model.parameters(), lr=BASE_LR,
            betas=(0.9, 0.999), eps=1e-08,
            weight_decay=WEIGHT_DECAY)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEPS, gamma=GAMMA)

# Track the best mean iou3d at 50% or 70% threshold to the corresponding epoch snapshot
num_batch = int(len(TRAIN_DATASET) / BATCH_SIZE)
best_iou3d_70 = 0.0
best_iou3d_50 = 0.0
best_epoch = 1
best_file = ""

# Start the training process
for epoch in range(MAX_EPOCH):
    log_string('\n**** EPOCH %03d ****' % (epoch + 1))
    sys.stdout.flush()
    log_string('Epoch %d/%s:' % (epoch + 1, MAX_EPOCH))
    
    time1 = time.perf_counter()

    # Store batch losses
    train_losses = {
        'total_loss': 0.0,
        'mask_loss': 0.0, #fpointnet
        'heading_class_loss': 0.0,
        'size_class_loss': 0.0,
        'heading_residual_normalized_loss': 0.0,
        'size_residual_normalized_loss': 0.0,
        'stage1_center_loss': 0.0,
        'corners_loss': 0.0
    }
    train_metrics = {
        'seg_acc': 0.0, #fpointnet
        'iou2d': 0.0,
        'iou3d': 0.0,
        'iou3d_0.5': 0.0,
        'iou3d_0.7': 0.0,
    }

    n_batches = 0
    # Start training the batches
    for i, data_dicts in tqdm(enumerate(train_dataloader),\
        total=len(train_dataloader), smoothing=0.9):
        n_batches += 1

        # the data dictionary from the data loader
        data_dicts_var = {key: value.cuda() for key, value in data_dicts.items()}
        
        # set the model to training mode
        optimizer.zero_grad()
        model = model.train()

        # Forward pass (also calculates the losses and metrics)
        losses, metrics = model(data_dicts_var)
        total_loss = losses['total_loss']
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()

        # Store the batch losses
        for key in train_losses.keys():
            if key in losses.keys():
                train_losses[key] += losses[key].detach().item()
        for key in train_metrics.keys():
            if key in metrics.keys():
                train_metrics[key] += metrics[key]

    # Compute the average loss per batch
    for key in train_losses.keys():
        train_losses[key] /= n_batches
    for key in train_metrics.keys():
        train_metrics[key] /= n_batches

    log_string('[%d: %d/%d] train' % (epoch + 1, i, len(train_dataloader)))

    for key, value in train_losses.items():
        log_string(str(key)+':'+"%.6f"%(value))
    for key, value in train_metrics.items():
        log_string(str(key)+':'+"%.6f"%(value))

    time2 = time.perf_counter()
    print('\ntraining time:%.2f s/batch'%((time2-time1)/n_batches))
    
    # Validation step for the epoch
    test_losses, test_metrics = test_one_epoch(model, test_dataloader)

    log_string('[%d: %d/%d] %s'%(epoch + 1, i, len(train_dataloader), 'test'))

    # Log the validation loss / metric figures
    for key, value in test_losses.items():
        log_string(str(key)+':'+"%.6f"%(value))
    for key, value in test_metrics.items():
        log_string(str(key)+':'+"%.6f"%(value))
    
    # Update the learning rate
    scheduler.step()

    # Update the learning rate
    if MIN_LR > 0:
        if scheduler.get_lr()[0] < MIN_LR:
            for param_group in optimizer.param_groups:
                param_group['lr'] = MIN_LR
    log_string("learning rate: {:.6f}".format(optimizer.param_groups[0]['lr']))

    # Store the best model based on mean iou3d at 70% threshold
    if test_metrics['iou3d_0.5'] >= best_iou3d_50:
        #best_iou3d_70 = test_metrics['iou3d_0.7']
        best_iou3d_50 = test_metrics['iou3d_0.5']
        best_epoch = epoch + 1

        savepath = LOG_DIR + '/acc%.3f-epoch%03d.pth' % \
                        (test_metrics['iou3d_0.5'], epoch)
        log_string('save to:'+str(savepath))

        best_file = savepath
        state = {
            'epoch': epoch + 1,
            'train_iou3d_0.5': train_metrics['iou3d_0.5'],
            'test_iou3d_0.5': test_metrics['iou3d_0.5'],
            #'train_iou3d_0.7': train_metrics['iou3d_0.7'],
            #'test_iou3d_0.7': test_metrics['iou3d_0.7'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)
        log_string('Saved model to %s'%savepath)

        log_string('Best Test acc: %f(Epoch %d)' % (best_iou3d_50, best_epoch))

log_string("Time {} hours".format(float(time.perf_counter() - start)/3600))
