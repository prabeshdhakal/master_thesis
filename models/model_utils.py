"""Utilities required for training and evaluation of Frustum PointNet models."""

# Partially based on the following works:
# (1) Charles R. Qi (https://github.com/charlesq34/frustum-pointnets) 
#       The main author of the Frustum PointNet paper. The source code was shared with Apache Licence v2.0.
# (ii) Siming Fan (https://github.com/simon3dv/frustum_pointnets_pytorch) 
#       Permission granted by the author of the source code in written form.

import time
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.data_object import g_mean_size_arr

NUM_HEADING_BIN = 4
NUM_SIZE_CLUSTER = 1
NUM_OBJECT_POINT = 1024


def parse_output_to_tensors(box_pred, logits, mask, stage1_center):
    '''
    :param box_pred: (bs,59)
    :param logits: (bs,1024,2)
    :param mask: (bs,1024)
    :param stage1_center: (bs,3)
    :return:
        center_boxnet:(bs,3)
        heading_scores:(bs,12)
        heading_residual_normalized:(bs,12),-1 to 1
        heading_residual:(bs,12)
        size_scores:(bs,8)
        size_residual_normalized:(bs,8)
        size_residual:(bs,8)
    '''
    bs = box_pred.shape[0]
    # center
    center_boxnet = box_pred[:, :3]
    c = 3

    # heading
    heading_scores = box_pred[:, c:c + NUM_HEADING_BIN]
    c += NUM_HEADING_BIN
    heading_residual_normalized = \
        box_pred[:, c:c + NUM_HEADING_BIN]
    heading_residual = \
        heading_residual_normalized * (np.pi / NUM_HEADING_BIN)
    c += NUM_HEADING_BIN

    # size
    size_scores = box_pred[:, c:c + NUM_SIZE_CLUSTER]
    c += NUM_SIZE_CLUSTER
    size_residual_normalized = \
        box_pred[:, c:c + 3 * NUM_SIZE_CLUSTER].contiguous()
    size_residual_normalized = \
        size_residual_normalized.view(bs,NUM_SIZE_CLUSTER,3)
    size_residual = size_residual_normalized * \
                     torch.from_numpy(g_mean_size_arr).unsqueeze(0).repeat(bs,1,1).cuda()
    return center_boxnet,\
            heading_scores, heading_residual_normalized, heading_residual,\
            size_scores, size_residual_normalized, size_residual


def point_cloud_masking(pts, logits, xyz_only=True):
    '''
    :param pts: bs,c,n in frustum
    :param logits: bs,n,2
    :param xyz_only: bool
    :return:
    '''
    bs = pts.shape[0]
    n_pts = pts.shape[2]

    # Binary Classification for each point
    mask = logits[:, :, 0] < logits[:, :, 1]
    mask = mask.unsqueeze(1).float()
    mask_count = mask.sum(2,keepdim=True).repeat(1, 3, 1)
    
    pts_xyz = pts[:, :3, :]
    
    mask_xyz_mean = (mask.repeat(1, 3, 1) * pts_xyz).sum(2,keepdim=True)  
    mask_xyz_mean = mask_xyz_mean / torch.clamp(mask_count,min=1) 
    mask = mask.squeeze()
    
    pts_xyz_stage1 = pts_xyz - mask_xyz_mean.repeat(1, 1, n_pts)

    if xyz_only:
        pts_stage1 = pts_xyz_stage1
    else:
        pts_features = pts[:, 3:, :]
        pts_stage1 = torch.cat([pts_xyz_stage1, pts_features], dim=-1)
    object_pts, _ = gather_object_pts(pts_stage1, mask, NUM_OBJECT_POINT)
    object_pts = object_pts.reshape(bs, NUM_OBJECT_POINT, -1)
    object_pts = object_pts.float().view(bs,3,-1)
    
    return object_pts, mask_xyz_mean.squeeze(), mask


def gather_object_pts(pts, mask, n_pts=NUM_OBJECT_POINT):
    '''
    :param pts: (bs,c,1024)
    :param mask: (bs,1024)
    :param n_pts: max number of points of an object
    :return:
        object_pts:(bs,c,n_pts)
        indices:(bs,n_pts)
    '''
    bs = pts.shape[0]
    indices = torch.zeros((bs, n_pts), dtype=torch.int64)
    object_pts = torch.zeros((bs, pts.shape[1], n_pts))

    for i in range(bs):
        pos_indices = torch.where(mask[i, :] > 0.5)[0]
        if len(pos_indices) > 0:
            if len(pos_indices) > n_pts:
                choice = np.random.choice(len(pos_indices),
                                          n_pts, replace=False)
            else:
                choice = np.random.choice(len(pos_indices),
                                          n_pts - len(pos_indices), replace=True)
                choice = np.concatenate(
                    (np.arange(len(pos_indices)), choice))
            np.random.shuffle(choice)
            indices[i, :] = pos_indices[choice]
            object_pts[i,:,:] = pts[i,:,indices[i,:]]

    return object_pts, indices


def get_box3d_corners_helper(centers, headings, sizes):
    """ Input: (N,3), (N,), (N,3), Output: (N,8,3) """

    N = centers.shape[0]
    l = sizes[:,0].view(N,1)
    w = sizes[:,1].view(N,1)
    h = sizes[:,2].view(N,1)

    x_corners = torch.cat([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2], dim=1)
    y_corners = torch.cat([h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2], dim=1)
    z_corners = torch.cat([w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2], dim=1)
    corners = torch.cat([x_corners.view(N,1,8), y_corners.view(N,1,8),\
                            z_corners.view(N,1,8)], dim=1)

    c = torch.cos(headings).cuda()
    s = torch.sin(headings).cuda()
    ones = torch.ones([N], dtype=torch.float32).cuda()
    zeros = torch.zeros([N], dtype=torch.float32).cuda()
    row1 = torch.stack([c,zeros,s], dim=1)
    row2 = torch.stack([zeros,ones,zeros], dim=1)
    row3 = torch.stack([-s,zeros,c], dim=1)
    R = torch.cat([row1.view(N,1,3), row2.view(N,1,3), \
                      row3.view(N,1,3)], axis=1)

    corners_3d = torch.bmm(R, corners)
    corners_3d +=centers.view(N,3,1).repeat(1,1,8)
    corners_3d = torch.transpose(corners_3d,1,2)
    return corners_3d


def get_box3d_corners(center, heading_residual, size_residual):
    """
    Inputs:
        center: (bs,3)
        heading_residual: (bs,NH)
        size_residual: (bs,NS,3)
    Outputs:
        box3d_corners: (bs,NH,NS,8,3) tensor
    """
    bs = center.shape[0]
    heading_bin_centers = torch.from_numpy(\
            np.arange(0,2*np.pi,2*np.pi/NUM_HEADING_BIN)).float()
    headings = heading_residual + heading_bin_centers.view(1,-1).cuda()

    mean_sizes = torch.from_numpy(g_mean_size_arr).float().view(1,NUM_SIZE_CLUSTER,3).cuda()\
                 + size_residual.cuda()
    sizes = mean_sizes + size_residual
    sizes = sizes.view(bs,1,NUM_SIZE_CLUSTER,3)\
                .repeat(1,NUM_HEADING_BIN,1,1).float()
    headings = headings.view(bs,NUM_HEADING_BIN,1).repeat(1,1,NUM_SIZE_CLUSTER)
    centers = center.view(bs,1,1,3).repeat(1,NUM_HEADING_BIN, NUM_SIZE_CLUSTER,1)
    N = bs*NUM_HEADING_BIN*NUM_SIZE_CLUSTER

    corners_3d = get_box3d_corners_helper(centers.view(N,3),headings.view(N),\
                                    sizes.view(N,3))

    return corners_3d.view(bs, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8, 3) #[32, 4, 1, 8, 3]


def huber_loss(error, delta=1.0):
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return torch.mean(losses)


class FrustumPointNetLoss(nn.Module):
    def __init__(self):
        super(FrustumPointNetLoss, self).__init__()

    def forward(self, logits, mask_label, \
                center, center_label, stage1_center, \
                heading_scores, heading_residual_normalized, heading_residual, \
                heading_class_label, heading_residual_label, \
                size_scores,size_residual_normalized,size_residual,
                size_class_label,size_residual_label,
                corner_loss_weight=10.0, box_loss_weight=1.0):
        '''
        1.Instance Segmenation
        logits: torch.Size([32, 1024, 2]) torch.float32
        mask_label: [32, 1024]
        2.Center Loss (T-Net and Bounding Box PointNet)
        center: torch.Size([32, 3]) torch.float32
        stage1_center: torch.Size([32, 3]) torch.float32
        center_label:[32,3]
        3.Heading (Bounding Box PointNet)
        heading_scores: torch.Size([32, 12]) torch.float32
        heading_residual_snormalized: torch.Size([32, 12]) torch.float32
        heading_residual: torch.Size([32, 12]) torch.float32
        heading_class_label:(32)
        heading_residual_label:(32)
        4.Size (Bounding Box PointNet)
        size_scores: torch.Size([32, 8]) torch.float32
        size_residual_normalized: torch.Size([32, 8, 3]) torch.float32
        size_residual: torch.Size([32, 8, 3]) torch.float32
        size_class_label:(32)
        size_residual_label:(32,3)
        5.Corner (Bounding Box PointNet)
        6.Weight (multipliers to calculate the multi-task loss of the entire model)
        corner_loss_weight: float scalar
        box_loss_weight: float scalar

        '''
        # Batch size
        bs = logits.shape[0]

        # 3D Instance Segmentation PointNet Loss
        logits = F.log_softmax(logits.view(-1,2),dim=1)
        mask_label = mask_label.view(-1).long()

        mask_loss = F.nll_loss(logits, mask_label)

        # Center Regression Loss
        center_dist = torch.norm(center - center_label,dim=1)
        center_loss = huber_loss(center_dist, delta=2.0)
        stage1_center_dist = torch.norm(center - stage1_center,dim=1)
        stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)

        # Heading Loss
        heading_scores_soft = F.log_softmax(heading_scores, dim=1)
        heading_class_long = torch.reshape(heading_class_label, (-1, )).long()
        heading_class_loss = F.nll_loss(heading_scores_soft, \
                                        heading_class_long)
        
        hcls_onehot = torch.eye(NUM_HEADING_BIN)[heading_class_label.long()].cuda()
        
        heading_residual_normalized_label = \
            heading_residual_label / (np.pi / NUM_HEADING_BIN)
        heading_residual_normalized_dist = torch.sum( \
            heading_residual_normalized * hcls_onehot.float(), dim=1)
        
        heading_residual_normalized_loss = \
            huber_loss(heading_residual_normalized_dist -
                       heading_residual_normalized_label, delta=1.0)
        
        # Size loss
        size_scores_soft = F.log_softmax(size_scores,dim=1)
        size_class_label_long = torch.reshape(size_class_label, (-1, )).long()

        size_class_loss = F.nll_loss(size_scores_soft,size_class_label_long)

        scls_onehot = torch.eye(NUM_SIZE_CLUSTER)[size_class_label.long()].cuda()
        scls_onehot_repeat = scls_onehot.view(-1, NUM_SIZE_CLUSTER, 1).repeat(1, 1, 3)
        
        predicted_size_residual_normalized_dist = torch.sum( \
            size_residual_normalized * scls_onehot_repeat.cuda(), dim=1)

        mean_size_arr_expand = torch.from_numpy(g_mean_size_arr).float().cuda() \
            .view(1, NUM_SIZE_CLUSTER, 3)
        
        mean_size_label = torch.sum(scls_onehot_repeat * mean_size_arr_expand, dim=1)
        
        size_residual_label_normalized = size_residual_label / mean_size_label.cuda()

        size_normalized_dist = torch.norm(size_residual_label_normalized-\
                    predicted_size_residual_normalized_dist,dim=1)
        size_residual_normalized_loss = huber_loss(size_normalized_dist, delta=1.0)

        # Corner Loss
        corners_3d = get_box3d_corners(center,\
                    heading_residual,size_residual).cuda()
        gt_mask = hcls_onehot.view(bs,NUM_HEADING_BIN,1).repeat(1,1,NUM_SIZE_CLUSTER) * \
                  scls_onehot.view(bs,1,NUM_SIZE_CLUSTER).repeat(1,NUM_HEADING_BIN,1)
        
        corners_3d_pred = torch.sum(\
            gt_mask.view(bs,NUM_HEADING_BIN,NUM_SIZE_CLUSTER,1,1)\
            .float().cuda() * corners_3d,\
            dim=[1, 2])
        
        heading_bin_centers = torch.from_numpy(\
            np.arange(0, 2 * np.pi, 2 * np.pi / NUM_HEADING_BIN)).float().cuda()
        heading_label = heading_residual_label.view(bs,1) + \
                        heading_bin_centers.view(1,NUM_HEADING_BIN)

        heading_label = torch.sum(hcls_onehot.float() * heading_label, 1)
        mean_sizes = torch.from_numpy(g_mean_size_arr)\
                    .float().view(1,NUM_SIZE_CLUSTER,3).cuda()
        size_label = mean_sizes + \
                     size_residual_label.view(bs,1,3)
        size_label = torch.sum(\
           scls_onehot.view(bs,NUM_SIZE_CLUSTER,1).float() * size_label, axis=[1])

        heading_label_np = heading_label.data.cpu().numpy()
        heading_label_max_only = torch.from_numpy(heading_label_np.max(axis=1)).float().cuda()
        corners_3d_gt = get_box3d_corners_helper( \
            center_label, heading_label_max_only, size_label)
        corners_3d_gt_flip = get_box3d_corners_helper( \
            center_label, heading_label_max_only + np.pi, size_label)

        corners_dist = torch.min(torch.norm(corners_3d_pred - corners_3d_gt, dim=-1),
                                  torch.norm(corners_3d_pred - corners_3d_gt_flip, dim=-1))
        corners_loss = huber_loss(corners_dist, delta=1.0)

        # Weighted sum of all losses
        total_loss = mask_loss + box_loss_weight * (center_loss + \
                    heading_class_loss + size_class_loss + \
                    heading_residual_normalized_loss * 20 + \
                    size_residual_normalized_loss * 20 + \
                    stage1_center_loss + \
                    corner_loss_weight * corners_loss)

        losses = {
            'total_loss': total_loss,
            'mask_loss': mask_loss,
            'mask_loss': box_loss_weight * center_loss,
            'heading_class_loss': box_loss_weight * heading_class_loss,
            'size_class_loss': box_loss_weight * size_class_loss,
            'heading_residual_normalized_loss': box_loss_weight * heading_residual_normalized_loss * 20,
            'size_residual_normalized_loss': box_loss_weight * size_residual_normalized_loss * 20,
            'stage1_center_loss': box_loss_weight * size_residual_normalized_loss * 20,
            'corners_loss': box_loss_weight * corners_loss * corner_loss_weight,
        }

        return losses


def test_one_epoch(model, loader):
    time1 = time.perf_counter()

    test_losses = {
        'total_loss': 0.0,
        'mask_loss': 0.0,
        'heading_class_loss': 0.0,
        'size_class_loss': 0.0,
        'heading_residual_normalized_loss': 0.0,
        'size_residual_normalized_loss': 0.0,
        'stage1_center_loss': 0.0,
        'corners_loss': 0.0
    }
    test_metrics = {
        'seg_acc': 0.0,
        'iou2d': 0.0,
        'iou3d': 0.0,
        'iou3d_0.5': 0.0,
        'iou3d_0.7': 0.0,
    }

    n_batches = 0
    for i, data_dicts in tqdm(enumerate(loader), \
                              total=len(loader), smoothing=0.9):
        n_batches += 1

        data_dicts_var = {key: value.cuda() for key, value in data_dicts.items()}

        model = model.eval()

        with torch.no_grad():
            losses, metrics = model(data_dicts_var)

        for key in test_losses.keys():
            if key in losses.keys():
                test_losses[key] += losses[key].detach().item()
        for key in test_metrics.keys():
            if key in metrics.keys():
                test_metrics[key] += metrics[key]

    for key in test_losses.keys():
        test_losses[key] /= n_batches
    for key in test_metrics.keys():
        test_metrics[key] /= n_batches

    time2 = time.perf_counter()
    print('\ntest time:%.2f s/batch'%((time2-time1)/n_batches))
    return test_losses, test_metrics

