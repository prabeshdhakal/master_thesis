"""Version 1 of the Frustum PointNet model used to train models for 3D object detection."""

# Partially based on the following works:
# (1) Charles R. Qi (https://github.com/charlesq34/frustum-pointnets) 
#       The main author of the Frustum PointNet paper. The source code was shared with Apache Licence v2.0.
# (ii) Siming Fan (https://github.com/simon3dv/frustum_pointnets_pytorch) 
#       Permission granted by the author of the source code in written form.

import os
import sys
sys.path.append(os.getcwd()+"/models")

from torch.nn import init

from torch.nn import init

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from provider import compute_box3d_iou
from model_utils import FrustumPointNetLoss, point_cloud_masking, parse_output_to_tensors


NUM_HEADING_BIN = 4
NUM_SIZE_CLUSTER = 1 # one size cluster per object class


class InstanceSegNet(nn.Module):
    def __init__(self, n_classes=1, n_channel=3):
        """
        3D Instance Segmentation Network for Frustum PointNet v1.
        :param n_classes: number of classes of objects that the net is being trained on
        :param n_channel: number of channels for each point (x, y, z) NOTE that reflectance is ignored in this
        implementation
        """

        super(InstanceSegNet, self).__init__()

        self.conv1 = nn.Conv1d(n_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.conv6 = nn.Conv1d(1088 + n_classes, 512, 1)  # shape: (1024+64, 512, 1)
        self.conv7 = nn.Conv1d(512, 256, 1)
        self.conv8 = nn.Conv1d(256, 128, 1)
        self.conv9 = nn.Conv1d(128, 128, 1)
        self.conv10 = nn.Conv1d(128, 2, 1)

        self.dropout = nn.Dropout(p=0.5)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(128)
        self.bn9 = nn.BatchNorm1d(128)

    def forward(self, points, one_hot_vec):
        """
        3D Instance Segmentation Network for Frustum PointNet
        :param points: [batch_size, 3, n] input points from each frustum point cloud; reflectance is ignored in this
        implementation
        :param one_hot_vec: one hot vector for the classes being detected (see original implementation)
        :return: [batch_size, n, 2] logits for points belonging to the object of interest vs. background
        """

        batch_size = points.size()[0]
        num_points = points.size()[2]

        out = F.relu(self.bn1(self.conv1(points)))  # shape:(batch_size, 64, n)
        out = F.relu(self.bn2(self.conv2(out)))  # shape:(batch_size, 64, n)
        point_features = out
        out = F.relu(self.bn3(self.conv3(out)))  # shape:(batch_size, 64, n)
        out = F.relu(self.bn4(self.conv4(out)))  # shape:(batch_size, 128, n)
        out = F.relu(self.bn5(self.conv5(out)))  # shape:(batch_size, 1024, n)

        global_feature = torch.max(out, 2, keepdim=True)[0]  # shape:()
        one_hot_vec = one_hot_vec.view(batch_size, -1, 1)  # shape:()
        global_feature = torch.cat([global_feature, one_hot_vec], 1)  # shape:(batch_size, 1024 + n_classes, 1)
        global_feature_repeat = global_feature.view(batch_size, -1, 1).repeat(1, 1, num_points)
        concatenated_feature = torch.cat([point_features, global_feature_repeat], 1)

        out = F.relu(self.bn6(self.conv6(concatenated_feature)))  # shape:(batch_size, 512, n)
        out = F.relu(self.bn7(self.conv7(out)))  # shape:(batch_size, 256, n)
        out = F.relu(self.bn8(self.conv8(out)))  # shape:(batch_size, 128, n)
        out = F.relu(self.bn9(self.conv9(out)))  # shape:(batch_size, 128, n)
        out = self.dropout(out)
        out = self.conv10(out)  # shape:(batch_size, 2, n)

        out = out.transpose(2, 1).contiguous()

        return out  # logits (softmax implemented in the loss calculation stage)


class TNet(nn.Module):

    def __init__(self, n_classes=1):
        super(TNet, self).__init__()

        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)

        self.fc1 = nn.Linear(256 + n_classes, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

    def forward(self, points, one_hot_vec):
        batch_size = points.size()[0]

        out = F.relu(self.bn1(self.conv1(points)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = torch.max(out, 2)[0]

        one_hot_vec = one_hot_vec.view(batch_size, -1)  # shape: (batch_size, 1)
        out = torch.cat([out, one_hot_vec], 1)

        out = F.relu(self.bn4(self.fc1(out)))
        out = F.relu(self.bn5(self.fc2(out)))
        out = self.fc3(out)

        return out


class BBoxNet(nn.Module):
    def __init__(self, n_classes=1, n_channel=3):
        super(BBoxNet, self).__init__()

        self.conv1 = nn.Conv1d(n_channel, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)

        self.fc1 = nn.Linear(512 + n_classes, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3 + (2 * NUM_HEADING_BIN) + (4 * NUM_SIZE_CLUSTER))

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(256)

    def forward(self, points, one_hot_vec):
        batch_size = points.size()[0]

        out = F.relu(self.bn1(self.conv1(points)))  # shape: (batch_size, 128, n)
        out = F.relu(self.bn2(self.conv2(out)))  # shape: (batch_size, 128, n)
        out = F.relu(self.bn3(self.conv3(out)))  # shape: (batch_size, 256, n)
        out = F.relu(self.bn4(self.conv4(out)))  # shape: (batch_size, 512, n)
        global_feature = torch.max(out, 2, keepdim=False)[0]  # shape: (batch_size, 512)

        one_hot_vec = one_hot_vec.view(batch_size, -1)  # shape: (batch_size, n_classes)
        global_feature = torch.cat([global_feature, one_hot_vec], 1)  # shape: (batch_size, 512 + n_classes)

        out = F.relu(self.bn5(self.fc1(global_feature)))  # shape: (batch_size, 512)
        out = F.relu(self.bn6(self.fc2(out)))  # shape: (batch_size, 256)
        out = self.fc3(out)  # shape: (batch_size, 3+ 4*NUM_SIZE_CLUSTER + 3*NUM_HEADING_BIN)

        return out


class FrustumPointNet(nn.Module):
    def __init__(self, n_classes=1, n_channel=3, return_preds=False):
        super(FrustumPointNet, self).__init__()
        self.n_classes = n_classes
        self.n_channel = n_channel
        self.return_preds = return_preds

        self.instance_seg_net = InstanceSegNet(self.n_classes, self.n_channel)
        self.t_net = TNet(self.n_classes)
        self.bbox_net = BBoxNet(self.n_classes)

        self.FPNLoss = FrustumPointNetLoss()

    def forward(self, data_dicts):
        img_id = data_dicts.get("id")
        point_cloud = data_dicts.get('point_cloud')
        point_cloud = point_cloud[:, :self.n_channel, :]
        one_hot = data_dicts.get('one_hot')
        bs = point_cloud.shape[0]

        seg_label = data_dicts.get('seg')
        box3d_center_label = data_dicts.get('box3d_center')
        size_class_label = data_dicts.get('size_class')
        size_residual_label = data_dicts.get('size_residual')
        heading_class_label = data_dicts.get('angle_class')
        heading_residual_label = data_dicts.get('angle_residual')

        # 3D Instance Segmentation PointNet
        logits = self.instance_seg_net(point_cloud, one_hot)

        # Mask Point Centroid
        object_pts_xyz, mask_xyz_mean, mask = \
                 point_cloud_masking(point_cloud, logits)

        # T-Net
        object_pts_xyz = object_pts_xyz.cuda()
        center_delta = self.t_net(object_pts_xyz,one_hot)
        stage1_center = center_delta + mask_xyz_mean

        object_pts_xyz_new = object_pts_xyz - \
                    center_delta.view(center_delta.shape[0],-1,1).repeat(1,1,object_pts_xyz.shape[-1])

        # 3D Box Estimation
        box_pred = self.bbox_net(object_pts_xyz_new, one_hot)

        center_boxnet, \
        heading_scores, heading_residual_normalized, heading_residual, \
        size_scores, size_residual_normalized, size_residual = \
                parse_output_to_tensors(box_pred, logits, mask, stage1_center)

        box3d_center = center_boxnet + stage1_center

        # Calculate loss values
        losses = self.FPNLoss(logits, seg_label, \
            box3d_center, box3d_center_label, stage1_center, \
            heading_scores, heading_residual_normalized, \
            heading_residual, \
            heading_class_label, heading_residual_label, \
            size_scores, size_residual_normalized, \
            size_residual, \
            size_class_label, size_residual_label)

        for key in losses.keys():
            losses[key] = losses[key]/bs


        with torch.no_grad():

            seg_correct = torch.argmax(logits.detach().cpu(), 2).eq(seg_label.detach().cpu()).numpy()
            seg_accuracy = np.sum(seg_correct) / float(point_cloud.shape[-1])

            # Calculate the top-view and 3D box IOU for the boxes
            iou2ds, iou3ds = compute_box3d_iou( \
                box3d_center.detach().cpu().numpy(),
                heading_scores.detach().cpu().numpy(),
                heading_residual.detach().cpu().numpy(),
                size_scores.detach().cpu().numpy(),
                size_residual.detach().cpu().numpy(),
                box3d_center_label.detach().cpu().numpy(),
                heading_class_label.detach().cpu().numpy(),
                heading_residual_label.detach().cpu().numpy(),
                size_class_label.detach().cpu().numpy(),
                size_residual_label.detach().cpu().numpy())

        metrics = {
            'seg_acc': seg_accuracy,
            'iou2d': iou2ds.mean(),
            'iou3d': iou3ds.mean(),
            'iou3d_0.5': np.sum(iou3ds >= 0.5) / bs,
            'iou3d_0.7': np.sum(iou3ds >= 0.7) / bs
        }

        if self.return_preds:
            preds = {
                "img_id": img_id,
                "box3d_center": box3d_center.detach().cpu().numpy(),
                "heading_scores": heading_scores.detach().cpu().numpy(),
                "heading_residual": heading_residual.detach().cpu().numpy(),
                "size_scores": size_scores.detach().cpu().numpy(),
                "size_residual": size_residual.detach().cpu().numpy(),
                "box3d_center_label": box3d_center_label.detach().cpu().numpy(),
                "heading_class_label": heading_class_label.detach().cpu().numpy(),
                "heading_residual_label": heading_residual_label.detach().cpu().numpy(),
                "size_class_label": size_class_label.detach().cpu().numpy(),
                "size_residual_label": size_residual_label.detach().cpu().numpy()
            }

            return losses, metrics, preds
        else:
            return losses, metrics

