"""Dataset class and other utilities for the data object used to train Frustum PointNet."""

# Based on the following works:
# (1) Charles R. Qi (https://github.com/charlesq34/frustum-pointnets) 
#       The main author of the Frustum PointNet paper. The source code was shared with Apache Licence v2.0.
# (ii) Siming Fan (https://github.com/simon3dv/frustum_pointnets_pytorch) 
#       Permission granted by the author of the source code in written form.

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

import numpy as np

import _pickle as pickle

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.box_util import box3d_iou

NUM_HEADING_BIN = 4 # 4 per object class
NUM_SIZE_CLUSTER = 1 # 1 per object class
NUM_OBJECT_POINT = 1024 # 1024 points

INTEREST_OBJECT = "car"


if INTEREST_OBJECT == "car":
    g_type2class={'Car':0}
    g_class2type = {g_type2class[t]:t for t in g_type2class}

elif INTEREST_OBJECT == "chair":
    g_type2class={'chair':0}
    g_class2type = {g_type2class[t]:t for t in g_type2class}


g_class2type = {g_type2class[t]:t for t in g_type2class}
g_type_mean_size = {
    'Car': np.array([3.88311640418, 1.62856739989, 1.52563191462]),
    'table': np.array([0.791118, 1.279516, 0.718182]), # array([0.91167996, 1.34810616, 0.70753093])
    'chair': np.array([0.591958, 0.552978, 0.827272]),
    }


g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3)) # size clustrs

for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i,:] = g_type_mean_size[g_class2type[i]]


def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval],[sinval, cosval]])
    pc[:,[0,2]] = np.dot(pc[:,[0,2]], np.transpose(rotmat))
    return pc


def angle2class(angle, num_class):
    ''' Convert continuous angle to discrete class and residual.

    Input:
        angle: rad scalar, from 0-2pi (or -pi~pi), class center at
            0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        num_class: int scalar, number of classes N
    Output:
        class_id, int, among 0,1,...,N-1
        residual_angle: float, a number such that
            class*(2pi/N) + residual_angle = angle
    '''
    angle = angle%(2*np.pi)
    assert(angle>=0 and angle<=2*np.pi)
    angle_per_class = 2*np.pi/float(num_class)
    shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
    class_id = int(shifted_angle/angle_per_class)
    residual_angle = shifted_angle - \
        (class_id * angle_per_class + angle_per_class/2)
    return class_id, residual_angle


def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
    angle_per_class = 2*np.pi/float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle>np.pi:
        angle = angle - 2*np.pi
    return angle


def size2class(size, type_name):
    ''' Convert 3D bounding box size to template class and residual.
    todo (rqi): support multiple size clusters per type.
 
    Input:
        size: numpy array of shape (3,) for (l,w,h)
        type_name: string
    Output:
        size_class: int scalar
        size_residual: numpy array of shape (3,)
    '''
    size_class = g_type2class[type_name]
    size_residual = size - g_type_mean_size[type_name]
    return size_class, size_residual


def class2size(pred_cls, residual):
    ''' Inverse function to size2class. '''
    mean_size = g_type_mean_size[g_class2type[pred_cls]]
    residual = np.array(residual)
    mean_size = np.array(residual)
    print("class2size pred cls: {} mean_size:{}".format(pred_cls, mean_size))
    #mean_size = np.array([3.88311640418,1.62856739989,1.52563191462]) # only training kitti
    return mean_size + residual


def cart2hom(self, pts_3d):
    ''' Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    '''
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
    return pts_3d_hom


def project_rect_to_image(self, pts_3d_rect):
    ''' Input: nx3 points in rect camera coord.
        Output: nx2 points in image2 coord.
    '''
    pts_3d_rect = self.cart2hom(pts_3d_rect)
    pts_2d = np.dot(pts_3d_rect, np.transpose(self.P)) # nx3
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    return pts_2d[:,0:2]


class FrustumDataset(object):
    ''' Dataset class for Frustum PointNets training/evaluation.
    Load prepared KITTI data from pickled files, return individual data element
    [optional] along with its annotations.
    '''
    def __init__(
        self, 
        npoints=1024,
        rotate_to_center=True,
        random_flip=False,
        random_shift=False,
        processed_pickle_path=None,
        one_hot=True,
        ):
        '''
        Input:
            npoints: int scalar, number of points for frustum point cloud.
            split: string, train or val
            random_flip: bool, in 50% randomly flip the point cloud
                in left and right (after the frustum rotation if any)
            random_shift: bool, if True randomly shift the point cloud
                back and forth by a random distance
            rotate_to_center: bool, whether to do frustum rotation
            overwritten_data_path: string, specify pickled file path.
                if None, use default path (with the split)
            from_rgb_detection: bool, if True we assume we do not have
                groundtruth, just return data elements.
            one_hot: bool, if True, return one hot vector
            gen_ref: bool, if True, generate ref data for fconvnet
        '''
        self.npoints = npoints
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        self.one_hot = one_hot

        # Read the data pickle (which is a pandas dataframe object)
        with open(processed_pickle_path, "rb") as file:
            df = pickle.load(file)

        self.id_list = np.array(df["id"])
        self.box2d_list = np.array(df["box_2d"])
        self.box3d_list = np.array(df["box_3d"])
        self.input_list = np.array(df["input"])
        self.label_list = np.array(df["seg_label"])
        self.type_list = np.array(df["object_class"])
        self.heading_list = np.array(df["heading_angle"])
        self.size_list = np.array(df["box_size"])
        self.frustum_angle_list = np.array(df["frustum_angle"])

        self.xmap = np.array([[j for i in range(1024)] for j in range(375)])
        self.ymap = np.array([[i for i in range(1024)] for j in range(375)])

    def __len__(self):
            return len(self.input_list)

    def __getitem__(self, index):
        ''' Get index-th element from the picked file dataset. '''

        #img_id = id_list[index]

        # ------------------------------ INPUTS ----------------------------
        rot_angle = self.get_center_view_rot_angle(index)
        #np.pi/2.0 + self.frustum_angle_list [index]float,[-pi/2,pi/2]

        box = self.box2d_list[index]

        # Compute one hot vector
        if self.one_hot:#True
            #cls_type = self.type_list[index]
            #assert(cls_type in ['Car', 'Pedestrian', 'Cyclist'])
            #one_hot_vec = np.zeros((3))
            #one_hot_vec[g_type2onehotclass[cls_type]] = 1
            one_hot_vec = np.array([1])

        # Get point cloud
        if self.rotate_to_center: # always True
            point_set = self.get_center_view_point_set(index)#(n,4) #pts after Frustum rotation
        else:
            point_set = self.input_list[index]

        # Use extra feature as channel

        #print("before:", point_set.shape)
        point_set = point_set[:,:3] # do not use reflection/rgb as channel

        #print("\nafter:",point_set.shape)

        # Resample
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]

        # ------------------------------ LABELS ----------------------------
        seg = self.label_list[index]
        seg = seg[choice] #(1024,) array([0., 1., 0., ..., 1., 1., 1.])

        # Get center point of 3D box
        if self.rotate_to_center:# True
            box3d_center = self.get_center_view_box3d_center(index) #array([ 0.07968819,  0.39      , 46.06915834])
        else:
            box3d_center = self.get_box3d_center(index)

        # Heading
        if self.rotate_to_center:#True
            heading_angle = self.heading_list[index] - rot_angle #-1.6480684951683866 #alpha
        else:
            heading_angle = self.heading_list[index]# rotation_y

        angle_class, angle_residual = angle2class(heading_angle, NUM_HEADING_BIN)

        # Size
        size_class, size_residual = size2class(self.size_list[index], self.type_list[index]) #5, array([0.25717603, 0.00293633, 0.12301873])

        # Data Augmentation
        if self.random_flip:
            # note: rot_angle won't be correct if we have random_flip
            # so do not use it in case of random flipping.
            if np.random.random() > 0.5: # 50% chance flipping
                point_set[:,0] *= -1
                box3d_center[0] *= -1
                heading_angle = np.pi - heading_angle

        if self.random_shift:
            dist = np.sqrt(np.sum(box3d_center[0]**2+box3d_center[1]**2))
            shift = np.clip(np.random.randn()*dist*0.05, dist*0.8, dist*1.2)
            point_set[:,2] += shift
            box3d_center[2] += shift

        box3d_size = self.size_list[index]

        data_inputs = {
            #'img_id': img_id,
            'point_cloud': torch.FloatTensor(point_set).transpose(1, 0),
            'rot_angle': torch.FloatTensor([rot_angle]),
            'box3d_center': torch.FloatTensor(box3d_center),
            'seg': seg,
            'size_class':torch.LongTensor([size_class]),
            'size_residual':torch.FloatTensor([size_residual]),
            'angle_class':torch.LongTensor([angle_class]),
            'angle_residual':torch.FloatTensor([angle_residual])
        }

        if self.one_hot:
            data_inputs.update({'one_hot': torch.FloatTensor(one_hot_vec)})

        return data_inputs

    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi/2.0 + self.frustum_angle_list[index]

    def get_box3d_center(self, index):
        ''' Get the center (XYZ) of 3D bounding box. '''
        box3d_center = (self.box3d_list[index][0,:] + \
            self.box3d_list[index][6,:])/2.0
        return box3d_center

    def get_center_view_box3d_center(self, index):
        ''' Frustum rotation of 3D bounding box center. '''
        box3d_center = (self.box3d_list[index][0,:] + \
            self.box3d_list[index][6,:])/2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center,0), \
            self.get_center_view_rot_angle(index)).squeeze()
        
    def get_center_view_box3d(self, index):
        ''' Frustum rotation of 3D bounding box corners. '''
        box3d = self.box3d_list[index]
        box3d_center_view = np.copy(box3d)
        return rotate_pc_along_y(box3d_center_view, \
            self.get_center_view_rot_angle(index))

    def get_center_view_point_set(self, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])
        return rotate_pc_along_y(point_set, \
            self.get_center_view_rot_angle(index))

    def generate_ref(self, box, P):
        cx, cy = (box[0] + box[2]) / 2., (box[1] + box[3]) / 2.,#678.73,206.76

        xyz1 = np.zeros((len(self.z1), 3))
        xyz1[:, 0] = cx
        xyz1[:, 1] = cy
        xyz1[:, 2] = self.z1
        xyz1_rect = project_image_to_rect(xyz1, P)

        xyz2 = np.zeros((len(self.z2), 3))
        xyz2[:, 0] = cx
        xyz2[:, 1] = cy
        xyz2[:, 2] = self.z2
        xyz2_rect = project_image_to_rect(xyz2, P)

        xyz3 = np.zeros((len(self.z3), 3))
        xyz3[:, 0] = cx
        xyz3[:, 1] = cy
        xyz3[:, 2] = self.z3
        xyz3_rect = project_image_to_rect(xyz3, P)

        xyz4 = np.zeros((len(self.z4), 3))
        xyz4[:, 0] = cx
        xyz4[:, 1] = cy
        xyz4[:, 2] = self.z4
        xyz4_rect = project_image_to_rect(xyz4, P)

        return xyz1_rect, xyz2_rect, xyz3_rect, xyz4_rect

    def generate_ref_labels(self, center, dimension, angle, ref_xyz, P):
        box_corner1 = get_3d_box(dimension * 0.5, angle, center)
        box_corner2 = get_3d_box(dimension, angle, center)

        labels = np.zeros(len(ref_xyz))#(140,)
        _, inside1 = extract_pc_in_box3d(ref_xyz, box_corner1)#(140,)
        _, inside2 = extract_pc_in_box3d(ref_xyz, box_corner2)#(140,)

        labels[inside2] = -1
        labels[inside1] = 1
        # dis = np.sqrt(((ref_xyz - center)**2).sum(1))
        # print(dis.min())
        if inside1.sum() == 0:
            dis = np.sqrt(((ref_xyz - center) ** 2).sum(1))
            argmin = np.argmin(dis)
            labels[argmin] = 1

        return labels

    def get_center_view(self, point_set, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(point_set)
        return rotate_pc_along_y(point_set,
                                 self.get_center_view_rot_angle(index))



def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (l,w,h)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def compute_box3d_iou(center_pred,
                      heading_logits, heading_residual,
                      size_logits, size_residual,
                      center_label,
                      heading_class_label, heading_residual_label,
                      size_class_label, size_residual_label):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.

    Inputs:
        center_pred: (B,3)
        heading_logits: (B,NUM_HEADING_BIN)
        heading_residual: (B,NUM_HEADING_BIN)
        size_logits: (B,NUM_SIZE_CLUSTER)
        size_residual: (B,NUM_SIZE_CLUSTER,3)
        center_label: (B,3)
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    batch_size = heading_logits.shape[0]
    heading_class = np.argmax(heading_logits, 1) # B
    heading_residual = np.array([heading_residual[i,heading_class[i]] \
        for i in range(batch_size)]) # B,
    size_class = np.argmax(size_logits, 1) # B
    size_residual = np.vstack([size_residual[i,size_class[i],:] \
        for i in range(batch_size)])

    iou2d_list = [] 
    iou3d_list = [] 
    for i in range(batch_size):
        heading_angle = class2angle(heading_class[i],
            heading_residual[i], NUM_HEADING_BIN)
        box_size = class2size(size_class[i], size_residual[i])
        corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])

        heading_angle_label = class2angle(heading_class_label[i],
            heading_residual_label[i], NUM_HEADING_BIN)
        box_size_label = class2size(size_class_label[i], size_residual_label[i])
        corners_3d_label = get_3d_box(box_size_label,
            heading_angle_label, center_label[i])

        iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label) 
        iou3d_list.append(iou_3d)
        iou2d_list.append(iou_2d)
    return np.array(iou2d_list, dtype=np.float32), \
        np.array(iou3d_list, dtype=np.float32)


def from_prediction_to_label_format(center, angle_class, angle_res,\
                                    size_class, size_res, rot_angle):
    ''' Convert predicted box parameters to label format. '''
    l,w,h = class2size(size_class, size_res)
    ry = class2angle(angle_class, angle_res, NUM_HEADING_BIN) + rot_angle
    tx,ty,tz = rotate_pc_along_y(np.expand_dims(center,0),-rot_angle).squeeze()
    ty += h/2.0
    return h,w,l,tx,ty,tz,ry

def project_image_to_rect(uv_depth, P):
    '''
    :param box: nx3 first two channels are uv in image coord, 3rd channel
                is depth in rect camera coord
    :param P: 3x3 or 3x4
    :return: nx3 points in rect camera coord
    '''
    c_u = P[0,2]
    c_v = P[1,2]
    f_u = P[0, 0]
    f_v = P[1, 1]
    if P.shape[1] == 4:
        b_x = P[0, 3] / (-f_u)  # relative
        b_y = P[1, 3] / (-f_v)
    else:
        b_x = 0
        b_y = 0
    n = uv_depth.shape[0]
    x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u + b_x
    y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v + b_y
    pts_3d_rect = np.zeros((n, 3), dtype=uv_depth.dtype)
    pts_3d_rect[:, 0] = x
    pts_3d_rect[:, 1] = y
    pts_3d_rect[:, 2] = uv_depth[:, 2]
    return pts_3d_rect

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

