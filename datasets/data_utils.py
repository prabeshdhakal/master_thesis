import gzip
import pickle

import pandas as pd
import numpy as np



def create_transfer_data_splits(train, val, val_prop, dataset="kitti"):

    """
    Create Target Dataset at different sample sizes.
    """


    # Read the data pickle (which is a pandas dataframe object)
    with open(train, "rb") as file:
        train_df = pickle.load(file)
        
    with open(val, "rb") as file:
        val_df = pickle.load(file)

    # Total (train+val) size for transfer = Val size of the source network (in this case SUN)
    if dataset == "sunrgbd":
        transfer_train_size_val_100 = 9120
        transfer_val_size_val_100 = 1632
    else:
        transfer_train_size_val_100 = 6208  # kitti dataset
        transfer_val_size_val_100 = 1088  # kitti dataset

    transfer_train_size = int(transfer_train_size_val_100 * val_prop / 100)
    transfer_val_size = int(transfer_val_size_val_100 * val_prop / 100)

    save_path = "val_{}".format(val_prop)

    save_file_train = save_path + "/transfer_train_{}.pickle".format(val_prop)
    save_file_val = save_path + "/transfer_val_{}.pickle".format(val_prop)


    transfer_train = train_df.sample(transfer_train_size, replace=False)
    transfer_val = val_df.sample(transfer_val_size, replace=False)

    print("Size prop. of SN val size: ", val_prop)
    print("Transfer train size: ", transfer_train.shape)
    print("Transfer val size:", transfer_val.shape)

    with open(save_file_train, "wb") as file:
        pickle.dump(transfer_train, file)

    with open(save_file_val, "wb") as file:
        pickle.dump(transfer_val, file)



def load_zipped_pickle(filename):
    """
    Load the SUNRGBD zipped pickle train and val files created using code
    provided in https://github.com/charlesq34/frustum-pointnets/.
    :param filename: SUN RGB-D zipped pickle filename
    :return: loaded zipped pickle object
    """
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object


def convert_pickle_to_df(pickle_path, target_object="table"):
    """
    Read the zipped pickle file, then convert to a pandas df, and select the
    target object.
    :param pickle_path: path to the pickle file
    :return: pandas dataframe of just the rows with the target object.
    """
    id_list, box2d_list, box3d_list, input_list, label_list, type_list,\
        heading_list, size_list, frustum_angle_list = load_zipped_pickle(pickle_path)

    df_ = pd.DataFrame(
        {
            "id": id_list,
            "box_2d": box2d_list,
            "box_3d": box3d_list,
            "input": input_list,  # input label
            "seg_label": label_list,
            "object_class": type_list,
            "heading_angle": heading_list,
            "box_size": size_list,
            "frustum_angle": frustum_angle_list
        }
    )

    df = df_[df_["object_class"] == target_object]

    return df

def get_filtered_df_from_pickle(processed_kitti_path, target_object="Car", save_path="."):
    """
    Reads the processed KITTI pickle file provided by the authors of Frustum PointNet in
    https://github.com/charlesq34/frustum-pointnets/, converts it to pandas DataFrame object, filters
    out the objects other than the target objects, and saves the resulting dataframe
    as a pickle file.

    This step results to reduction in file size that needs to be read in during
    training and testing.
    
    :param processed_kitti_path: Path to the processed kitti train or val file from Qi et al.
    :param target_object: the object of interest; defaults to "Car"
    :param save_path: the location where the processed pandas df pickle needs to be saved
    """

    with open(processed_kitti_path, "rb") as file:
        id_list = pickle.load(file)
        box2d_list = pickle.load(file, encoding="latin1")
        box3d_list = pickle.load(file, encoding="latin1")
        input_list = pickle.load(file, encoding="latin1")
        label_list = pickle.load(file, encoding="latin1")
        type_list = pickle.load(file, encoding="latin1")
        heading_list = pickle.load(file, encoding="latin1")
        size_list = pickle.load(file, encoding="latin1")
        frustum_angle_list = pickle.load(file, encoding="latin1")

    df_raw = pd.DataFrame(
        {
            "id": id_list,
            "box_2d": box2d_list,
            "box_3d": box3d_list,
            "input": input_list,
            "seg_label": label_list,
            "object_class": type_list,
            "heading_angle": heading_list,
            "box_size": size_list,
            "frustum_angle": frustum_angle_list
        }
    )

    df = df_raw[df_raw["object_class"] == target_object]

    return df
