import pickle
import random
import pandas as pd
import numpy as np

from datasets.data_utils import load_zipped_pickle, convert_pickle_to_df
from datasets.data_utils import create_transfer_data_splits


# 0. Load pickle df that contain only the car (or chair) frustum point clouds
car_train = ""
car_val = ""

# 1. Create the training, validation, and hold-out sets for training the source model

# Read the data pickle (which is a pandas dataframe object)
with open(car_train, "rb") as file:
    kitti_train = pickle.load(file)
    
with open(car_val, "rb") as file:
    kitti_val = pickle.load(file)

# define size of the train, validation, and hold out sets
# the final sizes will differ from below because the frustum point clouds
# included in different sets are forced to be from independent scenes
TRAIN_LEN = 57472
VAL_LEN = 10752  
HOLD_OUT_LEN = 3584

# Combine the train and val sets
kitti = kitti_train.append(kitti_val)

# Shuffle the file ids randomly
id_set = list(set(kitti.id))
random.shuffle(id_set)

# get the list of column names
col_list = list(kitti.columns)

# empty data frames to append with rows later
train_df = pd.DataFrame(columns=col_list)
val_df = pd.DataFrame(columns=col_list)
hold_out_df = pd.DataFrame(columns=col_list)

# Hold out dataset
for idx in id_set:
    selected_rows = kitti[kitti["id"] == idx]
    hold_out_df = hold_out_df.append(selected_rows)
    # remove id from original id set (ids to ignore in the next step)
    id_set.remove(idx)
    # stop if HOLD_OUT_LEN has been reached
    if hold_out_df.shape[0] >= HOLD_OUT_LEN:
        break

# Validation dataset
for idx in id_set:
    selected_rows = kitti[kitti["id"] == idx]
    val_df = val_df.append(selected_rows)
    # remove id from original id set (ids to ignore in the next step)
    id_set.remove(idx)
    # stop if VAL_LEN has been reached
    if val_df.shape[0] >= VAL_LEN:
        break

# Train dataset
for idx in id_set:
    selected_rows = kitti[kitti["id"] == idx]
    train_df = train_df.append(selected_rows)

print("Train size:", train_df.shape)
print("Val size:", val_df.shape)
print("Hold out size:", hold_out_df.shape)

# Check that the model ids are really independent between the train/val/hold-out sets
#print("SAME IDX train==val:", np.sum(train_df["id"].eq(val_df["id"])))
#print("SAME IDX val==hold_out:", np.sum(hold_out_df["id"].eq(val_df["id"])))
#print("SAME IDX train==hold_out:", np.sum(train_df["id"].eq(hold_out_df["id"])))

# this is where you save the pickle files for use in training the models
destination_dir = ""

train_set_name = destination_dir + "train_Car.pickle"
val_set_name = destination_dir + "val_Car.pickle"
hold_out_set_name = destination_dir + "hold_out_Car.pickle"

# Save the data frames as pickle files
with open(train_set_name, "wb") as file:
    pickle.dump(train_df, file)

with open(val_set_name, "wb") as file:
    pickle.dump(val_df, file)

with open(hold_out_set_name, "wb") as file:
    pickle.dump(hold_out_df, file)


# 2. Process KITTI for Transfer Learning (Creation of the different Target Datasets)

# 2.1 Create the largest Target Dataset

# Files out of which the 
car_train = "" # training set created in step 1 above
car_val = "" # validation set created in step 1 above

# Read the data pickle (which is a pandas dataframe object)
with open(car_train, "rb") as file:
    kitti_train = pickle.load(file)
    
with open(car_val, "rb") as file:
    kitti_val = pickle.load(file)

# Total (train+val) size for transfer = Val size of the source network (in this case SUN)
transfer_train_size = 6208  # 194 batches
transfer_val_size = 1088  # 34 batches

kitti_transfer_train = kitti_train.sample(transfer_train_size, replace=False)
kitti_transfer_val = kitti_val.sample(transfer_val_size, replace=False)

# Save the training and validation sets of the Target Dataset

save_file_train = ""
save_file_val = ""

with open(save_file_train, "wb") as file:
    pickle.dump(kitti_transfer_train, file)

with open(save_file_val, "wb") as file:
    pickle.dump(kitti_transfer_val, file)

# 2.2 Create other target datasets 

# Other splits based on the SUN RGB-D Source Network
#                                       train	val
# KITTI  (car) – Transfer_val_100	    6208	1088    (Largest Target Dataset)
# KITTI  (car) – Transfer_val_75	    4656	816		
# KITTI  (car) – Transfer_val_50	    3104	544		
# KITTI  (car) – Transfer_val_25	    1552	272		
# KITTI  (car) – Transfer_val_10	    620.8	108.8		
# KITTI  (car) – Transfer_val_5	        310.4	54.4

# 75% of the SN val size
create_transfer_data_splits(car_train, car_val, val_prop = 75)

# 50% of the SN val size
create_transfer_data_splits(car_train, car_val, val_prop = 50)

# 25% of the SN val size
create_transfer_data_splits(car_train, car_val, val_prop = 25)

# 10% of the SN val size
create_transfer_data_splits(car_train, car_val, val_prop = 10)

# 5% of the SN val size
create_transfer_data_splits(car_train, car_val, val_prop = 5)

# 3. SUNRGBD

# 3.1 Load the SUNRGBD zipped pickle file created using the code
# provided in https://github.com/charlesq34/frustum-pointnets/

# path to the files:
processed_pickle_train_path = ".../train_1002_aug5x.zip.pickle"
processed_pickle_val_path = ".../val_1002.zip.pickle"

target_object = "chair" # chair, table

sunrgbd_train = convert_pickle_to_df(processed_pickle_train_path, target_object)
sunrgbd_val = convert_pickle_to_df(processed_pickle_val_path, target_object)

# verified that train set ids are not the same as val set ids
#  (files are independent)
print("SAME IDX train==val", np.sum(sunrgbd_train["id"].eq(sunrgbd_val["id"])))

# append the train pickle and val pickle together (will be split later)
sunrgbd = sunrgbd_train.append(sunrgbd_val)
print("SUNRGBD Shape with " + target_object, sunrgbd.shape)

# define size of the train, validation, and hold out sets
TRAIN_LEN = 38976
VAL_LEN = 7296
HOLD_OUT_LEN = 2432

# get the list of file ids and shuffle them
id_set = list(set(sunrgbd.id))
random.shuffle(id_set)

# get the list of column names
col_list = list(sunrgbd_train.columns)

# empty data frames to append with rows later
train_df = pd.DataFrame(columns=col_list)
val_df = pd.DataFrame(columns=col_list)
hold_out_df = pd.DataFrame(columns=col_list)

# Hold out dataset
for idx in id_set:
    selected_rows = sunrgbd[sunrgbd["id"] == idx]
    hold_out_df = hold_out_df.append(selected_rows)

    # remove id from original id set (ids to ignore in the next step)
    id_set.remove(idx)
    
    # stop if HOLD_OUT_LEN has been reached
    if hold_out_df.shape[0] >= HOLD_OUT_LEN:
        break

# Validation dataset
for idx in id_set:
    selected_rows = sunrgbd[sunrgbd["id"] == idx]
    val_df = val_df.append(selected_rows)

    # remove id from original id set (ids to ignore in the next step)
    id_set.remove(idx)

    # stop if VAL_LEN has been reached
    if val_df.shape[0] >= VAL_LEN:
        break

# Train dataset
for idx in id_set:
    selected_rows = sunrgbd[sunrgbd["id"] == idx]
    train_df = train_df.append(selected_rows)

print("Train size:", train_df.shape)
print("Val size:", val_df.shape)
print("Hold out size:", hold_out_df.shape)

# Verify that the file id of the frustum point clouds in the different
# sets are independent from one another
#print("SAME IDX train==val:", np.sum(train_df["id"].eq(val_df["id"])))
#print("SAME IDX val==hold_out:", np.sum(hold_out_df["id"].eq(val_df["id"])))
#print("SAME IDX train==hold_out:", np.sum(train_df["id"].eq(hold_out_df["id"])))

# Save the train, val, and hold-out data files in the destination of your choice
# this is where you save the pickle files
destination_dir = ""

train_set_name = destination_dir + "train_new_{}.pickle".format(target_object)
val_set_name = destination_dir + "val_new_{}.pickle".format(target_object)
hold_out_set_name = destination_dir + "hold_out_new_{}.pickle".format(target_object)

# Save the data frames as pickle files
with open(train_set_name, "wb") as file:
    pickle.dump(train_df, file)

with open(val_set_name, "wb") as file:
    pickle.dump(val_df, file)

with open(hold_out_set_name, "wb") as file:
    pickle.dump(hold_out_df, file)

# 3.2 Create the largest Target Dataset to train models 
# for Transfer Networks and Reference Networks

# Path to the training and validation set data pickles created in Step 3.1
chair_train = ""
chair_val = ""

# Read the data pickle (which is a pandas dataframe object)
with open(chair_train, "rb") as file:
    sunrgbd_train = pickle.load(file)
    
with open(chair_val, "rb") as file:
    sunrgbd_val = pickle.load(file)

# Total (train+val) size for transfer = Val size of the source network (in this case KITTI)
transfer_train_size = 9120  # 285 batches
transfer_val_size = 1632  # 51 batches

sunrgbd_transfer_train = sunrgbd_train.sample(transfer_train_size, replace=False)
sunrgbd_transfer_val = sunrgbd_val.sample(transfer_val_size, replace=False)


# Save the files
save_path = ""

with open(save_path + "transfer_train_chair.pickle", "wb") as file:
    pickle.dump(sunrgbd_transfer_train, file)

with open(save_path+"transfer_val_chair.pickle", "wb") as file:
    pickle.dump(sunrgbd_transfer_val, file)

# 3.3 Create the rest of the Target Datasets

# For SUN RGB-D, the process of creating the Target Dataset
# is the same as above with the following splits

#                                       train	val				
# SUNRGBD (chair) – Transfer_val_100	9120	1632	 (largest Target Dataset from 3.2)
# SUNRGBD (chair) – Transfer_val_75	    6840	1224	
# SUNRGBD (chair) – Transfer_val_50	    4560	816		
# SUNRGBD (chair) – Transfer_val_25	    2280	408		
# SUNRGBD (chair) – Transfer_val_10	    912	    163.2		
# SUNRGBD (chair) – Transfer_val_5	    456	    81.6


# Paths to the files saved in step 3.2
train_path = ""
val_path = ""


# 75% of the SN val size
create_transfer_data_splits(train_path, val_path, val_prop = 75, dataset="sunrgbd")

# 50% of the SN val size
create_transfer_data_splits(train_path, val_path, val_prop = 50, dataset="sunrgbd")

# 25% of the SN val size
create_transfer_data_splits(train_path, val_path, val_prop = 25, dataset="sunrgbd")

# 10% of the SN val size
create_transfer_data_splits(train_path, val_path, val_prop = 10, dataset="sunrgbd")

# 5% of the SN val size
create_transfer_data_splits(train_path, val_path, val_prop = 5, dataset="sunrgbd")

