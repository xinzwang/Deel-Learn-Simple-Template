import glob
import os
import os.path as osp
import pickle
import sys

import cv2
import lmdb

sys.path.append("../")
from utils import ProgressBar

sys.path.append("../")

# configurations
# img_folder = "/home/lzx/SRDatasets/DIV2K_train/HR/x4/*"
# lmdb_save_path = "/home/lzx/SRDatasets/DIV2K_train/HR/x4_new.lmdb"
# img_folder = 'F:/wangxinzhe/codes/datasets/DIV2K/DIV2K_valid_LR_unknown/X4_mini/*'
img_folder = "../../../datasets/DIV2K/DIV2K_valid_LR_bicubic/X4_mini/*"
lmdb_save_path = '../../../datasets/SRDatasets/DIV2K_valid/BicLR/x4_mini.lmdb'

meta_info = {"name": "x4"}

mode = (
    2  # 1 for reading all the images to memory and then writing to lmdb (more memory);
)
# 2 for reading several images and then writing to lmdb, loop over (less memory)
batch = 1000  # Used in mode 2. After batch images, lmdb commits.
###########################################
if not lmdb_save_path.endswith(".lmdb"):
    raise ValueError("lmdb_save_path must end with 'lmdb'.")
#### whether the lmdb file exist
if osp.exists(lmdb_save_path):
    print("Folder [{:s}] already exists. Exit...".format(lmdb_save_path))
    sys.exit(1)
img_list = sorted(glob.glob(img_folder))
if mode == 1:
    print("Read images...")
    dataset = [cv2.imread(v, cv2.IMREAD_UNCHANGED) for v in img_list]
    data_size = sum([img.nbytes for img in dataset])
elif mode == 2:
    print("Calculating the total size of images...")
    data_size = sum(os.stat(v).st_size for v in img_list)
else:
    raise ValueError("mode should be 1 or 2")

key_l = []
resolution_l = []
pbar = ProgressBar(len(img_list))
env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
txn = env.begin(write=True)  # txn is a Transaction object
for i, v in enumerate(img_list):
    pbar.update("Write {}".format(v))
    base_name = osp.splitext(osp.basename(v))[0]
    key = base_name.encode("ascii")
    data = dataset[i] if mode == 1 else cv2.imread(v, cv2.IMREAD_UNCHANGED)
    if data.ndim == 2:
        H, W = data.shape
        C = 1
    else:
        H, W, C = data.shape
    txn.put(key, data)
    key_l.append(base_name)
    resolution_l.append("{:d}_{:d}_{:d}".format(C, H, W))
    # commit in mode 2
    if mode == 2 and i % batch == 1:
        txn.commit()
        txn = env.begin(write=True)

txn.commit()
env.close()

print("Finish writing lmdb.")

#### create meta information
# check whether all the images are the same size
same_resolution = len(set(resolution_l)) <= 1
if same_resolution:
    meta_info["resolution"] = [resolution_l[0]]
    meta_info["keys"] = key_l
    print("All images have the same resolution. Simplify the meta info...")
else:
    meta_info["resolution"] = resolution_l
    meta_info["keys"] = key_l
    print("Not all images have the same resolution. Save meta info for each image...")

#### pickle dump
pickle.dump(meta_info, open(osp.join(lmdb_save_path, "meta_info.pkl"), "wb"))
print("Finish creating lmdb meta info.")
