#import nibabel as nib
import numpy as np
#import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from PIL import Image as IM


dir_data = "../datasets/"
fdir = "test6/input/"
ldir = "test6/class_1/"
out_file = "test6_class1.csv"
im_list = glob.glob(dir_data+fdir+'*.png')
label_list = glob.glob(dir_data+ldir+'*.png')
rel_im_paths = [path[len(dir_data):] for path in im_list]
rel_lbl_paths = [path[len(dir_data):] for path in label_list]
rel_im_paths.sort()
rel_lbl_paths.sort()
train = pd.DataFrame()
train['image'] = rel_im_paths
train['label'] = rel_lbl_paths
print(train.shape)
train.to_csv(dir_data+out_file)
'''
dir_file = pd.read_csv(dir_data+"test_defect_class1.csv")
new_file = pd.DataFrame(columns=["image", "label"])
list_img = []
list_lbl = []
for ix, row in dir_file.iterrows():
    dir_img = dir_data + row['label']
    im = IM.open(dir_img)
    im_arr = np.array(im)
    #im_arr[(im_arr == 1) | (im_arr == 2)] = 0
    #im_arr[im_arr == 4] = 255
    new_val = np.unique(im_arr)
    print(ix, len(new_val),new_val)
    if len(new_val)>=2:
        list_img.append(row['image'])
        list_lbl.append(row['label'])
        
    #png_im = IM.fromarray(im_arr)  
    #png_im.save(dir_img)       

new_file['image'] = list_img
new_file['label'] = list_lbl
new_file.to_csv(dir_data+"new_test_defect_class1.csv")
'''
