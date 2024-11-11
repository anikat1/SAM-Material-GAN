import numpy as np
import os
import glob
from PIL import Image as IM
import pandas as pd
import shutil

testid = 6#"few-shot"
num_files = 4
dir_main = f"/mnt/DGX01/AmirZ/2024_SAM_Model/test{testid}/"
#dir_main = f"/mnt/DGX01/anika/datasets/gan-generated/defects/{testid}/"
dir_out = "/mnt/DGX01/anika/datasets/gan-generated/"
fdir = f"defects/test{testid}/"
#fdir = f"defects/{testid}/"
#pair_name = ["pair_1p_Ti64"] #test 1
#pair_name = ["pair_5_Ti64"] #test2
#test 3
#pair_name = ['Pair_1_FDK143_AFAPB05_8s_GTcombinedTr16andObaidSeg', 'Pair_2_FDK2132_AFAPB05_8s_GTcombinedTr16andObaidSeg',
#           'Pair_4_FDK146_AFAPB10_4s_GTObaidSeg', 'Pair_5_FDK292_AFAPB10_4s_GTObaidSeg', 'Pair_6_FDK2132_AFAPB10_4s_GTObaidSeg']

#pair_name = ["Training_3_only316L_pair1"] #test 4
#pair_name = ["pair_316L_7"] #test 5
pair_name = ["test_inp_6_for_fewShotTrain_matched"] #test 6
#pair_name = ["pair_1", "pair_2"] #few-shot train

list_img = []
list_lbl1 = []
list_lbl2 = []
list_lbl3 = []
for l in pair_name:
    dir_data = dir_main+l+'/' #_matched
    dir_data_gt = dir_main+l+'/'
    indir = 'input/'
    #indir1 = 'original-input/'
    ldir1 = "class_1_labels/"
    ldir2 = "class_2_labels/"
    ldir3 = "class_3_labels/"

    os.makedirs(dir_out+fdir+indir, exist_ok=True) #indir1
    os.makedirs(dir_out+fdir+ldir1, exist_ok=True)
    os.makedirs(dir_out+fdir+ldir2, exist_ok=True)
    os.makedirs(dir_out+fdir+ldir3, exist_ok=True)

    
    dir_gt = dir_data_gt+"GT/"
    list_dir = glob.glob(dir_data+indir+"*.png")
    #list_dir.sort()
    print('total files:',len(list_dir))
    for fid in range(len(list_dir)):
        fname = os.path.basename(list_dir[fid])
        out_fname = f"{l}_{fname}"
        list_img.append(fdir+indir+out_fname) #indir1
        #shutil.copy2(list_dir[fid], dir_out+fdir+indir+out_fname) #indir1
        
        list_lbl1.append(fdir+ldir1+out_fname)
        list_lbl2.append(fdir+ldir2+out_fname)
        list_lbl3.append(fdir+ldir3+out_fname)
        
        #im = IM.open(list_dir[fid])
        gt = IM.open(dir_gt+fname)
        np_data = np.asarray(gt)
        np_data = np.uint8(np_data)
        
        binary_class_1 = (np_data == 1).astype(int)
        #binary_class_2 = np.where(np_data >= 2, 1, 0).astype(int)
        binary_class_2 = (np_data == 2).astype(int) 
        binary_class_3 = (np_data == 3).astype(int) 
        binary_class_1 = binary_class_1*255
        binary_class_2 = binary_class_2*255
        binary_class_3 = binary_class_3*255  
        
        label1 = IM.fromarray(binary_class_1.astype(np.uint8))
        label2 = IM.fromarray(binary_class_2.astype(np.uint8))
        label3 = IM.fromarray(binary_class_3.astype(np.uint8))

        label1.save(dir_out+fdir+ldir1+out_fname)
        label2.save(dir_out+fdir+ldir2+out_fname)
        label3.save(dir_out+fdir+ldir3+out_fname)
        
        

test1 = pd.DataFrame()
test1['image'] = list_img
test1['label'] = list_lbl1

test2 = pd.DataFrame()
test2['image'] = list_img
test2['label'] = list_lbl2

test3 = pd.DataFrame()
test3['image'] = list_img
test3['label'] = list_lbl3

test1.to_csv(dir_out+f"train{testid}_defect_class1.csv") #hist_match_
test2.to_csv(dir_out+f"train{testid}_defect_class2.csv") #hist_match_
test3.to_csv(dir_out+f"train{testid}_defect_class3.csv") #hist_match_

print(f"data process finished successfully!")




