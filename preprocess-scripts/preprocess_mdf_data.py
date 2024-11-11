import numpy as np
import os
import glob
from PIL import Image as IM
import pandas as pd
import shutil

dir_data = "/mnt/DGX01/AmirZ/AMMT_Related_Works/Long Exposure Scans/Kakapo_TrainingPairs_NoFalsePositive/"
total_pair = 9
pair_name_af ="AFAwNiInclusionsAndFilledDefects_pair_"
pair_name_inc ="Inc718_pair_"

dir_out = "/mnt/DGX01/anika/datasets/gan-generated/"


fdir = "defects/input/"
ldir1 = "defects/class_1/labels/" 
ldir2 = "defects/class_2/labels/"
ldir3 = "defects/class_3/labels/"  

#os.makedirs(dir_out+fdir, exist_ok=True)
#os.makedirs(dir_out+ldir1, exist_ok=True)
#os.makedirs(dir_out+ldir2, exist_ok=True)
os.makedirs(dir_out+ldir3, exist_ok=True)


list_img_afc = []
list_lbl1_afc = []
list_lbl2_afc = []
list_lbl3_afc = []
#list the src images
#convert multiclass 2 binary
#move src and binary GT to respective directory
for i in range(1,total_pair):
    dir_name = dir_data+pair_name_af+f"{i}/input/*.png"
    dir_gt = dir_data+pair_name_af+f"{i}/GT/"
    list_dir = glob.glob(dir_name)
    list_dir.sort()
    print(f"pair {i}: {len(list_dir)}")
    
    for fid in range(len(list_dir)):
        fname = os.path.basename(list_dir[fid])
        out_fname = f"{pair_name_af}{i}_{fname}"
        
        list_img_afc.append(fdir+out_fname)
        list_lbl1_afc.append(ldir1+out_fname)
        list_lbl2_afc.append(ldir2+out_fname)  
        list_lbl3_afc.append(ldir3+out_fname)
        im = IM.open(list_dir[fid])
        gt = IM.open(dir_gt+fname)
        np_data = np.asarray(gt)
        np_data = np.uint8(np_data)
        binary_class_3 = (np_data == 3).astype(int) 
        
        binary_class_1 = (np_data == 1).astype(int)
        #binary_class_2 = np.where(np_data >= 2, 1, 0).astype(int)
        binary_class_2 = (np_data == 2).astype(int)
        
        binary_class_1 = binary_class_1*255
        binary_class_2 = binary_class_2*255
        
        binary_class_3 = binary_class_3*255  
        
        shutil.copy2(list_dir[fid], dir_out+fdir+out_fname)
        
        label1 = IM.fromarray(binary_class_1.astype(np.uint8))
        label2 = IM.fromarray(binary_class_2.astype(np.uint8))
        
        label3 = IM.fromarray(binary_class_3.astype(np.uint8))
        
        label1.save(dir_out+ldir1+out_fname)
        label2.save(dir_out+ldir2+out_fname)
        
        label3.save(dir_out+ldir3+out_fname)

print("total data afc:", len(list_img_afc))

list_img_inc = []  
#list_lbl1_inc = []
#list_lbl2_inc = []
list_lbl3_inc = []
#inc718
for i in range(5, total_pair):
    dir_name = dir_data+pair_name_inc+f"{i}/input/*.png"
    dir_gt = dir_data+pair_name_inc+f"{i}/GT/"
    list_dir = glob.glob(dir_name)
    list_dir.sort()
    print(f"pair {i}: {len(list_dir)}")
    
    for fid in range(len(list_dir)):
        fname = os.path.basename(list_dir[fid])
        out_fname = f"{pair_name_inc}{i}_{fname}"
        
        list_img_inc.append(fdir+out_fname)
        '''
        list_lbl1_inc.append(ldir1+out_fname)
        '''
        #list_lbl2_inc.append(ldir2+out_fname)
        
        list_lbl3_inc.append(ldir3+out_fname)
        
        #im = IM.open(list_dir[fid])
        gt = IM.open(dir_gt+fname)

        #resized_image = im.resize((512, 512))
        
        resized_gt = gt.resize((512, 512))
        
        np_data = np.asarray(resized_gt)
        np_data = np.uint8(np_data)
        '''
        binary_class_1 = (np_data == 1).astype(int)
        binary_class_1 = binary_class_1*255
        '''
        binary_class_2 = (np_data == 2).astype(int)
        binary_class_2 = binary_class_2*255 
        
        binary_class_3 = (np_data == 3).astype(int)
        binary_class_3 = binary_class_3*255 
        
        #resized_image.save(dir_out+fdir+out_fname)
        
        #label1 = IM.fromarray(binary_class_1.astype(np.uint8))
        label2 = IM.fromarray(binary_class_2.astype(np.uint8))
        label3 = IM.fromarray(binary_class_3.astype(np.uint8))

        #label1.save(dir_out+ldir1+out_fname)
        label2.save(dir_out+ldir2+out_fname)
        label3.save(dir_out+ldir3+out_fname)

print("total data inc:", len(list_img_inc))
'''
train_afc = int(len(list_img_afc)*0.8)
train_inc = int(len(list_img_inc)*0.8)

print('train:', train_afc, train_inc)

train1 = pd.DataFrame()
test1 = pd.DataFrame()

train1['image'] = list_img_afc[:train_afc]+list_img_inc[:train_inc]
train1['label'] = list_lbl1_afc[:train_afc]+list_lbl1_inc[:train_inc] 

train1.to_csv(dir_out+"train_defect_class1.csv")

test1['image'] = list_img_afc[train_afc:]+list_img_inc[train_inc:]
test1['label'] = list_lbl2_afc[train_afc:]+list_lbl2_inc[train_inc:] 

test1.to_csv(dir_out+"test_defect_class1.csv")

train2 = pd.DataFrame()
test2 = pd.DataFrame()

train2['image'] = list_img_afc[:train_afc]+list_img_inc[:train_inc]
train2['label'] = list_lbl1_afc[:train_afc]+list_lbl1_inc[:train_inc] 

train2.to_csv(dir_out+"train_defect_class2.csv")

test2['image'] = list_img_afc[train_afc:]+list_img_inc[train_inc:]
test2['label'] = list_lbl2_afc[train_afc:]+list_lbl2_inc[train_inc:] 

test2.to_csv(dir_out+"test_defect_class2.csv")

train3 = pd.DataFrame()
test3 = pd.DataFrame()

train3['image'] = list_img_afc[:train_afc]+list_img_inc[:train_inc]
train3['label'] = list_lbl3_afc[:train_afc]+list_lbl3_inc[:train_inc] 

train3.to_csv(dir_out+"train_defect_class3.csv")

test3['image'] = list_img_afc[train_afc:]+list_img_inc[train_inc:]
test3['label'] = list_lbl3_afc[train_afc:]+list_lbl3_inc[train_inc:] 

test3.to_csv(dir_out+"test_defect_class3.csv")

print("Data process finished successfully!")
'''
'''

# Specify the directory containing the files
directory_path = dir_out+"test_defect_class2.csv"

# Iterate through all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".png.png"):
        # Define the old and new file paths
        old_file_path = os.path.join(directory_path, filename)
        new_filename = filename.replace(".png.png", ".png")
        new_file_path = os.path.join(directory_path, new_filename)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)

print("All files have been renamed.")


df = pd.read_csv(directory_path)

# Replace '*.png.png' with '*.png' in all columns
df = df.drop(df.columns[:1], axis=1)
df = df.replace('.png.png', '.png', regex=True)
#df = df.replace('class_1', 'class_2', regex=True)

print(df.head())

df.to_csv(directory_path)
'''