import numpy as np
#import matplotlib.pyplot as plt
import os
import glob
from PIL import Image as IM
import pandas as pd
from sklearn.metrics import accuracy_score, jaccard_score

import torch
import torchmetrics
from torchmetrics.detection import IntersectionOverUnion
import pickle

dir_data = "/mnt/DGX01/anika/datasets/gan-generated"
testid = "test1"
class_id = 3
dir_out = f"/mnt/DGX01/anika/datasets/gan-generated/output-defects/defect_class{class_id}/{testid}/hist-match/" 
#dir_out = "AutogluonModels/ag-20240828_143435/ddp_prediction_cache" #defect class 2
dir_save = os.path.join(dir_data,f"output-defects/defect_class{class_id}/{testid}/hist-match")
#dir_out = "/mnt/DGX01/anika/datasets/gan-generated/outputs/output_real_real.pkl"
test_file = f"hist_match_{testid}_defect_class{class_id}.csv"
num_gpus = 4 # 3 for class1, 4 for class 2

os.makedirs(dir_save+'pred', exist_ok=True)
os.makedirs(dir_save+'label', exist_ok=True)

def expand_path(df, dataset_dir):
    for col in ["image", "label"]:
        df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


test_df = expand_path(pd.read_csv(os.path.join(dir_data, test_file)), dir_data)
#print(test_df.shape)

avg_jac = []
iou = torchmetrics.JaccardIndex(task="binary")
num=0

print(test_df.head)

file_name = f"output_defect_class{class_id}.pkl"
with open(dir_out+file_name, 'rb') as f: 
    out_file = pickle.load(f)


for idx in range(out_file[0]['logits'].size()[0]):
    logit, label = out_file[0]['logits'][idx], out_file[0]['label'][idx]
    #logit, label = logit.squeeze(), label.squeeze()
    
    image_name = test_df.iloc[idx,1].split('/')[-1] #im_id --> idx

    png_pred = IM.fromarray(logit.squeeze().cpu().numpy()*255)
    png_lbl = IM.fromarray(label.squeeze().cpu().numpy()*255)
    png_pred = png_pred.convert("L")
    png_lbl = png_lbl.convert("L")
    png_pred.save(os.path.join(dir_save,"pred")+"/"+image_name)
    png_lbl.save(os.path.join(dir_save,"label")+"/"+image_name)
    iou_sc = iou(logit, label)
    #lval = np.unique(logit.squeeze().cpu().numpy()) #multiple values eps in float, hence converting to int
    print(num, iou_sc.item())
    avg_jac.append(iou_sc.item())
    num+=1 

print(f"Avg Jaccard:{np.mean(avg_jac)}")

iou_df = pd.DataFrame()
iou_df[f"iou_defect_class{class_id}"] = avg_jac
iou_df.to_csv(os.path.join(dir_save,f"iou_defect_class{class_id}.csv"))

