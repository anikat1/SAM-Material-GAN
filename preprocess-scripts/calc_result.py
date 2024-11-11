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
testid = "train"
class_id = 3#
hist_match = False
#dir_out = "AutogluonModels/ag-20240826_214753/ddp_prediction_cache" #defect class 1
#dir_out = "AutogluonModels/ag-20240828_143435/ddp_prediction_cache" #defect class 2
dir_out = "AutogluonModels/ag-20240905_025905/ddp_prediction_cache" #defect class 3
#few-shot
#dir_out = "AutogluonModels/ag-20240911_160522/ddp_prediction_cache" #defect class 1
#dir_out = "AutogluonModels/ag-20240911_162908/ddp_prediction_cache" #defect class 2
#dir_out = "AutogluonModels/ag-20240911_164044/ddp_prediction_cache" #defect class 3

#few-shot-pair2
#dir_out = "AutogluonModels/ag-20240912_005045/ddp_prediction_cache" #defect class 1
#dir_out = "AutogluonModels/ag-20240912_005821/ddp_prediction_cache" #defect class 2
#dir_out = "AutogluonModels/ag-20240912_010755/ddp_prediction_cache" #defect class 3

num_gpus = [0, 1, 2, 3] # 3 for class1, 4 for class 2, 4 for class 3
pref="/"
pref1 = ""
dir_save = os.path.join(dir_data,f"output-defects/defect_class{class_id}/{testid}/")
if hist_match:
    pref = "/hist_match/"
    pref1 = "hist_match_"
    dir_save = os.path.join(dir_data,f"output-defects/defect_class{class_id}/training{pref}")
#dir_out = "/mnt/DGX01/anika/datasets/gan-generated/outputs/output_real_real.pkl"
test_file = f"{pref1}{testid}_defect_class{class_id}.csv" #hist_match_


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
#for idx in range(out_file[0]['logits'].size()[0]):
for gpu in num_gpus:
    out = torch.load(os.path.join(dir_out, f"predictions_rank_{gpu}.pt")) 
    indx = torch.load(os.path.join(dir_out, f"sample_indices_rank_{gpu}.pt"))
    print('gpu:', gpu, len(out), len(indx[0]))

    #for idx in range(len(out)):
    for idx in range(len(indx[0])):
        #logit, label = out_file[0]['logits'][idx], out_file[0]['label'][idx]
        #logit, label = logit.squeeze(), label.squeeze()
        logit, label = out[idx]['logits'], out[idx]['label'] #each size (1,1,1024,1024)
        im_id = indx[0][idx][0]
        image_name = test_df.iloc[im_id,1].split('/')[-1] #im_id --> idx
        #print(logit.size(), label.size())
        png_pred = IM.fromarray(logit.squeeze().cpu().numpy()*255)
        png_lbl = IM.fromarray(label.squeeze().cpu().numpy()*255)
        png_pred = png_pred.convert("L")
        png_lbl = png_lbl.convert("L")
        png_pred.save(os.path.join(dir_save,"pred")+"/"+image_name)
        png_lbl.save(os.path.join(dir_save,"label")+"/"+image_name)
        iou_sc = iou(logit, label)
        #lval = np.unique(logit.squeeze().cpu().numpy()) #multiple values eps in float, hence converting to int
        print(num, im_id, iou_sc.item())
        avg_jac.append(iou_sc.item())
        num+=1 

'''
out1 = torch.load(os.path.join(dir_out, "predictions_rank_0.pt"))
out2 = torch.load(os.path.join(dir_out, "predictions_rank_1.pt"))
out3 = torch.load(os.path.join(dir_out, "predictions_rank_2.pt"))

indx1 = torch.load(os.path.join(dir_out, "sample_indices_rank_0.pt"))
indx2 = torch.load(os.path.join(dir_out, "sample_indices_rank_1.pt"))
indx3 = torch.load(os.path.join(dir_out, "sample_indices_rank_2.pt"))
#indx1[0]+=indx2[0]

for idx in range(len(out1)):
    #logit, label = out_file[0]['logits'][idx], out_file[0]['label'][idx]
    #logit, label = logit.squeeze(), label.squeeze()
    logit, label = out1[idx]['logits'], out1[idx]['label']
    im_id = indx1[0][idx][0]
    image_name = test_df.iloc[im_id,1].split('/')[-1] #im_id --> idx
    png_pred = IM.fromarray(logit.squeeze().cpu().numpy()*255)
    png_lbl = IM.fromarray(label.squeeze().cpu().numpy()*255)
    png_pred = png_pred.convert("L")
    png_lbl = png_lbl.convert("L")
    png_pred.save(os.path.join(dir_save,"pred")+"/"+image_name)
    png_lbl.save(os.path.join(dir_save,"label")+"/"+image_name)
    iou_sc = iou(logit, label)
    print(num, iou_sc.item())
    avg_jac.append(iou_sc.item())
    num+=1 
    

for idx in range(len(out2)):
    logit, label = out2[idx]['logits'], out2[idx]['label']    
    im_id = indx2[0][idx][0]
    image_name = test_df.iloc[im_id,1].split('/')[-1]
    png_pred = IM.fromarray(logit.squeeze().cpu().numpy()*255)
    png_lbl = IM.fromarray(label.squeeze().cpu().numpy()*255)
    png_pred = png_pred.convert("L")
    png_lbl = png_lbl.convert("L")
    png_pred.save(os.path.join(dir_save,"pred")+"/"+image_name)
    png_lbl.save(os.path.join(dir_save,"label")+"/"+image_name)
    iou_sc = iou(logit, label)
    print(num, iou_sc.item())
    avg_jac.append(iou_sc.item())
    num+=1

for idx in range(len(out3)):
    logit, label = out3[idx]['logits'], out3[idx]['label']    
    im_id = indx3[0][idx][0]
    image_name = test_df.iloc[im_id,1].split('/')[-1]
    png_pred = IM.fromarray(logit.squeeze().cpu().numpy()*255)
    png_lbl = IM.fromarray(label.squeeze().cpu().numpy()*255)
    png_pred = png_pred.convert("L")
    png_lbl = png_lbl.convert("L")
    png_pred.save(os.path.join(dir_save,"pred")+"/"+image_name)
    png_lbl.save(os.path.join(dir_save,"label")+"/"+image_name)
    iou_sc = iou(logit, label)
    print(num, iou_sc.item())
    avg_jac.append(iou_sc.item())
    num+=1

'''

print(f"Avg Jaccard:{np.mean(avg_jac)}")

iou_df = pd.DataFrame()
iou_df["iou_defect_class2"] = avg_jac
iou_df.to_csv(os.path.join(dir_save,f"{testid}_iou_defect_class{class_id}.csv"))

