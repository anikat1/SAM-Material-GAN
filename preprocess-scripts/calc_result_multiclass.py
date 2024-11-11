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

def expand_path(df, dataset_dir):
    for col in ["image", "label"]:
        df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df

def map_mask2_color(semantic_mask, num_class, colors):
    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    # Map each class to its color
    #for j in range(1024):
        #for k in range(1024):
    for i in range(num_class):
        mask = semantic_mask[i] > 0
        print(mask.size(), np.unique(mask, return_counts=True))
        image[mask] = colors[i]
    
    return image

dir_out = "/mnt/DGX01/anika/datasets/gan-generated/outputs/output_multiclass_real.pkl"
dir_data = "/mnt/DGX01/anika/datasets/gan-generated"

with open(dir_out, "rb") as f:
    out_file = pickle.load(f)

print(out_file[0].keys())
print(out_file[0]['logits'].size(), out_file[0]['class_logits'].size(), out_file[0]['moe_loss'], out_file[0]['semantic_mask'].size())

colors = [
    (255, 0, 0),    # Red for class 0
    (0, 255, 0),    # Green for class 1
    (0, 0, 255),    # Blue for class 2
    (255, 255, 0),  # Yellow for class 3
    (0, 255, 255)   # Cyan for class 4
]
num_class = 5 
test_df = expand_path(pd.read_csv(os.path.join(dir_data, f"multiclass_test_real.csv")), dir_data)
dir_save = os.path.join(dir_data,"outputs/multiclass")

for idx in range(out_file[0]['semantic_mask'].size()[0]):
    pred = out_file[0]['semantic_mask'][idx, 0]
    pred_image = map_mask2_color(pred, num_class, colors)
    image_name = test_df.iloc[idx,1].split('/')[-1]
    png_pred = IM.fromarray(pred_image)
    png_pred.save(os.path.join(dir_save,image_name))



