import numpy as np
import os
import glob
from PIL import Image as IM
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def calculate_iou(prediction, ground_truth, classes):
    iou_scores = []
    
    for class_id in classes:
        # Create binary masks for the current class
        pred_mask = (prediction == class_id)
        gt_mask = (ground_truth == class_id)
        
        # Calculate intersection and union
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union == 0:
            iou = np.nan  # Avoid division by zero
        else:
            iou = intersection / union
        
        iou_scores.append(iou)
    
    mean_iou = np.nanmean(iou_scores)
    
    return mean_iou

def plot_rgb(input, pred_arr, lbl_arr, color, dir_out, str_split):
    label_im = np.zeros((pred_arr.shape[0], pred_arr.shape[1], 3))
    pred_im = np.zeros((pred_arr.shape[0], pred_arr.shape[1], 3))


    for label, rgb in color.items():
        label_im[lbl_arr == label] = rgb
        pred_im[pred_arr == label] = rgb 
    
    #label_im = label_im.astype(np.uint8)
    #pred_im = pred_im.astype(np.uint8)
    
    plt.figure(figsize=(10, 8))
    plt.subplot(131), plt.imshow(input, cmap='gray'), plt.title('Source Image')
    plt.subplot(132), plt.imshow(label_im), plt.title('Multiclass GT')
    #plt.subplot(163), plt.imshow(pred1, cmap='gray'), plt.title('pred class 1')
    #plt.subplot(164), plt.imshow(pred2, cmap='gray'), plt.title('pred class 2')
    #plt.subplot(165), plt.imshow(pred3, cmap='gray'), plt.title('pred class 3')
    plt.subplot(133), plt.imshow(pred_im), plt.title('pred multiclass')
    plt.axis('off')
    plt.savefig(dir_out+f'all/all_{str_split}', bbox_inches='tight')
    plt.close() 

def plot(args):
    pref = "/" 
    pref1 = ""
    if args.hist_match:
        pref = "/hist_match/"
        pref1="hist_match_"
    dir_data = "/mnt/DGX01/anika/datasets/gan-generated/"
    dir_pred1 = dir_data+f"output-defects/defect_class1/{args.test_id}{pref}pred/" 
    dir_pred2 = dir_data+f"output-defects/defect_class2/{args.test_id}{pref}pred/"
    dir_pred3 = dir_data+f"output-defects/defect_class3/{args.test_id}{pref}pred/"  

    dir_lbl1 = dir_data+f"output-defects/defect_class1/{args.test_id}{pref}label/" 
    dir_lbl2 = dir_data+f"output-defects/defect_class2/{args.test_id}{pref}label/"
    dir_lbl3 = dir_data+f"output-defects/defect_class3/{args.test_id}{pref}label/"  

    dir_out = dir_data+f"defects/prediction-full3/{args.test_id}{pref}"
    
    #dir_raw = args.raw_path+f"test{args.test_id}/"
    #dir_raw = args.raw_path
    file_class1 = f"{pref1}{args.test_id}_defect_class1.csv" #hist_match_
    #file_class2 = f"test{args.test_id}_defect_class2.csv"
    
    os.makedirs(dir_out+"pred/", exist_ok=True)
    os.makedirs(dir_out+"label/", exist_ok=True)
    os.makedirs(dir_out+"all/", exist_ok=True)
    
    test_df = pd.read_csv(dir_data+file_class1)
    color = {0: [128, 0, 128], 128: [0, 255, 0], 255: [255, 255, 0], 200: [255, 0, 0]}
    all_img = []
    all_jacc = []
    for it, row in test_df.iterrows():
        in_file = dir_data+row['image']
        str_split = row['label'].split('/')[-1]
        dir_name = str_split.split('_')
        #img_name = dir_name[-1]
        pair_name = '_'.join(dir_name[:-1])
        pair_name = f"{pair_name}/GT/"
        #print(pair_name, img_name)
        try:
            input_im = IM.open(in_file)
            #label_im = IM.open(dir_raw+pair_name+img_name)
            pred1 = IM.open(dir_pred1+str_split)
            pred2 = IM.open(dir_pred2+str_split)
            pred3 = IM.open(dir_pred3+str_split)
        
            lbl1 = IM.open(dir_lbl1+str_split)
            lbl2 = IM.open(dir_lbl2+str_split)
            lbl3 = IM.open(dir_lbl3+str_split)

            np_lbl1 = np.asarray(lbl1)
            np_lbl1 = np.uint8(np_lbl1)
        
            np_lbl2 = np.asarray(lbl2)
            np_lbl2 = np.uint8(np_lbl2)

            np_lbl3 = np.asarray(lbl3)
            np_lbl3 = np.uint8(np_lbl3)
        
            np_pred1 = np.asarray(pred1)
            np_pred1 = np.uint8(np_pred1)
            
            np_pred2 = np.asarray(pred2)
            np_pred2 = np.uint8(np_pred2)

            np_pred3 = np.asarray(pred3)
            np_pred3 = np.uint8(np_pred3)
        
            
            multiclass_array = np.zeros_like(np_pred1)
            lbl_array = np.zeros_like(np_pred1)
            multiclass_array[np_pred1 > 0] = 128
            multiclass_array[np_pred2 > 0] = 255
            multiclass_array[np_pred3 > 0] = 200
        
            lbl_array[np_lbl1 > 0] = 128
            lbl_array[np_lbl2 > 0] = 255
            lbl_array[np_lbl3 > 0] = 200
        
            miu_score = calculate_iou(multiclass_array, lbl_array, [0,128,200,255])
            print(it, np.unique(lbl_array), np.unique(multiclass_array), miu_score)
            all_img.append(str_split)
            all_jacc.append(miu_score)
            
            multiclass_image = IM.fromarray(multiclass_array, mode='L')
            multiclass_image.save(dir_out+'pred/'+str_split)

            multiclass_lbl = IM.fromarray(lbl_array, mode='L')
            multiclass_lbl.save(dir_out+'label/'+str_split)
            #plot_rgb(input, multiclass_array, lbl_array, color, dir_out, str_split) 
            
            plt.figure(figsize=(10, 8))
            plt.subplot(131), plt.imshow(input_im, cmap='gray'), plt.title('Source Image')
            plt.subplot(132), plt.imshow(multiclass_lbl, cmap='gray'), plt.title('Multiclass GT')
            #plt.subplot(163), plt.imshow(pred1, cmap='gray'), plt.title('pred class 1')
            #plt.subplot(164), plt.imshow(pred2, cmap='gray'), plt.title('pred class 2')
            #plt.subplot(165), plt.imshow(pred3, cmap='gray'), plt.title('pred class 3')
            plt.subplot(133), plt.imshow(multiclass_image, cmap='gray'), plt.title('pred multiclass')
            plt.axis('off')
            plt.savefig(dir_out+f'all/all_{str_split}', bbox_inches='tight')
            plt.close()
            #plt.show()
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
        #if it>10:
        #    break
    agg_pred_df = pd.DataFrame()
    agg_pred_df['image'] = all_img
    agg_pred_df['miu'] = all_jacc

    agg_pred_df.to_csv(dir_out+f'multiclass_pred_score.csv')
    print("mean miu score:", np.mean(all_jacc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script plot the multiclass prediction from binary class")
    #parser.add_argument("--test_id", type=int, default=5)
    parser.add_argument("--test_id", type=str, default="")
    #parser.add_argument("--class_id", type=int, default=1)
    parser.add_argument("--raw_path", type=str, default="/mnt/DGX01/AmirZ/2024_SAM_Model/", help="raw data path.")
    parser.add_argument('--hist_match', action='store_true', help='histogram matching input or not')
    args = parser.parse_args()
    plot(args) 