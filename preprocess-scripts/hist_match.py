import os
import glob
import numpy as np
from PIL import Image, ImageEnhance
from numpy import asarray
import cv2
import random
#import matplotlib.pyplot as plt
import scipy
from skimage.filters import threshold_otsu

#Amirs 
def calc_output(x, p1, p2, mp1, mp2):
    #x: input image
    # p1,p2: are the location of the peak in the deisred distribution
    # mp1, mp2:  location of the peaks in the input distribution
    
    dt = x.dtype
    x = x.astype('float32')
    output = np.subtract(x, mp1.astype(dt), out=x)  # Subtract mp1 from x, store result in x (in-place)
    scale = (p2 - p1) / (mp2 - mp1)
    output *= scale  # Multiply by (p2 - p1) / (mp2 - mp1), in-place
    output += p1  # Add p1, in-place
    output = output.astype(dt)
    return output
    
 
def hist_cluster_match_anytype(rec_fdk_fn,img_org):
    # rec_fdk_fn: input image
    # img_org: image with desired distribution
    
    output = scipy.ndimage.zoom(img_org,zoom=1/10,order=0)#
    ret = threshold_otsu(output)
    lt_indices = np.where(output < ret)
    ge_indices = np.where(output >= ret)
 
    # Compute medians using the indices
    peak_1 = np.median(output[lt_indices])
    peak_2 = np.median(output[ge_indices])
    print([peak_1,peak_2])
 
    output = scipy.ndimage.zoom(rec_fdk_fn,zoom=1/10,order=0)#
    ret = threshold_otsu(output)
    # Get the indices of elements that are less than and greater than or equal to the threshold
    lt_indices = np.where(output < ret)
    ge_indices = np.where(output >= ret)
 
    # Compute medians using the indices
    m1p = np.median(output[lt_indices])
    m2p = np.median(output[ge_indices])
    #print([m1p,m2p])
 
    output = calc_output(rec_fdk_fn,peak_1,peak_2,m1p,m2p)
 
 
    return  output,m1p,m2p

if __name__ == "__main__":
    testid = 4
    dir_main = f"/mnt/DGX01/anika/datasets/gan-generated/defects/test{testid}/"
    dir_ref = "/mnt/DGX01/anika/datasets/gan-generated/defects/input/"
    dir_src = dir_main+"original_input/"
    dir_out = dir_main+"input/"
    os.makedirs(dir_out, exist_ok=True)

    img_list = glob.glob(dir_src+'*.png') 
    ref_img_list = glob.glob(dir_ref+'*.png')
    
    
    for fid in range(len(img_list)):
        fname = os.path.basename(img_list[fid])
        src_image = cv2.imread(img_list[fid], cv2.IMREAD_GRAYSCALE)
        rid = random.randint(0, len(ref_img_list))
        ref_image = cv2.imread(ref_img_list[rid], cv2.IMREAD_GRAYSCALE)
        
        '''
        # crop src image to be of same size for hist match
        start_x = int((src_image.shape[0]-ref_image.shape[0])/2) # (768 - 512) / 2
        start_y = int((src_image.shape[1]-ref_image.shape[1])/2)  # (768 - 512) / 2
    
        end_x = start_x + ref_image.shape[0]
        end_y = start_y + ref_image.shape[1]
        cropped_image = src_image[start_y:end_y, start_x:end_x]
        '''
        #alternative2: resize ref image as src
        resized_image = cv2.resize(ref_image, src_image.shape, interpolation=cv2.INTER_LINEAR)
    
        #matched_image1, m1p, m2p = hist_cluster_match_anytype(cropped_image, ref_image)
        matched_image2, m1p, m2p = hist_cluster_match_anytype(src_image, resized_image)
        #print(fid, matched_image1.shape, matched_image2.shape)
        factor = 5  # Change this value to adjust brightness
        #png_matched_img1 = Image.fromarray(matched_image1)
        #enhancer1 = ImageEnhance.Brightness(png_matched_img1)
    
        png_matched_img2 = Image.fromarray(matched_image2)
        #enhancer2 = ImageEnhance.Brightness(png_matched_img2)
        #brightened_image2 = enhancer2.enhance(factor)
        
        print(fid, matched_image2.shape)
        png_matched_img2 = png_matched_img2.convert("RGB")
        #brightened_image2.save(dir_out+fname)
        #png_matched_img2.save(dir_out+fname)
        


