import os
import numpy as np
import time
from skimage.filters import threshold_otsu,threshold_multiotsu
from scipy.ndimage import binary_erosion
from skimage.measure import label, regionprops
from tqdm import tqdm
import scipy

part_mask = (xgt>0).astype('uint8')
defects_mask =  (xgt>1).astype('uint8')
 
 
def separate_defects(part_mask, defects_mask, iterations=3):
    # Erode the part mask by the specified number of iterations (voxels)
    eroded_part_mask = binary_erosion(part_mask, iterations=iterations)
 
    # Subtract the eroded part mask from the original part mask to obtain the edge mask
    edge_mask = part_mask & ~eroded_part_mask
 
    # Multiply the edge mask by the defects volume to create a volume containing only the surface defects
    surface_defects = edge_mask & defects_mask
 
    # Subtract the surface defects volume from the original defects volume to obtain the internal defects volume
    internal_defects = defects_mask & ~surface_defects
 
    return surface_defects, internal_defects, eroded_part_mask

def calculate_metrics(internal_defects, internal_defects_ref,margin = 0):
    # internal_defects: binary map of pores or particles
    # internal_defects_ref: ground truth binary map

    lap = label(internal_defects)
    # rlap = regionprops(lap)
    # for rlapi in tqdm(rlap):
    #     bb = rlapi.bbox
    #     bbdiff = np.array([bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]])
    #     idbb = np.where(bbdiff > 1)[0]
    #     if len(idbb)<1:
    #         lap[lap==rlapi.label] = 0
    # lap =  lap>0
    # lap = label(lap)

    lld = label(internal_defects_ref.astype('uint16'))
    rld = regionprops(lld)
    flaw_info = []
    ## loop over the flaws from reference volume
    ## in each bounding box around the flaw from ref. if there is a flaw from object
    ## then mark it as one, otherwise mark it as zero
    count = 0
    max_top = np.shape(internal_defects_ref)
    min_bott = [0, 0, 0]
    count_miss_larg = 0
    for rldi in tqdm(rld):
        bb = rldi.bbox
        bbdiff = np.array([bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]])
        idbb = np.where(bbdiff > 1)[0]
        if len(idbb) > 0:
            if np.sum(internal_defects[np.max([bb[0] - margin, min_bott[0]]):np.min([bb[3] + margin, max_top[0] - 1]),
                      np.max([bb[1] - margin, min_bott[1]]):np.min([bb[4] + margin, max_top[1] - 1]),
                      np.max([bb[2] - margin, min_bott[2]]):np.min([bb[5] + margin, max_top[2] - 1])]) > 0:
                
                pt_lab = np.max(lap[np.max([bb[0] - margin, min_bott[0]]):np.min([bb[3] + margin, max_top[0] - 1]),
                      np.max([bb[1] - margin, min_bott[1]]):np.min([bb[4] + margin, max_top[1] - 1]),
                      np.max([bb[2] - margin, min_bott[2]]):np.min([bb[5] + margin, max_top[2] - 1])])
                flaw_info.append([rldi.equivalent_diameter_area, 1, pt_lab])

            else:
                flaw_info.append([rldi.equivalent_diameter_area, 0, 0])
                if rldi.equivalent_diameter_area > 200:
                    print(bb)
                    count_miss_larg += 1
            count += 1
    print('%d of the %d pores were more than one layer' % (count, len(rld)))

    flaw_info = np.array(flaw_info)
    TP = flaw_info[flaw_info[:, 1] == 1] # find the True positives
 

    if len(TP)>0:
        precision = np.min([np.max(TP[:,2]),len(TP)])/np.max(lap)
        recall = np.min([np.max(TP[:,2]),len(TP)])/count
        F1 = recall * precision * 2 / (precision + recall)
    else:
        precision = 0
        recall =0
        F1 = 0

    return F1, recall, precision
 

#xgt is your ground truth labels
label_pore = 2
label_inclusion =3

# first calculate the metrics for pores
# pore binary map: internal_defects_pores
internal_defects_ref = (xgt==label_pore).astype('uint16')
METRICS_POREs = calculate_metrics(internal_defects_pores, internal_defects_ref,margin = 0)


# first calculate the metrics for inclusions
# pore binary map: internal_defects_pores
internal_defects_ref = (xgt==label_inclusion).astype('uint16')
METRICS_inclusions = calculate_metrics(internal_defects_inclusions, internal_defects_ref,margin = 0) 