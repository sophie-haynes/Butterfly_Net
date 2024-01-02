import cv2, opencxr

from matplotlib.path import Path
import numpy as np

def mask_outside_lungs(img_np, spacing, seg_img, margin=0, crop=True, flatten_base = False):
    """
    Method to mask all tissue from the lung tissue outward.
    
    :param img_np: image to be masked, numpy array.
    :param spacing: image load spacing, numpy array.
    :param seg_result: segmentaion mask of lungs, numpy array.
    :param margin: boundary (in pixels) to add to mask border, int.
    :param crop: crop image to segmentation mask, boolean. 
    :param flatten_base: make bottom of mask flat, boolean
    :return: The masked image
    """
    # Make lung seg masked img with black bg
    blk_img_np = opencxr.utils.mask_crop.set_non_mask_constant(\
                    img_np, seg_img)
    if (crop):
        # crop seg masked img to the seg mask
        blk_img_np, blk_crop_changes = crop_to_mask(blk_img_np, spacing, seg_img, margin)
        # crop original image
        img_np, crop_changes = crop_to_mask(img_np, spacing, seg_img, margin)
        # crop segmentation map to match
        seg_img, _ = opencxr.utils.apply_size_changes_to_img(seg_img,spacing,crop_changes)

    # calculate mask metrics
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(seg_img.T)
    # catch failed connectedComponent
    if numLabels<3:
        raise Exception("Unable to identify mask components!")
    #drop row with max area (auto full img label)
    stats = np.delete(stats,np.argmax(stats.T[4]), axis=0)
    # calculate the left and right lung segments
    l_stat, r_stat = stats[np.argmin(stats.T[0])],stats[np.argmax(stats.T[0])]
    # get first intersecting point of the top Y value
    l_top_x = np.where(seg_img.T[l_stat[cv2.CC_STAT_TOP]]>0)[0][0]
    r_top_x = np.where(seg_img.T[r_stat[cv2.CC_STAT_TOP]]>0)[0][-1]

    if flatten_base:
        # create path of maximal outermost points around lung tissue
        # if left max is bigger than right max
        if (l_stat[cv2.CC_STAT_TOP] + l_stat[cv2.CC_STAT_HEIGHT]) > (r_stat[cv2.CC_STAT_TOP]+r_stat[cv2.CC_STAT_HEIGHT]):
            # add point below right-most point at same level as left
            new_point = (r_stat[cv2.CC_STAT_LEFT]+r_stat[cv2.CC_STAT_WIDTH],\
                        l_stat[cv2.CC_STAT_TOP] + l_stat[cv2.CC_STAT_HEIGHT])
        # else if right max is bigger or equal to left max
        else:
            # add point below left-most max point at same level as right
            new_point = (l_stat[cv2.CC_STAT_LEFT],\
                        r_stat[cv2.CC_STAT_TOP]+r_stat[cv2.CC_STAT_HEIGHT])
        pth = Path([[l_stat[cv2.CC_STAT_LEFT],l_stat[cv2.CC_STAT_TOP] + l_stat[cv2.CC_STAT_HEIGHT] ],\
            [l_top_x,l_stat[cv2.CC_STAT_TOP]],\
            [r_top_x,r_stat[cv2.CC_STAT_TOP]],\
            [r_stat[cv2.CC_STAT_LEFT]+r_stat[cv2.CC_STAT_WIDTH],r_stat[cv2.CC_STAT_TOP]+r_stat[cv2.CC_STAT_HEIGHT]],\
            [new_point[0],new_point[1]]],\
           closed=False
          )
    else:
        pth = Path([[l_stat[cv2.CC_STAT_LEFT],l_stat[cv2.CC_STAT_TOP] + l_stat[cv2.CC_STAT_HEIGHT] ],\
                [l_top_x,l_stat[cv2.CC_STAT_TOP]],\
                [r_top_x,r_stat[cv2.CC_STAT_TOP]],\
                [r_stat[cv2.CC_STAT_LEFT]+r_stat[cv2.CC_STAT_WIDTH],r_stat[cv2.CC_STAT_TOP]+r_stat[cv2.CC_STAT_HEIGHT]]],\
               closed=False
              )
    # generating path mask
    nr, nc = img_np.T.shape
    ygrid, xgrid = np.mgrid[:nr, :nc]
    xypix = np.vstack((xgrid.ravel(), ygrid.ravel())).T
    mask = pth.contains_points(xypix)
    mask = mask.reshape(img_np.T.shape)
    masked = np.ma.masked_array(img_np.T, ~mask)

    # create mask
    combined = np.ma.mask_or(np.ma.make_mask(mask,dtype=int),np.ma.make_mask(seg_img.T))
    # blackout mask on image
    combined_seg_img = opencxr.utils.mask_crop.set_non_mask_constant(img_np.T,combined)
    
    return combined_seg_img
    