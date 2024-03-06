import cv2, opencxr, os
from pathlib import Path
from matplotlib.path import Path as nodePath
import matplotlib.pyplot as plt
import numpy as np
from opencxr.utils.mask_crop import crop_with_params
from opencxr.utils.file_io import read_file, write_file
from scipy.ndimage import find_objects




# src: https://github.com/DIAGNijmegen/opencxr/blob/master/opencxr/utils/mask_crop.py
# patch: remove None values from find_objects
def crop_to_mask(img_np, spacing, mask_np, margin_in_mm):
    """
    A method to crop away regions outside the bounding box of a mask
    e.g. crop to smallest rectangle containing lungs
    :param img_np: the original image
    :param spacing: The spacing of the original image
    :param mask_np: the mask to crop around, e.g. lung mask,
               expected as binary 1,0 values
    :param margin_in_mm: a margin to allow around the tightest bounding box
    :return: The cropped image
             The size changes list for reference or future use see utils __init__.py
    """
    # convert margin in mm to margin_in_pixels for each of x and y
    margin_in_pixels_x = int(np.round(margin_in_mm / spacing[0]))
    margin_in_pixels_y = int(np.round(margin_in_mm / spacing[1]))

    # get bounding box for mask
    bbox = find_objects(mask_np)
    # ++++++ PATCH: Remove Nones ++++++
    bbox = [i for i in bbox if i is not None]
    # +++++++++++++++++++++++++++++++++
    min_x_mask = max(bbox[0][0].start - margin_in_pixels_x, 0)
    min_y_mask = max(bbox[0][1].start - margin_in_pixels_y, 0)
    max_x_mask = min(bbox[0][0].stop + margin_in_pixels_x, mask_np.shape[0])
    max_y_mask = min(bbox[0][1].stop + margin_in_pixels_y, mask_np.shape[1])

    cropped_img, size_changes = crop_with_params(
        img_np, [min_x_mask, max_x_mask, min_y_mask, max_y_mask]
    )
    return cropped_img, size_changes

def split_to_lung_halves(img_np, seg_result):
    """
    A method split image into two halves, using the midpoint between the
    innermost x coordinates of each lung.

    :param img_np: the original image
    :param seg_result: the segmentation mask of the lungs
    :return: Left and right lung image halves with same dimensions
    """
    # seperate lung masks
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(seg_result.T)
    #drop row with max area (automatic label of full img generated by method)
    stats = np.delete(stats,np.argmax(stats.T[4]), axis=0)
    # calculate the left and right lung segments
    l_stats, r_stats = stats[np.argmin(stats.T[0])],stats[np.argmax(stats.T[0])]
    # identify inner lung points
    l_in = l_stats[cv2.CC_STAT_LEFT]+l_stats[cv2.CC_STAT_WIDTH]
    r_in = r_stats[cv2.CC_STAT_LEFT]
    # lung centroid
    lung_centroid = int(l_in + ((r_in-l_in)/2))
    # crop dims
    crop_x_len = min(int(img_np.shape[0]-lung_centroid),int(0+lung_centroid))

    return img_np[int(lung_centroid-crop_x_len):lung_centroid,0:img_np.shape[1]].T,\
           img_np[lung_centroid:int(lung_centroid+crop_x_len),0:img_np.shape[1]].T


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
        pth = nodePath([[l_stat[cv2.CC_STAT_LEFT],l_stat[cv2.CC_STAT_TOP] + l_stat[cv2.CC_STAT_HEIGHT] ],\
            [l_top_x,l_stat[cv2.CC_STAT_TOP]],\
            [r_top_x,r_stat[cv2.CC_STAT_TOP]],\
            [r_stat[cv2.CC_STAT_LEFT]+r_stat[cv2.CC_STAT_WIDTH],r_stat[cv2.CC_STAT_TOP]+r_stat[cv2.CC_STAT_HEIGHT]],\
            [new_point[0],new_point[1]]],\
           closed=False
          )
    else:
        pth = nodePath([[l_stat[cv2.CC_STAT_LEFT],l_stat[cv2.CC_STAT_TOP] + l_stat[cv2.CC_STAT_HEIGHT] ],\
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

def mask_outside_lungs_and_split(img_np, spacing, seg_img, margin=0, crop=True, flatten_base=False):
    """
    Method to mask all tissue from the lung tissue outward.

    :param img_np: image to be masked, numpy array.
    :param spacing: image load spacing, numpy array.
    :param seg_result: segmentaion mask of lungs, numpy array.
    :param margin: boundary (in pixels) to add to mask border, int.
    :param crop: crop image to segmentation mask, boolean.
    :param :
    :return: Left Image, Right Image
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
        pth = nodePath([[l_stat[cv2.CC_STAT_LEFT],l_stat[cv2.CC_STAT_TOP] + l_stat[cv2.CC_STAT_HEIGHT] ],\
            [l_top_x,l_stat[cv2.CC_STAT_TOP]],\
            [r_top_x,r_stat[cv2.CC_STAT_TOP]],\
            [r_stat[cv2.CC_STAT_LEFT]+r_stat[cv2.CC_STAT_WIDTH],r_stat[cv2.CC_STAT_TOP]+r_stat[cv2.CC_STAT_HEIGHT]],\
            [new_point[0],new_point[1]]],\
           closed=False
          )
    else:
        pth = nodePath([[l_stat[cv2.CC_STAT_LEFT],l_stat[cv2.CC_STAT_TOP] + l_stat[cv2.CC_STAT_HEIGHT] ],\
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
    return split_to_lung_halves(combined_seg_img.T,seg_img)

def process_and_export_full_cxr(img_name, label, train_test, load_dir, \
                                save_dir, cxr_seg_alg, cxr_std_alg,margin=2,\
                                std = True, arch_mask=False, lung_mask=False,\
                                flatten_base=False, out_size = 1024):
    """
    Method to mask all tissue from the lung tissue outward.

    :param img_name: name image to be processed, string.
    :param label: label for image, string.
    :param train: label whether training, test or val, string.
    :param load_dir: path to where images to process are, string.
    :param save_dir: path for image export, child directories of configurations will be applied, string.
    :param cxr_seg_alg: opencxr lung segmentor object.
    :param cxr_std_alg: opencxr cxr standardiser object.
    :param margin: apply margin to crop, int.
    :param std: apply opencxr image standardisation, boolean.
    :param arch_mask: apply arch segmentation, boolean.
    :param lung_mask: apply lung segmentation, boolean.
    :param flatten_base: make bottom of mask flat, boolean.
    """
    # throw exception if both masks are set to True
    if arch_mask and lung_mask:
            raise Exception("Both mask options set to True. Please only specify one mask type to use and try again.")

    # parse configs to pathname
    masking_process = "lung_seg" if lung_mask else "arch_seg" if arch_mask else "crop"
    std_process = "std" if std else "raw"
    base = "flattened" if flatten_base and arch_mask else "unflattened" if arch_mask else ""

    save_out = os.path.join(save_dir,masking_process,base,std_process,str(out_size),train_test)

    if str(label) == "1":
        #nodule sample
        save_out = os.path.join(save_out,"nodule",img_name.split(".")[0]+".png")
    elif str(label) == "0":
        #normal sample
        save_out = os.path.join(save_out,"normal",img_name.split(".")[0]+".png")
    else:
        raise Exception("Invalid label parameter. Please pass 1 or 0 as label.")

    # load image
    print("loading {}".format(os.path.join(load_dir,img_name)))
    img_np, spacing, dcm_tags = read_file(os.path.join(load_dir,img_name))

    if std:
        # apply image standardisation
        img_np, spacing, std_size_changes = \
            cxr_std_alg.run(img_np, spacing, do_crop_to_lung_box=False)
    # make lung seg mask
    seg_img = cxr_seg_alg.run(img_np)

    if arch_mask:
        print("arch masking...")
        # use arch masking function
        out_img_np = mask_outside_lungs(img_np, spacing,seg_img,margin,True,flatten_base)
    elif lung_mask:
        print("lung seg. masking...")
        # mask from segmentor
        out_img_np = opencxr.utils.mask_crop.set_non_mask_constant(img_np, seg_img)
        out_img_np, out_crop_changes = crop_to_mask(out_img_np, spacing, seg_img, margin)

        out_img_np = out_img_np.T
    else:
        print("cropping only...")
        # crop to seg
        out_img_np, out_crop_changes = crop_to_mask(img_np, spacing, seg_img, margin)
        out_img_np = out_img_np.T
    # resize
    out_img_np, out_spacing, out_changes =opencxr.utils.resize_rescale.resize_to_x_y(\
                                                out_img_np, spacing,out_size,out_size)

    print("Saving image to {}".format(save_out))
    #creating save path
    Path("/".join(save_out.split('/')[:-1])).mkdir(parents=True, exist_ok=True)

    out_img_np = out_img_np * (255/out_img_np.max())
    # return (out_img_np)

    #save image
    cv2.imwrite(save_out,out_img_np.astype('uint8'))

    # plt.imsave(save_out, out_img_np, cmap="grey")
    # # BUG: reading image with plt vs opencxr results in different orientation
    # #       ~ not using opencxr after this, so use plt alignment by adding .T
    # # NOTE: Pytorch only supports 8-bit png atm - uint8
    # # # NOTE: image needs normalised to 255 or else encoding becomes garbled due to giant images
    # # # # NOTE: Ok, I give up. Now the images are coming out with tiny values,
    # # # #       and when multiplied by 255, produce floats between 0-16
    # # # #      ~ makes posterised effect, information loss!!! Use CV2 write
    # write_file(save_out, (out_img_np).astype('uint8').T, out_spacing)