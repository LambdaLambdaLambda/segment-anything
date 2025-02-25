import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import json
from PIL import Image

### THE MASKS HAVE 255 FOR BACKGROUND PIXELS AND 0 FOR FOLIAGE PIXELS
FOLIAGE = 0
BACKGROUND = 255

def merge_SAM_masks(containing_folder):
    """
    @:containing_folder: full path to a folder that contains a subfolder for each image processed by SAM.
                Example:
                SamAutomaticMaskGenerator_masks --> containing folder
                SamAutomaticMaskGenerator_masks/002_image/0.png
                ...
                SamAutomaticMaskGenerator_masks/002_image/165.png
                SamAutomaticMaskGenerator_masks/002_image/metadata.csv
    :return: does not return a value. For each subfolder like SamAutomaticMaskGenerator_masks/002_image/ creates a file
                SamAutomaticMaskGenerator_masks/002_mask.tiff that is the merge of all individual object masks
                SamAutomaticMaskGenerator_masks/002_image/0.png, ..., SamAutomaticMaskGenerator_masks/002_image/165.png
                The final tiff file is a binary 2D image whose pixels contain 1 for the Region Of Interest and
                0 for the background
    """
    mask_folder_list = [os.path.join(*[containing_folder, x]) for x in os.listdir(containing_folder) if
                        os.path.isdir(os.path.join(*[containing_folder, x]))
                        and (not x.endswith('.csv'))
                        and (not x.endswith('.png'))
                        and (not x.startswith('.'))
                        ]
    mask_folder_list.sort()
    print(f"Executing merge_SAM_masks({containing_folder})")
    print(f"mask_folder_list has {len(mask_folder_list)} elements.")
    for folder in mask_folder_list:
        # SAM generates a "metadata.csv" file inside containing_folder, and this file should not be processed by this function
        # MacOS also puts a file '.DS_Store' inside every folder
        mask_file_list = [os.path.join(*[folder, file]) for file in os.listdir(folder) if
                          os.path.isfile(os.path.join(*[folder, file]))
                          and file.endswith('.png')
                          and (not file.endswith('.csv'))
                          and (not file.startswith('.'))
                          ]
        mask_file_list.sort()
        print(f"{folder} has {len(mask_file_list)} elements.")
        result = None
        if len(mask_file_list) > 0:
            for i, filename in enumerate(mask_file_list):
                msk = plt.imread(filename)
                if i == 0:
                    result = np.zeros(msk.shape)#all generated masks have the same shape (equal to the container image)
                else:
                    result = np.logical_or(result, msk)
            dest_file = os.path.join(*[containing_folder, f"{path2name(folder)[:3]}_mask.tiff"])
            Image.fromarray(result.astype('uint8'), mode='L').save(dest_file)
            print(f"File {dest_file} saved.")
    pass

def contraction(filename):
    possible_suffixes = [".jpg", ".JPG", ".png", ".tiff", ".mat", ".json", ".yaml", ".jpeg"]
    for suffix in possible_suffixes:
        if filename.endswith(suffix):
            result = filename[:-len(suffix)]
            return result
    return filename

def path2name(p_name_full):
    path = os.path.normpath(p_name_full)
    parts = path.split(os.sep)
    p_name = parts[-1]
    return p_name

def plot_foreground_background(image, mask):
    """
    :param image: a 3D numpy matrix of dimension (rows, cols, 3) representing an RGB image
    :param mask: a 2D numpy matrix of dimension (rows, cols) representing black/white mask
            generated by SAM for image. The mask contains value 255 for every foreground pixel and 0
            for every background pixel
    """
    plt.imshow(image)
    plt.title('original picture')
    plt.show()
    plt.imshow(mask)
    plt.title('SAM mask reuslt of merging all detected objects')
    plt.show()
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    r_fg = np.where(mask == 0, 0, r).astype('uint8')
    g_fg = np.where(mask == 0, 0, g).astype('uint8')
    b_fg = np.where(mask == 0, 0, b).astype('uint8')
    fg = np.dstack(tup=(r_fg, g_fg, b_fg))
    plt.imshow(fg)
    plt.title('Foreground only')
    plt.show()
    r_bg = np.where(mask == 0, r, 0).astype('uint8')
    g_bg = np.where(mask == 0, g, 0).astype('uint8')
    b_bg = np.where(mask == 0, b, 0).astype('uint8')
    bg = np.dstack(tup=(r_bg, g_bg, b_bg))
    plt.imshow(bg)
    plt.title('Background only')
    plt.show()
    pass


def pad_RGB_image(image, new_rows, new_cols):
    """
    :param image: a 3D numpy matrix of dimension (rows, cols, 3) representing an RGB image
    :param new_rows: and integer >= rows
    :param new_cols: and integer >= cols
    :return: a 3D numpy matrix of dimension (new_rows, new_cols, 3) representing a new RGB image obtained by
           gluing black rectangular stripes to the right and bottom edges of image in order to reach the prescribed
           dimensions
    """
    rows = image.shape[0]
    cols = image.shape[1]
    # split RGB channels, pad each one separately with zeros and then merge the three padded layers
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    dr = new_rows - rows
    dc = new_cols - cols
    padded_r = np.pad(r, ((0, dr), (0, dc)), mode='constant')
    padded_g = np.pad(g, ((0, dr), (0, dc)), mode='constant')
    padded_b = np.pad(b, ((0, dr), (0, dc)), mode='constant')
    return np.dstack(tup=(padded_r, padded_g, padded_b))


def pad_binary_mask(mask, new_rows, new_cols):
    """
    :param mask: a 2D numpy matrix of dimension (rows, cols) representing a binary mask
    :param new_rows: and integer >= rows
    :param new_cols: and integer >= cols
    :return: a 2D numpy matrix of dimension (new_rows, new_cols) representing a new binary mask obtained by
             gluing black rectangular stripes to the right and bottom edges of mask in order to reach the prescribed
             dimensions
    """
    rows = mask.shape[0]
    cols = mask.shape[1]
    dr = new_rows - rows
    dc = new_cols - cols
    # glue black rectangular stripes to the right and bottom edges of fg in order to reach the prescribed dimensions
    return np.pad(mask, ((0, dr), (0, dc)), mode='constant')


def affine_transform(image, scale_factor, src_point, dst_point):
    """
    :param image: a 2D or 3D numpy matrix representing either a binary or an RGB image
    :param scale_factor: a positive real number
    :param src_point: an integer numpy array of shape (2,)
    :param dst_point: an integer numpy array of shape (2,)
    :return: a 2D or 3D numpy matrix representing either a binary or an RGB image resulting
             from the unique affine transformation (x,y) ---> scale_factor*(x,y) + (a,b)
             that maps src_point to dst_point
    """
    sf = scale_factor
    sp = src_point
    dp = dst_point
    # WARNING: the coordinates of a point P = [x, y] in the 2D-plane have inverted roles w.r.t. usual function plots
    #         x is the row and y is the column of the point. The matrix representing affine transform multiplication
    #         acts accordingly by multiplication with [[y],[x],[1]]
    M = np.array([sf, 0, dp[1] - sf * sp[1], 0, sf, dp[0] - sf * sp[0]]).reshape(2, 3)
    rows = image.shape[0]
    cols = image.shape[1]
    return cv2.warpAffine(image, M, (cols, rows))


def quaternary_merge(fg, fg_msk, bg, bg_msk, special_image_r, special_image_g, special_image_b):
    """
    :param fg: a 3D numpy matrix of dimension (fg_rows, fg_cols, 3) representing an RGB image
    :param fg_msk: a 2D numpy matrix of dimension (fg_rows, fg_cols) representing a segmentation mask whose
                    cells contain numbers between 0 and the numbers of values in the Enum Segments (call it N_classes).
    :param bg: a 3D numpy matrix of dimension (bg_rows, bg_cols, 3) representing an RGB image
    :param bg_msk: a 2D numpy matrix of dimension (bg_rows, bg_cols) representing a segmentation mask whose
                    cells contain numbers between 0 and N_classes.
    :param fg_class: the value 'esca' is passed when merging esca foliage over healthy background. In this case
                      the pixels that are covered neither by esca foliage nor by healthy backgound will be taken from
                      healthy foliage
                      the value 'healthy' is passed when merging healthy foliage over esca background. In this case
                      the pixels that are covered neither by healthy foliage nor by esca background will be taken from
                      healthy background
    :return: a 3D numpy matrix of dimension (bg_rows, bg_cols, 3) representing an RGB image
             obtained by superposition of the region of interest of the foreground
             upon the region of interest of the background. The co-region of interest of the foreground
             is covered by the co-region of interest of the background.
    """
    red_fg, green_fg, blue_fg = fg[:, :, 0], fg[:, :, 1], fg[:, :, 2]
    red_bg, green_bg, blue_bg = bg[:, :, 0], bg[:, :, 1], bg[:, :, 2]
    special_image_r = special_image_r[:bg.shape[0], :bg.shape[1]]
    special_image_g = special_image_g[:bg.shape[0], :bg.shape[1]]
    special_image_b = special_image_b[:bg.shape[0], :bg.shape[1]]
    # PAY ATTENTION: in blending foreground and background together, bg_mask operates opposite to fg_mask
    # before the merge the background ROI is filled with the foreground ROI
    # (in a first approximation we replace use the background ROI with that of a predefined image)
    red_bg = np.multiply(bg_msk, special_image_r) + np.multiply(1 - bg_msk, red_bg)
    green_bg = np.multiply(bg_msk, special_image_g) + np.multiply(1 - bg_msk, green_bg)
    blue_bg = np.multiply(bg_msk, special_image_b) + np.multiply(1 - bg_msk, blue_bg)
    # because when fg_msk[r][c] == 0 and bg_msk[r][c] == 1 --> blending_mask[r][c] = 0
    # This way the pixel will come from background ROI, which has been filled with the foreground ROI so that
    # when the foreground image is healthy and the background image is esca, no esca foliage will appear in the final merge
    # fg_msk[r][c] == 1 and bg_msk[r][c] == 1 --> blending_mask[r][c] = 1 (the pixel will come from foreground ROI)
    # fg_msk[r][c] == 0 and bg_msk[r][c] == 0 --> blending_mask[r][c] = 0 (the pixel will come from background co-ROI)
    # fg_msk[r][c] == 1 and bg_msk[r][c] == 0 --> blending_mask[r][c] = 1 (the pixel will come from foreground ROI)
    # fg_msk[r][c] == 0 and bg_msk[r][c] == 1 --> blending_mask[r][c] = 0 (the pixel will come from background ROI)
    # 0 op 0 = 0
    # 0 op 1 = 0
    # 1 op 0 = 1
    # 1 op 1 = 1
    # (A && B) || (A && ~B)
    blending_mask = np.add(np.multiply(fg_msk, bg_msk), np.multiply(fg_msk, 1 - bg_msk))
    opposite_blending_mask = 1 - blending_mask
    red = np.multiply(red_fg, blending_mask) + np.multiply(red_bg, opposite_blending_mask)
    green = np.multiply(green_fg, blending_mask) + np.multiply(green_bg, opposite_blending_mask)
    blue = np.multiply(blue_fg, blending_mask) + np.multiply(blue_bg, opposite_blending_mask)
    return np.dstack(tup=(red, green, blue))  # still added black margins need to be cropped out


def plot_avg_color_histogram(obj, label):
    plt.plot(obj["r_hist"], color='r')
    plt.xlim([0, 256])
    plt.plot(obj["g_hist"], color='g')
    plt.xlim([0, 256])
    plt.plot(obj["b_hist"], color='b')
    plt.xlim([0, 256])
    plt.title(f'Histogram for backgrounds of {label} elaborated_pictures')
    plt.show()


def image_to_mask_name(filename):
    """
    :param filename: healthy_875_cam3.jpg
    :return:         healthy_875_cam3_finalprediction.ome.tiff
    """
    suffix = ".jpg"
    i = filename.index(suffix)
    prefix = filename[: i]
    return prefix + '_finalprediction.ome.tiff'


def mask_to_image_name(filename):
    """
    :param filename: healthy_875_cam3_finalprediction.ome.tiff
    :return:         healthy_875_cam3.jpg
    """
    suffix = "_finalprediction.ome.tiff"
    i = filename.index(suffix)
    prefix = filename[: i]
    return prefix + '.jpg'


def prefix_of_mask_name(filename):
    """
    :param filename: healthy_875_cam3_finalprediction.ome.tiff
    :return:         healthy_875_cam3
    """
    suffix = "_finalprediction.ome.tiff"
    i = filename.index(suffix)
    return filename[: i]


def prefix_of_image_name(filename):
    """
    :param filename: healthy_875_cam3.jpg
    :return:         healthy_875_cam3_finalprediction.ome.tiff
    """
    suffix = ".jpg"
    i = filename.index(suffix)
    return filename[: i]

def find_unused_pictures(obj):
    """
    :param obj: can be either the healthy or esca dictionary object
    :return: fills the fields obj["unused_picture_list"] and obj["unused_mask_list"]
             with filenames healthy_yyy_cam3 that are in obj["picture_list"] and obj["mask_list"]
             but do not appear as background in any file #esca_xxx_cam1_over_healthy_yyy_cam3.jpg
    """
    #esca["picture_list"] = sorted(os.listdir(esca["picture_dir"]))
    pass

def compute_background_histograms(obj):
    """
    :param obj: a dictionary that has fields {
        "picture_dir": ...,
        "mask_dir": ...,
        "r_hist": ...,
        "g_hist": ...,
        "b_hist": ...
    :return: computes the three arrays obj["x_hist"] representing average histograms of the three color channels
    associated to the masked images of the obj["picture_dir"] directory. The images are masked in such a way that
    only the background region is taken into account
    """
    index = 0
    for filename in obj["picture_list"]:
        # filename does not have full path
        # the elements of mask_list does are not full paths
        mask_file = image_to_mask_name(filename)
        if mask_file != '':  # if the corresponding mask file has been found
            image = plt.imread(os.path.join(*[obj["picture_dir"], filename]))
            mask = plt.imread(os.path.join(*[obj['mask_dir'], mask_file]))
            ### CONVERT PIXEL VALUES TO 1 FOR FOLIAGE PIXELS AND 0 FOR BACKGROUND PIXELS
            mask = np.where(mask == BACKGROUND, 0, 1).astype('uint8')
            r_hist = cv2.calcHist([image], [0], mask, [256], [0, 256])
            obj["r_hist"] = np.add(obj["r_hist"], r_hist)
            g_hist = cv2.calcHist([image], [1], mask, [256], [0, 256])
            obj["g_hist"] = np.add(obj["g_hist"], g_hist)
            b_hist = cv2.calcHist([image], [2], mask, [256], [0, 256])
            obj["b_hist"] = np.add(obj["b_hist"], b_hist)
        else:
            print(f'The picture {filename} does not have a corresponding ,mask')
    n = len(obj["picture_list"])
    obj["r_hist"] = obj["r_hist"] / n
    obj["g_hist"] = obj["g_hist"] / n
    obj["b_hist"] = obj["b_hist"] / n
    # for cell in obj["r_hist"]:
    #    cell[0] = cell[0] / n
    # for cell in obj["g_hist"]:
    #    cell[0] = cell[0] / n
    # for cell in obj["b_hist"]:
    #    cell[0] = cell[0] / n


def save_results_to_file(outfile, result_list):
    """
    :param outfile: the name of the not-yet-existing file in which json data must be written.
    :param result_list: the array of json objects representing the data to be written to the file.
    :return: an array of python dictionaries
    """
    f = open(outfile, "w")
    y = json.dumps(result_list)
    f.write(y)
    f.close()


def read_results_from_file(outfile):
    """
    :param outfile: the existing file in which json data are written. Contains an array of json objects.
    :return: an array of python dictionaries
    """
    f = open(outfile, "r")
    x = f.read()
    data = json.loads(x)
    f.close()
    return data

def compute_best_match_for_given_bg_mask(bg_msk, bg_img, dict, foreground_obj):
    max_common_pixels = -1  # reset this value for each iteration of the for loop at line 225
    for fg_msk_name in foreground_obj["mask_list"]:
        fg_msk_fullname = os.path.join(*[foreground_obj["mask_dir"], fg_msk_name])
        if os.path.isfile(fg_msk_fullname) and fg_msk_fullname != os.path.join(
                *[foreground_obj["mask_dir"], '.DS_Store']):
            fg_msk = plt.imread(fg_msk_fullname)
            fg_msk = np.where(fg_msk == BACKGROUND, 0, 1).astype('uint8')  # BINARIZE THE MASK
            fg_img_name = mask_to_image_name(fg_msk_name)
            fg_img_fullname = os.path.join(*[foreground_obj["picture_dir"], fg_img_name])
            fg_img = (1 / 255) * plt.imread(fg_img_fullname)  # CAREFUL: CHECK THIS IS WELL DEFINED
            # HERE AN IMAGE IS ADJUSTED WHEN IT HAS BEEN FIRST PADDED AND THEN AFFINE TRANSFORMED
            (adjusted_fg_img, adjusted_fg_msk, padded_bg_img, padded_bg_msk) = adjust_fg_to_bg(fg_msk, fg_img, bg_msk,
                                                                                               bg_img)
            # if the function adjust_fg_to_bg returns correct values
            # the variable common_pixels will contain the number of
            # pixels that are foliage (value 1) both in adjusted_fg_msk and in padded_bg_msk
            intersection = np.multiply(adjusted_fg_msk, padded_bg_msk)
            common_pixels = np.count_nonzero(intersection)
            # common_pixels is the number of
            if common_pixels > max_common_pixels:
                max_common_pixels = common_pixels
                dict["fg_img_shape"] = fg_img.shape
                dict["fg_msk_shape"] = fg_msk.shape
                dict["fg_img_name"] = fg_img_name
                dict["fg_msk_name"] = fg_msk_name
                dict["adjusted_fg_img"] = adjusted_fg_img
                dict["adjusted_fg_msk"] = adjusted_fg_msk
                dict["padded_bg_img"] = padded_bg_img
                dict["padded_bg_msk"] = padded_bg_msk
                result = quaternary_merge(adjusted_fg_img, adjusted_fg_msk, padded_bg_img, padded_bg_msk,
                                          foreground_obj["special_image_r"], foreground_obj["special_image_g"],
                                          foreground_obj["special_image_b"])
                cropped_result = result[:bg_img.shape[0], :bg_img.shape[1]]
                dict["merged_image_name"] = prefix_of_image_name(dict["fg_img_name"]) + '_over_' + prefix_of_image_name(
                    dict["bg_img_name"]) + '.jpg'
                dict["merged_image"] = cropped_result
                ##################  PLOTS ##################
                # plt.imshow(dict['adjusted_fg_img'])
                # plt.title(dict['fg_img_name'])
                # plt.show()
                # plt.imshow(dict['adjusted_fg_msk'], cmap='gray')
                # plt.title(dict['fg_msk_name'])
                # plt.show()
                # plt.imshow(dict['padded_bg_img'])
                # plt.title(dict['bg_img_name'])
                # plt.show()
                # plt.imshow(dict['padded_bg_msk'], cmap='gray')
                # plt.title(dict['bg_msk_name'])
                # plt.show()
                ############################################


def compute_matching_list_and_merge(foreground_obj, background_obj):
    """
    :param foreground_obj:
    :param background_obj:
    :return: for every mask im1 of elaborated_pictures in the class associated to the background_obj,
     select a corresponding mask im2 such that:
     the complement of the Region-Of_Interest (co-ROI) of im2, once padded c
     (ROI) im2 of the class associated to the foreground_obj to be put over the ROI of im1.
     The selection is made
    """
    result = []
    for bg_msk_name in background_obj["mask_list"]:
        bg_msk_fullname = os.path.join(*[background_obj["mask_dir"], bg_msk_name])
        if os.path.isfile(bg_msk_fullname) and bg_msk_fullname != os.path.join(
                *[background_obj["mask_dir"], '.DS_Store']):
            bg_msk = plt.imread(bg_msk_fullname)
            bg_img_name = mask_to_image_name(bg_msk_name)
            if bg_img_name not in background_obj["used_bg_list"]:
                ### CONVERT PIXEL VALUES TO 1 FOR FOLIAGE PIXELS AND 0 FOR BACKGROUND PIXELS
                bg_msk = np.where(bg_msk == BACKGROUND, 0, 1).astype('uint8')  # BINARIZE THE MASK
                bg_img = (1 / 255) * plt.imread(
                    os.path.join(*[background_obj["picture_dir"], bg_img_name]))  # CAREFUL: CHECK THIS IS WELL DEFINED
                dict = {
                    "bg_img_name": bg_img_name,  # only filename, no full path
                    "bg_msk_name": bg_msk_name,  # only filename, no full path
                    "bg_img_shape": bg_img.shape,
                    "bg_msk_shape": bg_msk.shape,
                    "fg_img_name": None,  # only filename, no full path
                    "fg_msk_name": None,  # only filename, no full path
                    "fg_img_shape": None,
                    "fg_msk_shape": None,
                    "adjusted_fg_img": None,
                    "adjusted_fg_msk": None,
                    "padded_bg_img": None,
                    "padded_bg_msk": None,
                    "merged_image_name": None,
                    "merged_image": None
                }
                compute_best_match_for_given_bg_mask(bg_msk, bg_img, dict, foreground_obj)
                result = np.append(result, dict)
                # MERGED PICTURE BELONGS TO THE FOREGROUND CLASS
                new_name = os.path.join(*[foreground_obj['merge_dir'], dict["merged_image_name"]])
                Image.fromarray((dict["merged_image"] * 255).astype(np.uint8)).save(new_name)
                ##################  PLOTS ##################
                # plt.imshow(dict['adjusted_fg_img'])
                # plt.title(dict['fg_img_name'])
                # plt.show()
                # plt.imshow(dict['adjusted_fg_msk'], cmap='gray')
                # plt.title(dict['fg_msk_name'])
                # plt.show()
                # plt.imshow(dict['padded_bg_img'])
                # plt.title(dict['bg_img_name'])
                # plt.show()
                # plt.imshow(dict['padded_bg_msk'], cmap='gray')
                # plt.title(dict['bg_msk_name'])
                # plt.show()
                ############################################
    return result


def adjust_fg_to_bg(fg_msk, fg_img, bg_msk, bg_img):
    """
    :param fg_msk:
    :param fg_img:
    :param bg_msk:
    :param bg_img:
    :return:
    """
    fg_poi = np.count_nonzero(fg_msk)  # Pixels Of Interest in the foreground
    bg_poi = np.count_nonzero(bg_msk)  # Pixels Of Interest in the background
    if fg_poi != 0 and bg_poi != 0:
        poi_ratio = bg_poi / fg_poi  # ratio of Pixels Of Interest between background and foreground
        sf = np.sqrt(poi_ratio)  # scale factor
        # NOTE: padding operations do not alter neither poi_ratio, nor sf, nor center of mass because padding
        #       extends the axes of the picture downward and on the right. Every pixel of the region of interest in the
        #       original mask maintains same coordinates in the padded mask
        max_rows = max(fg_img.shape[0], bg_img.shape[0])
        max_cols = max(fg_img.shape[1], bg_img.shape[1])
        padded_fg_img = pad_RGB_image(fg_img, max_rows, max_cols)
        padded_fg_msk = pad_binary_mask(fg_msk, max_rows, max_cols)
        padded_bg_img = pad_RGB_image(bg_img, max_rows, max_cols)
        padded_bg_msk = pad_binary_mask(bg_msk, max_rows, max_cols)
        ##################  PLOTS ##################
        # plt.imshow(padded_fg_img)
        # plt.title('Padded foreground')
        # plt.show()
        # plt.imshow(padded_fg_msk, cmap='gray')
        # plt.title('Padded foreground binary mask')
        # plt.show()
        # plt.imshow(padded_bg_img)
        # plt.title('Padded background')
        # plt.show()
        # plt.imshow(padded_bg_msk, cmap='gray')
        # plt.title('Padded background binary mask')
        # plt.show()
        ############################################
        ###################  SAVE ALL IMAGES PRODUCED ##################
        # Image.fromarray((fg_img*255).astype(np.uint8)).save('pad_fg.jpg')
        # Image.fromarray((fg_msk * 255).astype('uint8'), mode='L').save('pad_fg_mask.tiff')
        # Image.fromarray((bg_img * 255).astype(np.uint8)).save('pad_bg.jpg')
        # Image.fromarray((bg_mask * 255).astype('uint8'), mode='L').save('pad_bg_mask.tiff')
        # WARNING: the function quaternary_merge takes in images and binary masks
        ############################################
        # row and column of center-of-mass of foreground region of interest
        mts = cv2.moments(padded_fg_msk.astype('uint8'))
        fg_mask_centroid_row = int(mts["m01"] / mts["m00"])
        fg_mask_centroid_col = int(mts["m10"] / mts["m00"])
        fg_com = np.array([fg_mask_centroid_row, fg_mask_centroid_col])
        # row and column of center-of-mass of background region of interest
        mts = cv2.moments(padded_bg_msk.astype('uint8'))
        bg_mask_centroid_row = int(mts["m01"] / mts["m00"])
        bg_mask_centroid_col = int(mts["m10"] / mts["m00"])
        bg_com = np.array([bg_mask_centroid_row, bg_mask_centroid_col])
        #
        adjusted_fg_img = affine_transform(padded_fg_img, sf, fg_com, bg_com)
        #
        # plt.imshow(adjusted_fg_img)
        # plt.title('Affine_transform(pad(foreground image))')
        # plt.show()
        # Image.fromarray((fg_img*255).astype(np.uint8)).save('trans_pad_fg.jpg')
        #
        adjusted_fg_msk = padded_fg_msk.astype('float64')
        adjusted_fg_msk = affine_transform(adjusted_fg_msk, sf, fg_com, bg_com)
        adjusted_fg_msk = adjusted_fg_msk.astype('uint8')
        #
        # plt.imshow(adjusted_fg_msk, cmap='gray')
        # plt.title('Affine_transform(pad(foreground mask))')
        # plt.show()
        # Image.fromarray((fg_msk * 255).astype('uint8'), mode='L').save('trans_pad_fg_mask.tiff')
        #
        # PAY ATTENTION: the affine transform translates and scales the region of interest of the foreground to slide it
        #                onto the region of interest of the background. The edges of the entire foreground are scaled by a
        #                factor of np.sqrt(poi_ratio) because this scales the area of the foreground region of interest by
        #                a factor poi_ratio, therefore equating the area of the background region of interest
        return adjusted_fg_img, adjusted_fg_msk, padded_bg_img, padded_bg_msk
    else:
        return None, None, None, None
"""
# SHOW COLOR HISTOGRAM TO VERIFY THAT THE INTEGER VALUE Segments.LEAF.value
# IS PRESENT IN THE MASKS
dst = cv2.calcHist(fg_mask, [0], None, [maxValue], [0, maxValue])
plt.hist(fg_mask.ravel(), maxValue, [0, maxValue])
# plt.title(f'Histogram for image {fg_multiclass_mask_file}')
plt.show()
dst = cv2.calcHist(bg_mask, [0], None, [maxValue], [0, maxValue])
plt.hist(bg_mask.ravel(), maxValue, [0, maxValue])
# plt.title(f'Histogram for image {bg_multiclass_mask_file}')
plt.show()
# binarize the segmentation masks in order to isolate regions of interest of the prescribed classes
bin_fg_mask = np.where(fg_mask == Segments.FOLIAGE.value, 1, 0)
bin_bg_mask = np.where(bg_mask == Segments.FOLIAGE.value, 1, 0)
"""
