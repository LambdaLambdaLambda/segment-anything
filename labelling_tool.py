import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import random
from PIL import Image

base_folder_path = ['ESCA_dataset']

ESCA_dataset = {
    'esca': {
        'folder': os.path.join(*(base_folder_path + ['esca'])),
        'esca_foliage_over_healthy_bg': os.path.join(*(base_folder_path + ['esca', 'esca_foliage_over_healthy_bg'])),
        'masks': os.path.join(*(base_folder_path + ['esca', 'masks'])),
        'pictures': os.path.join(*(base_folder_path + ['esca', 'pictures'])),
        'SAM_masks': os.path.join(*(base_folder_path + ['esca', 'SAM_masks']))
    },
    'healthy': {
        'folder': os.path.join(*(base_folder_path + ['healthy'])),
        'healthy_foliage_over_esca_bg': os.path.join(*(base_folder_path + ['esca', 'healthy_foliage_over_esca_bg'])),
        'masks': os.path.join(*(base_folder_path + ['healthy', 'masks'])),
        'pictures': os.path.join(*(base_folder_path + ['healthy', 'pictures'])),
        'SAM_masks': os.path.join(*(base_folder_path + ['healthy', 'SAM_masks']))
    }
}

def strip_extension(filename):
    possible_suffixes = [".ome.tiff", ".jpg", ".JPG", ".png"]
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
def draw_superposition(img_filename, msk_filename, save_pics=False):
    img = plt.imread(img_filename).astype(np.uint8)
    rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    max_opacity = 255
    # ".ome.tiff" masks have 255 inside background pixels and 0 inside foliage pixels
    # ".png" masks instead have 1 inside foliage pixels and 0 inside background pixels
    msk = plt.imread(msk_filename).astype(np.uint8)
    if msk_filename.endswith(".ome.tiff"):
        msk = (msk/255).astype(np.uint8)
        msk = 1-msk
    if msk_filename.endswith(".png"):
        pass #nothing to do
    rgba[:, :, 3] = np.where(msk == 0, 0.4 * max_opacity, max_opacity).astype(np.uint8)
    #mskgray = cv2.cvtColor(msk, cv2.COLOR_RGB2GRAY)
    contours, hierarchy = cv2.findContours(msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    red = (0, 255, 0)
    thickness = 4
    all = -1
    msk_contours = np.ones_like(msk)
    cv2.drawContours(msk_contours, contours, all, red, thickness)
    # modify red channel to draw contours in red
    rgba[:, :, 0] = np.where(msk_contours == 0, 255, rgba[:, :, 0]).astype(np.uint8)
    # green channel to draw contours in red
    rgba[:, :, 1] = np.where(msk_contours == 0, 0, rgba[:, :, 1]).astype(np.uint8)
    # blue channel to draw contours in red
    rgba[:, :, 2] = np.where(msk_contours == 0, 0, rgba[:, :, 2]).astype(np.uint8)
    plt.imshow(rgba)
    plt.title(f"{img_filename} with transparency and\n contours from {msk_filename}")
    plt.show()
    #plt.imshow(msk)
    #plt.title(f"{msk_filename}")
    #plt.show()
    #plt.imshow(msk_contours*255)
    #plt.title(f"Contours of {msk_filename}")
    #plt.show()
    if save_pics:
        new_name = os.path.join(*['elaborated_pictures', f"{strip_extension(path2name(msk_filename))}_over_{strip_extension(path2name(img_filename))}.png"])
        Image.fromarray(rgba.astype(np.uint8)).save(new_name)
def sam_masks_ground_truth_annotation():
    dest_filename = 'mask_ground_truth.csv'
    df = None
    stop = False
    if os.path.isfile(dest_filename):
        df = pd.read_csv(dest_filename)  # load dataframe from existing csv file
    else:
        df = pd.DataFrame(columns=['picture_file', 'sam_mask_file', 'label'])

    for k in ESCA_dataset.keys():
        #mask_list = sorted(os.listdir(ESCA_dataset[k]['masks']))
        #mask_list = [x for x in mask_list if x.endswith('.ome.tiff')]
        #SAM_mask_list = sorted(os.listdir(ESCA_dataset[k]['SAM_masks']))
        picture_list = sorted(os.listdir(ESCA_dataset[k]['pictures']))
        picture_list = [x for x in picture_list if x.endswith('.jpg') or x.endswith('.JPG')]
        for i, p_name in enumerate(picture_list):
            if stop == True:
                break
            p_name_full = os.path.join(*[ESCA_dataset[k]['pictures'], p_name])
            #m_name_full = os.path.join(*[ESCA_dataset[k]['masks'], f"{p_name[:-4]}_finalprediction.ome.tiff"])
            SAM_single_mask_list = sorted(os.listdir(os.path.join(*[ESCA_dataset[k]['SAM_masks'], strip_extension(p_name)])))
            SAM_single_mask_list = [x for x in SAM_single_mask_list if x.endswith('.png') and not x.startswith(strip_extension(p_name))]
            for s_name in SAM_single_mask_list:
                if stop == True:
                    break
                s_name_full = os.path.join(*[ESCA_dataset[k]['SAM_masks'], f"{strip_extension(p_name)}", s_name])
                draw_superposition(p_name_full, s_name_full)
                choice = input("Leaf? [y/n] Enter 'q' to quit. If you quit a csv file will be saved.")
                if choice == 'q':
                    df.to_csv(dest_filename, index=False, header=True)
                    stop = True
                elif choice == 'y':
                    df.loc[len(df)] = [p_name_full, s_name_full, 'leaf']
                elif choice == 'n':
                    df.loc[len(df)] = [p_name_full, s_name_full, 'no_leaf']

def random_picture_choice(pic_class = None):
    if pic_class is None:
        k = random.choice(list(ESCA_dataset.keys()))
    else:
        k = pic_class
    picture_list = sorted(os.listdir(ESCA_dataset[k]['pictures']))
    picture_list = [x for x in picture_list if x.endswith('.jpg') or x.endswith('.JPG')]
    r = random.randint(0, len(picture_list) - 1)
    p_name = picture_list[r]
    p_name_full = os.path.join(*[ESCA_dataset[k]['pictures'], p_name])
    return p_name_full
def sam_mask_showcase(p_name_full=None, pic_class = None, save_pics=False):
    if p_name_full is None:
        if pic_class is None:
            k = random.choice(list(ESCA_dataset.keys()))
        p_name_full = random_picture_choice(k)
    if not p_name_full is None:
        p_name = path2name(p_name_full)
        if pic_class is None:
            if p_name.startswith('esca'):
                k = 'esca'
            if p_name.startswith('healthy'):
                k = 'healthy'
        else:
            assert p_name.startswith(pic_class)
    SAM_single_mask_list = sorted(
        os.listdir(os.path.join(*[ESCA_dataset[k]['SAM_masks'], strip_extension(p_name)])))#p_name is available due to hoisting
    SAM_single_mask_list = [x for x in SAM_single_mask_list if
                                x.endswith('.png') and not x.startswith(strip_extension(p_name))]
    for s_name in SAM_single_mask_list:
        s_name_full = os.path.join(*[ESCA_dataset[k]['SAM_masks'], f"{strip_extension(p_name)}", s_name])
        draw_superposition(p_name_full, s_name_full, save_pics)
    s_name_full = os.path.join(*[ESCA_dataset[k]['SAM_masks'], f"{strip_extension(p_name)}", f"{strip_extension(p_name)}_mask.png"])
    draw_superposition(p_name_full, s_name_full, save_pics)

def apeer_mask_showcase(p_name_full=None, pic_class = None, save_pics=False):
    if p_name_full is None:
        if pic_class is None:
            k = random.choice(list(ESCA_dataset.keys()))
        p_name_full = random_picture_choice(k)
    if not p_name_full is None:
        p_name = path2name(p_name_full)
        if pic_class is None:
            if p_name.startswith('esca'):
                k = 'esca'
            if p_name.startswith('healthy'):
                k = 'healthy'
        else:
            assert p_name.startswith(pic_class)
    s_name_full = os.path.join(*[ESCA_dataset[k]['masks'], f"{strip_extension(p_name)}_finalprediction.ome.tiff"])
    draw_superposition(p_name_full, s_name_full, save_pics)
def main():
    random.seed(10)
    filename = 'ESCA_dataset/healthy/pictures/healthy_033_cam3.jpg'
    sam_mask_showcase(filename, save_pics=True)
    #apeer_mask_showcase(filename)
    #sam_masks_ground_truth_annotation()

if __name__ == "__main__":
    main()