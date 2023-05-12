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
        r = random.randint(0,  len(picture_list)-1)
        for i, p_name in enumerate(picture_list):
            if i != r:
                continue
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
                s = plt.imread(s_name_full)
                p = plt.imread(p_name_full)
                rgba = cv2.cvtColor(p, cv2.COLOR_RGB2RGBA)
                rgba[:, :, 3] = np.where(s == 0, 127, 255).astype('uint8')
                plt.imshow(rgba)
                plt.title(f"Image {p_name} with mask {s_name}")
                plt.show()
                choice = input("Leaf? [y/n] Enter 'q' to quit. If you quit a csv file will be saved.")
                if choice == 'q':
                    df.to_csv(dest_filename, index=False, header=True)
                    stop = True
                elif choice == 'y':
                    df.loc[len(df)] = [p_name_full, s_name_full, 'leaf']
                elif choice == 'n':
                    df.loc[len(df)] = [p_name_full, s_name_full, 'no_leaf']

def show_image_with_mask_in_alpha_channel(img, img_filename, binary_msk, binary_msk_filename, class_name):
    rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    max_opacity = 255
    rgba[:, :, 3] = np.where(binary_msk == 0, 0.4*max_opacity, max_opacity).astype('uint8')
    new_name = os.path.join(*['elaborated_pictures', f"{strip_extension(binary_msk_filename)}_over_{strip_extension(img_filename)}.png"])
    #rgba.convert('RGB')
    Image.fromarray(rgba.astype(np.uint8)).save(new_name)
    plt.imshow(rgba)
    plt.title(f"Image {img_filename} of class {class_name} with mask {binary_msk_filename}\n\n")
    plt.show()
def print_two_pictures_with_masks():
    for k in ESCA_dataset.keys():
        picture_list = sorted(os.listdir(ESCA_dataset[k]['pictures']))
        picture_list = [x for x in picture_list if x.endswith('.jpg') or x.endswith('.JPG')]
        r = random.randint(0, len(picture_list) - 1)
        #Random choice of one picture
        p_name = picture_list[r]
        p_name_full = os.path.join(*[ESCA_dataset[k]['pictures'], p_name])
        p = plt.imread(p_name_full)
        m_name_full = os.path.join(*[ESCA_dataset[k]['masks'], f"{strip_extension(p_name)}_finalprediction.ome.tiff"])
        m = plt.imread(m_name_full)
        show_image_with_mask_in_alpha_channel(p, p_name, m, f"{strip_extension(p_name)}_finalprediction.ome.tiff", k)
        SAM_single_mask_list = sorted(os.listdir(os.path.join(*[ESCA_dataset[k]['SAM_masks'], strip_extension(p_name)])))
        SAM_single_mask_list = [x for x in SAM_single_mask_list if x.endswith('.png') and not x.startswith(strip_extension(p_name))]
        for s_name in SAM_single_mask_list:
            s_name_full = os.path.join(*[ESCA_dataset[k]['SAM_masks'], f"{strip_extension(p_name)}", s_name])
            s = plt.imread(s_name_full)
            show_image_with_mask_in_alpha_channel(p, p_name, s, s_name, k)

def main():
    print_two_pictures_with_masks()

if __name__ == "__main__":
    main()