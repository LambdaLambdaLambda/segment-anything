import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd

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

dest_filename = 'mask_ground_truth.csv'
df = None
stop = False

if os.path.isfile(dest_filename):
    df = pd.read_csv(dest_filename)  # load dataframe from existing csv file
else:
    df = pd.DataFrame(columns=['picture_file', 'sam_mask_file', 'label'])

for k in ESCA_dataset.keys():
    if stop == True:
        break
    mask_list = sorted(os.listdir(ESCA_dataset[k]['masks']))
    mask_list = [x for x in mask_list if x.endswith('.ome.tiff')]
    picture_list = sorted(os.listdir(ESCA_dataset[k]['pictures']))
    picture_list = [x for x in picture_list if x.endswith('.jpg') or x.endswith('.JPG')]
    SAM_mask_list = sorted(os.listdir(ESCA_dataset[k]['SAM_masks']))
    for p_name in picture_list:
        if stop == True:
            break
        p_name_full = os.path.join(*[ESCA_dataset[k]['pictures'], p_name])
        m_name_full = os.path.join(*[ESCA_dataset[k]['masks'], f"{p_name[:-4]}_finalprediction.ome.tiff"])
        SAM_single_mask_list = sorted(os.listdir(os.path.join(*[ESCA_dataset[k]['SAM_masks'], p_name[:-4]])))
        SAM_single_mask_list = [x for x in SAM_single_mask_list if x.endswith('.png') and not x.startswith(p_name[:-4])]
        for s_name in SAM_single_mask_list:
            if stop == True:
                break
            s_name_full = os.path.join(*[ESCA_dataset[k]['SAM_masks'], f"{p_name[:-4]}", s_name])
            s = plt.imread(s_name_full)
            p = plt.imread(p_name_full)
            rgba = cv2.cvtColor(p, cv2.COLOR_RGB2RGBA)
            rgba[:, :, 3] = np.where(s == 0, 127, 255).astype('uint8')
            plt.imshow(rgba)
            plt.title('Immagine RGB con maschera sovrapposta')
            plt.show()
            choice = input("Leaf? [y/n] Enter 'q' to quit. If you quit a csv file will be saved.")
            if choice == 'q':
                df.to_csv(dest_filename, index=False, header=True)
                stop = True
            elif choice == 'y':
                new_row = {
                    'picture_file': p_name_full,
                    'sam_mask_file': s_name_full,
                    'label': 'leaf'
                }
                df.loc[len(df)] = [p_name_full, s_name_full, 'leaf']
            elif choice == 'n':
                new_row = {
                    'picture_file': p_name_full,
                    'sam_mask_file': s_name_full,
                    'label': 'no_leaf'
                }
                df.loc[len(df)] = [p_name_full, s_name_full, 'no_leaf']

