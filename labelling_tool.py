import os
import matplotlib.pyplot as plt
import numpy as np

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

def main():
    for k in ESCA_dataset.keys():
        mask_list = sorted(os.listdir(ESCA_dataset[k]['masks']))
        picture_list = sorted(os.listdir(ESCA_dataset[k]['pictures']))
        SAM_mask_list = sorted(os.listdir(ESCA_dataset[k]['SAM_masks']))
        for p_name in picture_list:
            p_name_full = os.path.join(*[ESCA_dataset[k]['pictures'], p_name[:-4]])
            m_name_full = os.path.join(*[ESCA_dataset[k]['masks'], f"{p_name[:-4]}_finalprediction.ome.tiff"])
            SAM_single_mask_list = [x for x in SAM_mask_list if x.endswith('.png') and not x.startswith(p_name[:-4])]
            for s_name in SAM_single_mask_list:
                s_name_full = os.path.join(*[ESCA_dataset[k]['SAM_masks'], f"{p_name[:-4]}", s_name])
                p = plt.imread(p_name_full)
                plt.imshow(p)
                plt.title('prova')
                plt.show()
                s = plt.imread(s_name_full)
                plt.imshow(s)
                plt.title('prova')
                plt.show()
                cont = input("Continue? [y/n]: ")
                if cont == 'n':
                    return

if __name__ == '__main__':
    main()