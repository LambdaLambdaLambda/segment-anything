import os
import numpy as np
import matplotlib.pyplot as plt
from utils import contraction
from PIL import Image

CWFID_dataset = {
    'annotations': os.path.join(*['CWFID_dataset', 'annotations']),
    'SAM_annotations': os.path.join(*['CWFID_dataset', 'SAM_annotations']),
    'images': os.path.join(*['CWFID_dataset', 'images']),
    'masks': os.path.join(*['CWFID_dataset', 'masks']),
    'SamPredictor_masks': os.path.join(*['CWFID_dataset', 'SamPredictor_masks']),
    'SamAutomaticMaskGenerator_masks': os.path.join(*['CWFID_dataset', 'SamAutomaticMaskGenerator_masks'])
}

def plot_all(filename_list):
    for i, filename in enumerate(filename_list):
        plt.title(f"File: {filename}")
        img = plt.imread(filename)
        min_val = img.min()
        max_val = img.max()
        plt.imshow(img, cmap='gray', vmin=min_val, vmax=max_val)
        plt.show()
    pass
def main():
    """
    file_list = [
        os.path.join(*[CWFID_dataset['SamPredictor_masks'], f]) for f in os.listdir(CWFID_dataset['SamPredictor_masks']) if
            f.endswith('.tiff')
            and not f.startswith('.')
            and os.path.isfile(os.path.join(*[CWFID_dataset['SamPredictor_masks'], f]))
        ]
    file_list = [
        os.path.join(*[CWFID_dataset['SamAutomaticMaskGenerator_masks'], f]) for f in os.listdir(CWFID_dataset['SamAutomaticMaskGenerator_masks'])
        if
        f.endswith('.tiff')
        and not f.startswith('.')
        and os.path.isfile(os.path.join(*[CWFID_dataset['SamAutomaticMaskGenerator_masks'], f]))
    ]
    """
    file_list = [
        os.path.join(*[CWFID_dataset['masks'], f]) for f in os.listdir(CWFID_dataset['masks'])
        if
        f.endswith('.tiff')
        and not f.startswith('.')
        and os.path.isfile(os.path.join(*[CWFID_dataset['masks'], f]))
    ]
    file_list.sort()  # contains ordered list of full paths only of yaml files inside yaml_folder
    plot_all(file_list)
    pass

if __name__ == "__main__":
    main()
