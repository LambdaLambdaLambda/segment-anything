import yaml
import io
import os
from enum import Enum
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

class Labels(Enum):
    BACKGROUND = 0
    WEED = 1
    CROP = 2

CWFID_dataset = {
    'annotations': os.path.join(*['CWFID_dataset', 'annotations']),
    'images': os.path.join(*['CWFID_dataset', 'images']),
    'masks': os.path.join(*['CWFID_dataset', 'masks']),
    'SAM_masks': os.path.join(*['CWFID_dataset', 'SAM_masks'])
}
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

yaml_file_list = os.listdir(CWFID_dataset['annotations'])
yaml_file_list = [os.path.join(*[CWFID_dataset['annotations'], f]) for f in yaml_file_list if
                  f.endswith('.yaml')
                  and not f.startswith('._')
                  and os.path.isfile(os.path.join(*[CWFID_dataset['annotations'], f]))
                  ]
yaml_file_list.sort() # contains ordered list of full paths only of yaml files inside yaml_folder

for filename in yaml_file_list:
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
        filename = os.path.join(*[CWFID_dataset['images'], data_loaded['filename']])
        print(f"Full name: {filename}")
        for rec in data_loaded['annotation']:
            print(f"{rec['type']}")
            try: # some annotations do not contain coordinates represented as lists
                iterator = iter(rec['points']['x'])
                iterator = iter(rec['points']['y'])
            except TypeError:
                x = int(rec['points']['x'])
                y = int(rec['points']['y'])
                input_points = np.array([[x, y]])
            else:
                input_points = np.array([[int(x), int(y)] for (x, y) in list(zip(rec['points']['x'], rec['points']['y']))])
            n = len(input_points)
            # prepare the array of labels as requested by SAM in predictor mode
            # input_labels[j] == 1 ---> the point input_points[j] belongs to the mask
            # input_labels[j] == 0 ---> the point input_points[j] does not belong to the mask
            input_labels = np.ones(n).astype(int)
        img = plt.imread('truck.jpg')
        img = plt.imread(filename)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        show_points(input_points, input_labels, plt.gca())
        plt.axis('on')
        plt.show()
        predictor.set_image(img)
        masks, scores, logits = predictor.predict(
                                    point_coords=input_points,
                                    point_labels=input_labels,
                                    multimask_output=True
                                )
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(img)
            show_mask(mask, plt.gca())
            show_points(input_points, input_labels, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()
        pass