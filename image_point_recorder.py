import cv2
import os
import json
from utils import path2name, contraction
import time
from enum import Enum
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

class Labels(Enum):
    BACKGROUND = 0
    WEED = 1
    CROP = 2

global dictionary
global img
global CWFID_dataset
global window_name

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=280):
    pos_points = coords[labels==1]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green')
    neg_points = coords[labels==0]
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red')
    #ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    #ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def click_event(event, x, y, flags, params):
    """
    function records into the global variable dictionary the coordinates of the points clicked on the
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    msg = "*"
    if event == cv2.EVENT_LBUTTONDOWN:# checking for left mouse clicks
        dictionary['input_points'].append([int(x), int(y)]) # point to add inside the mask
        dictionary['input_labels'].append(1) # point to add inside the mask
        cv2.circle(dictionary['img'], (x, y), radius=2, color=(255, 0, 0), thickness=2)
        cv2.imshow(window_name, dictionary['img'])
        print(f"dictionary['input_points'] = {dictionary['input_points']}")
        print(f"dictionary['input_labels'] = {dictionary['input_labels']}")
        predict_and_show_mask(
            dictionary['predictor'],
            dictionary['input_points'],
            dictionary['input_labels'],
            dictionary['img']
        )
    elif event == cv2.EVENT_RBUTTONDOWN:# checking for right mouse clicks
        dictionary['input_points'].append([int(x), int(y)])  # point to remove from the mask
        dictionary['input_labels'].append(0)  # point to remove from the mask
        cv2.circle(dictionary['img'], (x, y), radius=2, color=(0, 0, 255), thickness=2)
        cv2.imshow(window_name, dictionary['img'])
        print(f"dictionary['input_points'] = {dictionary['input_points']}")
        print(f"dictionary['input_labels'] = {dictionary['input_labels']}")
        predict_and_show_mask(
            dictionary['predictor'],
            dictionary['input_points'],
            dictionary['input_labels'],
            dictionary['img']
        )

def predict_and_show_mask(predictor, input_points, input_labels, img):
    if predictor is not None:
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        input_points = np.array(input_points)
        input_labels = np.array(input_labels)
        show_points(input_points, input_labels, plt.gca())
        plt.axis('on')
        plt.show()
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False  # in order to get just the best mask
        )
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            show_mask(mask, plt.gca())
            show_points(input_points, input_labels, plt.gca())
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()
        return masks, scores, logits
    else:
        return None, None, None

# driver function
if __name__ == "__main__":

    #sam_checkpoint = "sam_vit_h_4b8939.pth"
    #model_type = "vit_h"
    #device = "cpu"
    #sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    #sam.to(device=device)
    #predictor = SamPredictor(sam)

    CWFID_dataset = {
        'annotations': os.path.join(*['CWFID_dataset', 'annotations']),
        'SAM_annotations': os.path.join(*['CWFID_dataset', 'SAM_annotations']),
        'images': os.path.join(*['CWFID_dataset', 'images']),
        'masks': os.path.join(*['CWFID_dataset', 'masks']),
        'SamAutomaticMaskGenerator_masks': os.path.join(*['CWFID_dataset', 'SamAutomaticMaskGenerator_masks']),
        'SamPredictor_masks': os.path.join(*['CWFID_dataset', 'SamPredictor_masks'])
    }

    img_file_list = os.listdir(CWFID_dataset['images'])
    img_file_list = [os.path.join(*[CWFID_dataset['images'], f]) for f in img_file_list if
                      f.endswith('.png')
                      and not f.startswith('.')
                      and os.path.isfile(os.path.join(*[CWFID_dataset['images'], f]))
                      ]
    img_file_list.sort()

    for filename in img_file_list:
        annotation_file = os.path.join(*[CWFID_dataset['SAM_annotations'], f'{contraction(path2name(filename))}.json'])
        if os.path.exists(annotation_file):
            continue    # proceed with annotation only if the annotation does not exist yet
        #### else process this image file

        print(f"Full name: {filename}")
        img = plt.imread(filename)
        # the png image when read becomes a 3D matrix of shape (width, height, 3) with float32 values in the interval [0, 1]
        # SAM predictor does not work on this data type and prefers jpg images which are 3D matrices of shape (width, height, 3) with int values in the interval [0, 255]
        # for example try to see the content of img = plt.imread('truck.jpg') in debugging mode
        img = (img * 255).astype(np.uint8)
        dictionary = {
            'add_points': [],
            'rem_points': [],
            'input_points': [],
            'input_labels': [],
            'img_file': filename,
            'img': img,
            'predictor': None, #predictor,
            'time': 0
        }
        start = time.time()#################
        #predictor.set_image(img)
        window_name = filename
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, click_event)
        # displaying the image
        cv2.imshow(window_name, dictionary['img'])
        # setting mouse handler for the image
        # and calling the click_event() function
        # wait for a key to be pressed to exit
        k = cv2.waitKey(0)
        if k == 0 or k == 27:
            break
        # close the window
        cv2.destroyAllWindows()
        end = time.time()
        dictionary['time'] = end-start
        jsonfile = os.path.join(*[CWFID_dataset['SAM_annotations'], f"{contraction(path2name(filename))}.json"])
        with open(jsonfile, "w") as outfile:
            json.dump({
            'add_points': dictionary['add_points'],
            'rem_points': dictionary['rem_points'],
            'input_points': dictionary['input_points'],
            'input_labels': dictionary['input_labels'],
            'img_file': dictionary['img_file'],
            'time': dictionary['time']
        }, outfile)

        print(f"Iteration on file {filename} completed. File {jsonfile} saved.")
    print("All images have been annotated.")
    pass