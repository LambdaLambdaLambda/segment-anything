import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import random
from PIL import Image
from utils import contraction, path2name, merge_SAM_masks

CWFID_dataset = {
    'annotations': os.path.join(*['CWFID_dataset', 'annotations']),
    'images': os.path.join(*['CWFID_dataset', 'images']),
    'masks': os.path.join(*['CWFID_dataset', 'masks']),
    'SAM_masks': os.path.join(*['CWFID_dataset', 'SAM_masks']),
    'SAM_annotations': os.path.join(*['CWFID_dataset', 'SAM_annotations'])
}

ESCA_dataset = {
    'esca': {
        'folder': os.path.join(*['ESCA_dataset', 'esca']),
        'esca_foliage_over_healthy_bg': os.path.join(*['ESCA_dataset', 'esca', 'esca_foliage_over_healthy_bg']),
        'masks': os.path.join(*['ESCA_dataset', 'esca', 'masks']),
        'pictures': os.path.join(*['ESCA_dataset', 'esca', 'pictures']),
        'SAM_masks': os.path.join(*['ESCA_dataset', 'esca', 'SAM_masks'])
    },
    'healthy': {
        'folder': os.path.join(*['ESCA_dataset', 'healthy']),
        'healthy_foliage_over_esca_bg': os.path.join(*['ESCA_dataset', 'esca', 'healthy_foliage_over_esca_bg']),
        'masks': os.path.join(*['ESCA_dataset', 'healthy', 'masks']),
        'pictures': os.path.join(*['ESCA_dataset', 'healthy', 'pictures']),
        'SAM_masks': os.path.join(*['ESCA_dataset', 'healthy', 'SAM_masks'])
    }
}

test_pictures = {
    'APEER_masks': os.path.join(*['test_data', 'APEER_masks']),
    'SAM_masks': os.path.join(*['test_data', 'SAM_masks']),
    'pictures': os.path.join(*['test_data', 'pictures'])
}

def intersectionOverUnion(ground_truth_mask, computed_mask):
    """
    @:ground_truth_mask: numpy matrix containing only zeros and ones
    @:computed_mask: numpy matrix containing only zeros and ones
    :return: a real number representing the Intersection over Union of the computed mask relative to ground_truth_mask
    """
    assert ground_truth_mask is not None
    assert computed_mask.shape is not None
    assert ground_truth_mask.shape == computed_mask.shape
    intersection = np.logical_and(ground_truth_mask, computed_mask)
    union = np.logical_or(ground_truth_mask, computed_mask)
    result = np.sum(intersection) / np.sum(union)
    return result

def diceCoefficient(ground_truth_mask, computed_mask):
    """
    @:ground_truth_mask: numpy matrix containing only zeros and ones
    @:computed_mask: numpy matrix containing only zeros and ones
    :return: a real number representing the Dice Coefficient of the computed mask relative to ground_truth_mask
    """
    assert ground_truth_mask is not None
    assert computed_mask.shape is not None
    assert ground_truth_mask.shape == computed_mask.shape
    TP = np.sum(computed_mask[ground_truth_mask == 1])
    pred_pos = np.sum(computed_mask)
    ground_pos = np.sum(ground_truth_mask)
    result = (2.0*TP) / (pred_pos + ground_pos)
    return result

def pixelAccuracy(ground_truth_mask, computed_mask):
    """
    @:ground_truth_mask: numpy matrix containing only zeros and ones
    @:computed_mask: numpy matrix containing only zeros and ones
    :return: a real number representing the pixel accuracy of the computed mask relative to ground_truth_mask
    (TP + TN) / (TP + TN + FP + FN)
    """
    assert ground_truth_mask is not None
    assert computed_mask.shape is not None
    assert ground_truth_mask.shape == computed_mask.shape
    dual_computed_mask = 1-computed_mask
    TP = np.sum(computed_mask[ground_truth_mask == 1])
    TN = np.sum(dual_computed_mask[ground_truth_mask == 0])
    all = computed_mask.shape[0]*computed_mask.shape[1]
    result = (TP + TN) / all
    return result

def precision(ground_truth_mask, computed_mask):
    """
    @:ground_truth_mask: numpy matrix containing only zeros and ones
    @:computed_mask: numpy matrix containing only zeros and ones
    :return: a real number representing the precision of the computed mask relative to ground_truth_mask
    (TP) / (TP + FP)
    """
    assert ground_truth_mask is not None
    assert computed_mask.shape is not None
    assert ground_truth_mask.shape == computed_mask.shape
    TP = np.sum(computed_mask[ground_truth_mask == 1])
    result = TP / (computed_mask.shape[0]*computed_mask.shape[1])
    return result

def recall(ground_truth_mask, computed_mask):
    """
    @:ground_truth_mask: numpy matrix containing only zeros and ones
    @:computed_mask: numpy matrix containing only zeros and ones
    :return: a real number representing the F1 score for computed_mask relative to ground_truth_mask
    TP / (TP + FN)
    """
    assert ground_truth_mask is not None
    assert computed_mask.shape is not None
    assert ground_truth_mask.shape == computed_mask.shape
    TP = np.sum(computed_mask[ground_truth_mask == 1])
    dual_computed_mask = 1 - computed_mask
    FN = np.sum(dual_computed_mask[ground_truth_mask == 1])
    result = TP / (TP + FN)
    return result

def f1Score(ground_truth_mask, computed_mask):
    """
    @:ground_truth_mask: numpy matrix containing only zeros and ones
    @:computed_mask: numpy matrix containing only zeros and ones
    :return: a real number representing the F1 score for computed_mask relative to ground_truth_mask
    """
    assert ground_truth_mask is not None
    assert computed_mask.shape is not None
    assert ground_truth_mask.shape == computed_mask.shape
    p = precision(ground_truth_mask, computed_mask)
    r = recall(ground_truth_mask, computed_mask)
    result = (2*p*r) / (p + r)
    return result

def normalizedSurfaceDistance(ground_truth_mask, computed_mask):
    """
    @:ground_truth_mask: numpy matrix containing only zeros and ones
    @:computed_mask: numpy matrix containing only zeros and ones
    :return: a real number representing the Normalized Surface Distance between the two masks
    """
    assert ground_truth_mask is not None
    assert computed_mask.shape is not None
    assert ground_truth_mask.shape == computed_mask.shape
    result = 0 # TODO
    return result

def symmetricContourDistance(ground_truth_mask, computed_mask):
    """
    @:ground_truth_mask: numpy matrix containing only zeros and ones
    @:computed_mask: numpy matrix containing only zeros and ones
    :return: a real number representing the Symmetric Contour Distance between the two masks
    """
    assert ground_truth_mask is not None
    assert computed_mask.shape is not None
    assert ground_truth_mask.shape == computed_mask.shape
    result = 0 # TODO
    return result

def hausdorffDistance(ground_truth_mask, computed_mask):
    """
    @:ground_truth_mask: numpy matrix containing only zeros and ones
    @:computed_mask: numpy matrix containing only zeros and ones
    :return: a real number representing the Hausdorff Distance between the two masks
    """
    assert ground_truth_mask is not None
    assert computed_mask.shape is not None
    assert ground_truth_mask.shape == computed_mask.shape
    result = 0 # TODO
    return result

def compute_metrics_and_save(image_folder, predicted_masks_folder, ground_truth_masks_folder, result_file):
    """
    @:image_folder: full path to a folder that contains all images of which the masks are either predicted or annotated.
                    It is assumed that the image files are direct children of the image_folder
                    It is assumed that the masks are binary:
                     - they are 2D matrices (contains )
    @:predicted_masks_folder: full path to a folder that contains all predicted masks. It is assumed that the mask files
                            are direct children of the predicted_masks_folder
    @:ground_truth_masks_folder: full path to a folder that contains all ground truth masks. It is assumed that the mask files
                            are direct children of the ground_truth_masks_folder.
                            It is also assumed that there is a 1-to-1 correspondence between the masks
                            directly contained in the two folders predicted_masks_folder and predicted_masks_folder.
    @:result_file: the full name of a csv file in which all results of the performance metrics will be saved
    :return: does not return anything. Creates a file on the disk.
    """
    assert predicted_masks_folder is not None
    assert os.path.exists(predicted_masks_folder)
    assert ground_truth_masks_folder is not None
    assert os.path.exists(ground_truth_masks_folder)
    assert result_file is not None

    image_list = [os.path.join(*[image_folder, x]) for x in os.listdir(image_folder) if
                    (x.endswith('.png') or x.endswith('.jpg') or x.endswith('.JPG') or x.endswith('.jpeg'))
                    and not x.startswith('.')
                    and os.path.isfile(os.path.join(*[image_folder, x]))
                    and not os.path.isdir(os.path.join(*[image_folder, x]))
                ]
    image_list.sort()

    predicted_mask_list = [os.path.join(*[predicted_masks_folder, x]) for x in os.listdir(predicted_masks_folder) if
                           x.endswith('.tiff')
                           and not x.startswith('.')
                           and os.path.isfile(os.path.join(*[predicted_masks_folder, x]))
                           and not os.path.isdir(os.path.join(*[predicted_masks_folder, x]))
                        ]
    predicted_mask_list.sort()

    ground_truth_mask_list = [os.path.join(*[ground_truth_masks_folder, x]) for x in os.listdir(ground_truth_masks_folder) if
                           x.endswith('.tiff')
                           and not x.startswith('.')
                           and os.path.isfile(os.path.join(*[ground_truth_masks_folder, x]))
                           and not os.path.isdir(os.path.join(*[ground_truth_masks_folder, x]))
                           ]
    ground_truth_mask_list.sort()

    records = []
    for image, predicted, ground_truth in list(zip(image_list, predicted_mask_list, ground_truth_mask_list)):
        if contraction(path2name(image)) == contraction(path2name(predicted)) == contraction(path2name(ground_truth)):
            rec = {
                'image': image,
                'predicted_mask': predicted,
                'ground_truth_mask': ground_truth,
                'intersectionOverUnion': intersectionOverUnion(ground_truth, predicted),
                'diceCoefficient': diceCoefficient(ground_truth, predicted),
                'pixelAccuracy': pixelAccuracy(ground_truth, predicted),
                'precision': precision(ground_truth, predicted),
                'recall': recall(ground_truth, predicted),
                'f1Score': f1Score(ground_truth, predicted),
                'normalizedSurfaceDistance': normalizedSurfaceDistance(ground_truth, predicted),
                'symmetricContourDistance': symmetricContourDistance(ground_truth, predicted),
                'hausdorffDistance': hausdorffDistance(ground_truth, predicted)
            }
            records.append(rec)
    result = pd.DataFrame(records, columns=[
        'image',
        'predicted_mask',
        'ground_truth_mask',
        'intersectionOverUnion',
        'diceCoefficient'
        'pixelAccuracy'
        'precision'
        'recall',
        'f1Score',
        'normalizedSurfaceDistance',
        'symmetricContourDistance',
        'hausdorffDistance'])
    result.to_csv(result_file, index=False, header=True)
    pass

def main():
    """
    merge_SAM_masks(CWFID_dataset['SAM_masks'])
    for filename in ground_truth_msk_list:
    msk = plt.imread(filename).astype(np.uint8)
    msk = 1 - msk# the mask files represent with 1 the background a 0 the Region Of Interest, so we invert them
    dest_file = os.path.join(*[CWFID_dataset['masks'], f'{contraction(path2name(filename))}.tiff'])
    if not os.path.exists(dest_file):  # save the file only if it does not yet exist
        Image.fromarray(msk.astype(np.uint8)).save(dest_file)
    print(f"All masks inverted.")
    """
    compute_metrics_and_save(
        image_folder=CWFID_dataset['images'],
        predicted_masks_folder=CWFID_dataset['SAM_masks'],
        ground_truth_masks_folder=CWFID_dataset['SAM_masks'],
        result_file='SAM_vs_ground_truth_on_CWFID.csv'
    )

if __name__ == "__main__":
    main()