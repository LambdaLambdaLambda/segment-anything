import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import random
from PIL import Image
from utils import contraction

CWFID_dataset = {
    'annotations': os.path.join(*['CWFID_dataset', 'annotations']),
    'images': os.path.join(*['CWFID_dataset', 'images']),
    'masks': os.path.join(*['CWFID_dataset', 'masks']),
    'SAM_masks': os.path.join(*['CWFID_dataset', 'SAM_masks'])
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

def compute_metrics_and_save(predicted_masks_folder, ground_truth_masks_folder, result_file):
    """
    @:predicted_masks_folder: full path to a folder that contains all predicted masks. It is assumed that the mask files
                            are direct children of the predicted_masks_folder
    @:ground_truth_masks_folder: full path to a folder that contains all ground truth masks. It is assumed that the mask files
                            are direct children of the ground_truth_masks_folder.
                            It is also assumed that there is a 1-to-1 correspondence between the masks
                            directly contained in the two folders predicted_masks_folder and predicted_masks_folder.
    @:result_file: the full name of a csv file in which all results of the performance metrics will be saved
    :return: does not return anything. Creates a file on the disk.
    """
    predicted_mask_list = os.listdir(predicted_masks_folder)
    predicted_mask_list = [os.path.join(*[predicted_masks_folder, x]) for x in predicted_mask_list if
                           (x.endswith('.png') or x.endswith('.tiff'))
                           and not x.startswith('.')
                           and os.path.isfile(os.path.join(*[predicted_masks_folder, x]))
                           and not os.path.isdir(os.path.join(*[predicted_masks_folder, x]))
                        ]
    predicted_mask_list.sort()
    ground_truth_mask_list = os.listdir(ground_truth_masks_folder)
    ground_truth_mask_list = [os.path.join(*[ground_truth_masks_folder, x]) for x in ground_truth_mask_list if
                           (x.endswith('.png') or x.endswith('.tiff'))
                           and not x.startswith('.')
                           and os.path.isfile(os.path.join(*[ground_truth_masks_folder, x]))
                           and not os.path.isdir(os.path.join(*[ground_truth_masks_folder, x]))
                           ]
    ground_truth_mask_list.sort()
    for file in msk_list:
        msk = plt.imread(file)
        print(f"Values in file {file}:  {np.unique(msk)}")

    file_list = os.listdir(test_pictures['pictures']['folder'])
    file_list = [x for x in file_list if (x.endswith('.jpg') or x.endswith('.JPG')) and (not x.startswith('.'))]
    file_list.sort()
    pass

def main():
    msk_list = os.listdir(CWFID_dataset['masks'])
    msk_list = [os.path.join(*[CWFID_dataset['masks'], x]) for x in msk_list if x.endswith('.png') and not x.startswith('.')]
    for file in msk_list:
        msk = plt.imread(file)
        print(f"Values in file {file}:  {np.unique(msk)}")

    file_list = os.listdir(test_pictures['pictures']['folder'])
    file_list = [x for x in file_list if (x.endswith('.jpg') or x.endswith('.JPG')) and (not x.startswith('.'))]
    file_list.sort()
    APEER_records = []
    SAM_records = []
    for file in file_list:
        ground_truth_mask = plt.imread(os.path.join(*[test_pictures['pictures'], file]))
        assert os.path.exists(os.path.join(*[test_pictures['APEER_masks'], f"{contraction(file)}.tiff"]))
        APEER_mask = plt.imread(os.path.join(*[test_pictures['APEER_masks'], f"{contraction(file)}.tiff"]))
        rec = {
            'file': os.path.join(*[test_pictures['pictures']['folder'], file]),
            'intersectionOverUnion': intersectionOverUnion(ground_truth_mask, APEER_mask),
            'diceCoefficient': diceCoefficient(ground_truth_mask, APEER_mask),
            'pixelAccuracy': pixelAccuracy(ground_truth_mask, APEER_mask),
            'precision': precision(ground_truth_mask, APEER_mask),
            'recall': recall(ground_truth_mask, APEER_mask),
            'f1Score': f1Score(ground_truth_mask, APEER_mask),
            'normalizedSurfaceDistance': normalizedSurfaceDistance(ground_truth_mask, APEER_mask),
            'symmetricContourDistance': symmetricContourDistance(ground_truth_mask, APEER_mask),
            'hausdorffDistance': hausdorffDistance(ground_truth_mask, APEER_mask)
        }
        APEER_records.append(rec)
        assert os.path.exists(os.path.join(*[test_pictures['SAM_masks'], f"{contraction(file)}.tiff"]))
        SAM_mask = plt.imread(os.path.join(*[test_pictures['SAM_masks'], f"{contraction(file)}.tiff"]))
        rec = {
            'file': os.path.join(*[test_pictures['pictures']['folder'], file]),
            'intersectionOverUnion': intersectionOverUnion(ground_truth_mask, SAM_mask),
            'diceCoefficient': diceCoefficient(ground_truth_mask, SAM_mask),
            'pixelAccuracy': pixelAccuracy(ground_truth_mask, SAM_mask),
            'precision': precision(ground_truth_mask, SAM_mask),
            'recall': recall(ground_truth_mask, SAM_mask),
            'f1Score': f1Score(ground_truth_mask, SAM_mask),
            'normalizedSurfaceDistance': normalizedSurfaceDistance(ground_truth_mask, SAM_mask),
            'symmetricContourDistance': symmetricContourDistance(ground_truth_mask, SAM_mask),
            'hausdorffDistance': hausdorffDistance(ground_truth_mask, SAM_mask)
        }
        SAM_records.append(rec)

    APEER_df = pd.DataFrame(APEER_records, columns=[
                'file',
                'intersectionOverUnion',
                'diceCoefficient'
                'pixelAccuracy'
                'precision'
                'recall',
                'f1Score',
                'normalizedSurfaceDistance',
                'symmetricContourDistance',
                'hausdorffDistance'])
    SAM_df = pd.DataFrame(SAM_records, columns=[
        'file',
        'intersectionOverUnion',
        'diceCoefficient'
        'pixelAccuracy'
        'precision'
        'recall',
        'f1Score',
        'normalizedSurfaceDistance',
        'symmetricContourDistance',
        'hausdorffDistance'])
    APEER_df.to_csv('APEER_metrics.csv', index=False, header=True)
    SAM_df.to_csv('SAM_metrics.csv', index=False, header=True)

if __name__ == "__main__":
    main()