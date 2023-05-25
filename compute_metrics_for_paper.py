import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import random
from PIL import Image
from utils import contraction

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

test_pictures = {
    'APEER_masks': os.path.join(*['test_data', 'APEER_masks']),
    'SAM_masks': os.path.join(*['test_data', 'SAM_masks']),
    'pictures': os.path.join(*['test_data', 'pictures'])
}

def intersectionOverUnion(ground_truth_mask, computed_mask):
    result = None
    return result

def diceCoefficient(ground_truth_mask, computed_mask):
    result = None
    return result

def pixelAccuracy(ground_truth_mask, computed_mask):
    result = None
    return result

def precision(ground_truth_mask, computed_mask):
    result = None
    return result

def recall(ground_truth_mask, computed_mask):
    result = None
    return result

def f1Score(ground_truth_mask, computed_mask):
    result = None
    return result

def normalizedSurfaceDistance(ground_truth_mask, computed_mask):
    result = None
    return result

def symmetricContourDistance(ground_truth_mask, computed_mask):
    result = None
    return result

def hausdorffDistance(ground_truth_mask, computed_mask):
    result = None
    return result

def main():
    file_list = os.listdir(test_pictures['pictures']['folder'])
    file_list = [x for x in file_list if x.endswith('.jpg') or x.endswith('.JPG')]
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