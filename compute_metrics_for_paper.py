import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import re
from skimage import metrics
import random
from PIL import Image
from utils import contraction, path2name, merge_SAM_masks
import seg_metrics.seg_metrics as sg

CWFID_dataset = {
    'annotations': os.path.join(*['CWFID_dataset', 'annotations']),
    'SAM_annotations': os.path.join(*['CWFID_dataset', 'SAM_annotations']),
    'images': os.path.join(*['CWFID_dataset', 'images']),
    'masks': os.path.join(*['CWFID_dataset', 'masks']),
    'SamPredictor_masks': os.path.join(*['CWFID_dataset', 'SamPredictor_masks']),
    'SamAutomaticMaskGenerator_masks': os.path.join(*['CWFID_dataset', 'SamAutomaticMaskGenerator_masks'])
}

ESCA_dataset = {
    'esca': {
        'folder': os.path.join(*['ESCA_dataset', 'esca']),
        'esca_foliage_over_healthy_bg': os.path.join(*['ESCA_dataset', 'esca', 'esca_foliage_over_healthy_bg']),
        'masks': os.path.join(*['ESCA_dataset', 'esca', 'masks']),
        'pictures': os.path.join(*['ESCA_dataset', 'esca', 'pictures']),
        'SamAutomaticMaskGenerator_masks': os.path.join(*['ESCA_dataset', 'esca', 'SamAutomaticMaskGenerator_masks'])
    },
    'healthy': {
        'folder': os.path.join(*['ESCA_dataset', 'healthy']),
        'healthy_foliage_over_esca_bg': os.path.join(*['ESCA_dataset', 'esca', 'healthy_foliage_over_esca_bg']),
        'masks': os.path.join(*['ESCA_dataset', 'healthy', 'masks']),
        'pictures': os.path.join(*['ESCA_dataset', 'healthy', 'pictures']),
        'SamAutomaticMaskGenerator_masks': os.path.join(*['ESCA_dataset', 'healthy', 'SamAutomaticMaskGenerator_masks'])
    }
}

test_pictures = {
    'APEER_masks': os.path.join(*['test_data', 'APEER_masks']),
    'SamAutomaticMaskGenerator_masks': os.path.join(*['test_data', 'SamAutomaticMaskGenerator_masks']),
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
    result = float(np.sum(intersection)) / float(np.sum(union))
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
    result = (2.0*float(TP)) / float((pred_pos + ground_pos))
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
    result = float((TP + TN)) / float(all)
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
    result = float(TP) / float((computed_mask.shape[0]*computed_mask.shape[1]))
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
    result = float(TP) / float(TP + FN)
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
    result = float(2*p*r) / float(p + r)
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
    result = None # TODO
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
    result = None # TODO
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
    result = None # TODO
    return result

def compute_metrics_and_save(image_folder, ground_truth_masks_folder, predicted_masks_folder, result_file):
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
    for image, ground_truth, prediction in list(zip(image_list, ground_truth_mask_list, predicted_mask_list)):
        ok = path2name(prediction) == path2name(ground_truth)
        ok = ok and path2name(prediction)[:3] == path2name(image)[:3]
        print(f"path2name(prediction) = {path2name(prediction)}")
        print(f"path2name(ground_truth) = {path2name(ground_truth)}")
        print(f"path2name(image)[:3] = {path2name(image)[:3]}")
        if ok:
            gt = plt.imread(ground_truth)
            pr = plt.imread(prediction)
            labels = [1]
            metrics = sg.write_metrics(labels=labels,  # exclude background
                                       gdth_img=gt,
                                       pred_img=pr,
                                       csv_file=None,#os.path.join(*['CWFID_dataset', 'computed_metrics', f"{path2name(image)[:3]}_comparison.csv"])
                                       metrics=['hd', 'msd'])
            # hd  --> Hausdorff distance
            # msd --> Average symmetric surface distance
            rec = {
                'image_file': image,
                'predicted_mask_file': prediction,
                'ground_truth_mask_file': ground_truth,
                'intersectionOverUnion': intersectionOverUnion(gt, pr),
                'diceCoefficient': diceCoefficient(gt, pr),
                'pixelAccuracy': pixelAccuracy(gt, pr),
                'precision': precision(gt, pr),
                'recall': recall(gt, pr),
                'f1Score': f1Score(gt, pr),
                'avgSymmetricSurfaceDistance': (metrics[0])['msd'][0],
                'hausdorffDistance': (metrics[0])['hd'][0]
            }
            records.append(rec)
            print(f"File {image} processed with its masks.")

    result = pd.DataFrame(records, columns=[
        'image_file',
        'predicted_mask_file',
        'ground_truth_mask_file',
        'intersectionOverUnion',
        'diceCoefficient',
        'pixelAccuracy',
        'precision',
        'recall',
        'f1Score',
        'avgSymmetricSurfaceDistance',
        'hausdorffDistance'])
    result.to_csv(result_file, index=False, header=True)
    print(f"File {result_file} saved.")
    pass

def save_predictor_running_times(source_txt, dest_csv):
    source = open(source_txt, 'r')
    allines = source.readlines()
    records = []
    for i, line in enumerate(allines):
        txt = line.strip()  # Strips the newline character
        arr = txt.split()
        if i%7 == 0:
            rec = {
                'filename': arr[2],
                'embedding_time': None,
                'prediction_time': None,
                'saving_time': None
            }
        elif i%7 == 1:
            pass
        elif i%7 == 2:
            pass
        elif i%7 == 3:
            rec['embedding_time'] = float(arr[2])
        elif i%7 == 4:
            rec['prediction_time'] = float(arr[2])
        elif i%7 == 5:
            rec['saving_time'] = float(arr[2])
        elif i%7 == 6:
            records.append(rec)

    result = pd.DataFrame(records, columns=[
        'filename',
        'embedding_time',
        'prediction_time',
        'saving_time'])
    result.to_csv(dest_csv, index=False, header=True)
    print(f"File {dest_csv} saved.")
    pass

def save_automatic_running_times(source_txt, dest_csv):
    source = open(source_txt, 'r')
    allines = source.readlines()
    records = []
    for i, line in enumerate(allines):
        txt = line.strip()  # Strips the newline character
        arr = txt.split()
        if i % 3 == 0:
            rec = {
                'filename': None,
                'prediction_time': None,
                'saving_time': None
            }
        elif i % 3 == 1:
            rec['prediction_time'] = float(arr[2])
            rec['filename'] = arr[4]
        elif i % 3 == 2:
            rec['saving_time'] = float(arr[2])
            records.append(rec)

    result = pd.DataFrame(records, columns=[
        'filename',
        'prediction_time',
        'saving_time'])
    result.to_csv(dest_csv, index=False, header=True)
    print(f"File {dest_csv} saved.")
    pass

def main():
    """
    for filename in ground_truth_msk_list:
    msk = plt.imread(filename).astype(np.uint8)
    msk = 1 - msk# the mask files represent with 1 the background a 0 the Region Of Interest, so we invert them
    dest_file = os.path.join(*[CWFID_dataset['masks'], f'{contraction(path2name(filename))}.tiff'])
    if not os.path.exists(dest_file):  # save the file only if it does not yet exist
        Image.fromarray(msk.astype(np.uint8)).save(dest_file)
    print(f"All masks inverted.")#DONE
        image_list = [os.path.join(*[CWFID_dataset['SamPredictor_masks'], x]) for x in os.listdir(CWFID_dataset['SamPredictor_masks']) if
                  x.endswith('.tiff')
                  and not x.startswith('.')
                  and os.path.isfile(os.path.join(*[CWFID_dataset['SamPredictor_masks'], x]))
                  and not os.path.isdir(os.path.join(*[CWFID_dataset['SamPredictor_masks'], x]))
                  ]
    image_list.sort()
    for filename in image_list:
        old_name = filename
        new_name = filename.replace('_image.', '_mask.')
        if os.path.isfile(new_name):
            print("The file already exists")
        else:
            os.rename(old_name, new_name)
    print(f"Renaming operation terminated")#DONE

    compute_metrics_and_save(#DONE
        image_folder=CWFID_dataset['images'],
        predicted_masks_folder=CWFID_dataset['SamAutomaticMaskGenerator_masks'],
        ground_truth_masks_folder=CWFID_dataset['masks'],
        result_file=os.path.join(*['CWFID_dataset', 'computed_metrics', 'SamAutomaticMaskGenerator_vs_ground_truth_on_CWFID.csv'])
    )
    compute_metrics_and_save(#DONE
        image_folder=CWFID_dataset['images'],
        predicted_masks_folder=CWFID_dataset['SamPredictor_masks'],
        ground_truth_masks_folder=CWFID_dataset['masks'],
        result_file=os.path.join(
            *['CWFID_dataset', 'computed_metrics', 'SamPredictor_vs_ground_truth_on_CWFID.csv'])
    )
    save_predictor_running_times(#DONE
        source_txt=os.path.join(*['CWFID_dataset', 'computed_metrics', 'SamPredictor_running_log.txt']),
        dest_csv=os.path.join(*['CWFID_dataset', 'computed_metrics', 'SamPredictor_running_times.csv'])
    )
    """
    save_automatic_running_times(  # DONE
        source_txt=os.path.join(*['CWFID_dataset', 'computed_metrics', 'SamAutomatic_running_log.txt']),
        dest_csv=os.path.join(*['CWFID_dataset', 'computed_metrics', 'SamAutomatic_running_times.csv'])
    )

"""
The SamAutomaticMaskGenerator took a total of 4290.769307374954 seconds to run on all 60 images 
"""

if __name__ == "__main__":
    main()