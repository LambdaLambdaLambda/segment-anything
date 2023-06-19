import yaml
import os
from enum import Enum
from utils import path2name, contraction
import json
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

class Labels(Enum):
    BACKGROUND = 0
    WEED = 1
    CROP = 2

CWFID_dataset = {
    'annotations': os.path.join(*['CWFID_dataset', 'annotations']),
    'images': os.path.join(*['CWFID_dataset', 'images']),
    'masks': os.path.join(*['CWFID_dataset', 'masks']),
    'SamPredictor_masks': os.path.join(*['CWFID_dataset', 'SamPredictor_masks']),
    'SamAutomaticMaskGenerator_masks': os.path.join(*['CWFID_dataset', 'SamAutomaticMaskGenerator_masks']),
    'SamPredictor_annotations': os.path.join(*['CWFID_dataset', 'SamPredictor_annotations']),
    'computed_metrics': os.path.join(*['CWFID_dataset', 'computed_metrics'])
}

def read_from_yaml(filename):
    input_points = []
    input_labels = []
    data_loaded = None
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for rec in data_loaded['annotation']:
        print(f"{rec['type']}")
        temp = None
        try:  # some annotations do not contain coordinates represented as lists
            iterator = iter(rec['points']['x'])
            iterator = iter(rec['points']['y'])
        except TypeError:
            x = rec['points']['x']
            y = rec['points']['y']
            temp = [[int(x), int(y)]]
        else:
            temp = [[int(x), int(y)] for (x, y) in list(zip(rec['points']['x'], rec['points']['y']))]
        input_points.extend(temp)
        # prepare the array of labels as requested by SAM in predictor mode
        # input_labels[j] == 1 ---> the point input_points[j] belongs to the mask
        # input_labels[j] == 0 ---> the point input_points[j] does not belong to the mask
        input_labels.extend([1] * len(temp))
        time = None
    return input_points, input_labels, time

def read_from_json(filename):
    input_points = []
    input_labels = []
    data_loaded = None
    with open(filename) as source:
        data_loaded = json.load(source)
    # prepare the array of labels as requested by SAM in predictor mode
    # input_labels[j] == 1 ---> the point input_points[j] belongs to the mask
    # input_labels[j] == 0 ---> the point input_points[j] does not belong to the mask
    temp = data_loaded['add_points']
    input_points.extend(temp)
    input_labels.extend([1] * len(temp))
    temp = data_loaded['rem_points']
    input_points.extend(temp)
    input_labels.extend([0] * len(temp))
    time = None
    if 'input_points' in data_loaded:
        input_points.extend(data_loaded['input_points'])
    if 'input_labels' in data_loaded:
        input_labels.extend(data_loaded['input_labels'])
    if 'time' in data_loaded:
        time = data_loaded['time']
    return input_points, input_labels, time

def main():
    yaml_file_list = os.listdir(CWFID_dataset['annotations'])
    yaml_file_list = [os.path.join(*[CWFID_dataset['annotations'], f]) for f in yaml_file_list if
                      f.endswith('.yaml')
                      and not f.startswith('.')
                      and os.path.isfile(os.path.join(*[CWFID_dataset['annotations'], f]))
                      ]
    yaml_file_list.sort() # contains ordered list of full paths only of yaml files inside yaml_folder

    json_file_list = os.listdir(CWFID_dataset['SamPredictor_annotations'])
    json_file_list = [os.path.join(*[CWFID_dataset['SamPredictor_annotations'], f]) for f in json_file_list if
                      f.endswith('.json')
                      and not f.startswith('.')
                      and os.path.isfile(os.path.join(*[CWFID_dataset['SamPredictor_annotations'], f]))
                      ]
    json_file_list.sort()  # contains ordered list of full paths only of yaml files inside yaml_folder

    records = []
    for filename in yaml_file_list:
        input_points, input_labels, time = read_from_yaml(filename)
        input_points = np.array(input_points)
        input_labels = np.array(input_labels)
        rec = {
            'filename': filename,
            'num_add_points': len(input_points[input_labels==1]),
            'num_rem_points': len(input_points[input_labels==0]),
            'time': time
        }
        records.append(rec)

    result = pd.DataFrame(records, columns=[
        'filename',
        'num_add_points',
        'num_rem_points',
        'time'
    ])
    dest_file = os.path.join(*[CWFID_dataset['computed_metrics'], 'Statistics_manual_masks.csv'])
    result.to_csv(dest_file, index=False, header=True)
    print(f"File {dest_file} saved.")

    records = []
    for filename in json_file_list:
        input_points, input_labels, time = read_from_json(filename)
        input_points = np.array(input_points)
        input_labels = np.array(input_labels)
        rec = {
            'filename': filename,
            'num_add_points': len(input_points[input_labels==1]),
            'num_rem_points': len(input_points[input_labels==0]),
            'time': time
        }
        records.append(rec)

    result = pd.DataFrame(records, columns=[
        'filename',
        'num_add_points',
        'num_rem_points',
        'time'])
    dest_file = os.path.join(*[CWFID_dataset['computed_metrics'], 'Statistics_predictor_masks.csv'])
    result.to_csv(dest_file, index=False, header=True)
    print(f"File {dest_file} saved.")

if __name__ == "__main__":
    wd = os.path.realpath(os.path.dirname(__file__))
    os.chdir(wd)
    main()