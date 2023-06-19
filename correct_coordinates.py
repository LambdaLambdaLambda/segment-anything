import os
from utils import path2name, contraction
import json

CWFID_dataset = {
    'annotations': os.path.join(*['CWFID_dataset', 'annotations']),
    'SAM_annotations': os.path.join(*['CWFID_dataset', 'SAM_annotations']),
    'images': os.path.join(*['CWFID_dataset', 'images']),
    'masks': os.path.join(*['CWFID_dataset', 'masks']),
    'SamPredictor_masks': os.path.join(*['CWFID_dataset', 'SamPredictor_masks']),
    'SamAutomaticMaskGenerator_masks': os.path.join(*['CWFID_dataset', 'SamAutomaticMaskGenerator_masks'])
}

def main():
    filename = os.path.join(*[CWFID_dataset['SAM_annotations'], '055_image.json'])
    with open(filename) as source:
        data_loaded = json.load(source)
    index_1 = data_loaded['input_points'].index([153, 48])
    index_2 = data_loaded['input_points'].index([239, 16])
    print(f"data_loaded['input_points'][index_1] = {data_loaded['input_points'][index_1]}")
    print(f"data_loaded['input_labels'][index_1] = {data_loaded['input_labels'][index_1]}")
    #data_loaded['input_labels'][index_1] = 0
    #print(f"data_loaded['input_labels'][index_1] = {data_loaded['input_labels'][index_1]}")

    print(f"data_loaded['input_points'][index_2] = {data_loaded['input_points'][index_2]}")
    print(f"data_loaded['input_labels'][index_2] = {data_loaded['input_labels'][index_2]}")
    #data_loaded['input_labels'][index_2] = 0
    #print(f"data_loaded['input_labels'][index_2] = {data_loaded['input_labels'][index_2]}")


    with open(filename, "w") as dest:
        json.dump({
            'add_points': data_loaded['add_points'],
            'rem_points': data_loaded['rem_points'],
            'input_points': data_loaded['input_points'],
            'input_labels': data_loaded['input_labels'],
            'img_file': data_loaded['img_file'],
            'time': data_loaded['time']
        }, dest)

    print(f"Correction terminated.")

if __name__ == "__main__":
    wd = os.path.realpath(os.path.dirname(__file__))
    os.chdir(wd)
    main()