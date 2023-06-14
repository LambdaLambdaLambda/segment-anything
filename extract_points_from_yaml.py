import yaml
import io
import os
import numpy as np
from enum import Enum

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


yaml_folder = CWFID_dataset['annotations']
yaml_file_list = os.listdir(yaml_folder)
yaml_file_list = [os.path.join(*[yaml_folder, f]) for f in yaml_file_list if f.endswith('.yaml') and not f.startswith('._')]
yaml_file_list.sort() # contains ordered list of full paths only of yaml files inside yaml_folder

for filename in yaml_file_list:
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
        print(f"{data_loaded['filename']}")
        for rec in data_loaded['annotation']:
            print(f"{rec['type']}")
            try: # some annotations do not contain coordinates represented as lists
                iterator = iter(rec['points']['x'])
                iterator = iter(rec['points']['y'])
            except TypeError:
                coordinates = [(rec['points']['x'], rec['points']['y'])]
            else:
                coordinates = list(zip(rec['points']['x'], rec['points']['y']))
            n = len(coordinates)
            labels = np.empty(n) #prepare the array of labels as requesed by SAM in predictor mode
            if rec['type'] == 'weed':
                labels.fill(Labels.WEED.value)
            elif rec['type'] == 'crop':
                labels.fill(Labels.CROP.value)
        pass