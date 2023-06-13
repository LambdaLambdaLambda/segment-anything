import yaml
import io
import os
import numpy as np

yaml_folder = os.path.join(*['annotations'])
yaml_file_list = os.listdir(yaml_folder)
yaml_file_list = [os.path.join(*[yaml_folder, f]) for f in yaml_file_list if f.endswith('.yaml') and not f.startswith('._')]
yaml_file_list.sort() # contains ordered list of full paths only of yaml files inside yaml_folder

for filename in yaml_file_list:
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
        for rec in data_loaded['annotation']:
            coordinates = list(zip(rec['points']['x'], rec['points']['y']))
            n = len(coordinates)
            labels = np.empty(n)
            labels.fill(1)
        pass
