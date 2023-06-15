# importing the module
import cv2
import os
import json
from utils import path2name, contraction

global dictionary
global img
global CWFID_dataset
global window_name

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        dictionary['add_points'].append([[int(x), int(y)]])
        color = (255, 0, 0)
        print(f"Points inside the mask: {dictionary['add_points']}")
    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        dictionary['rem_points'].append([[int(x), int(y)]])
        color = (255, 255, 0)
        print(f"Points outside the mask: {dictionary['rem_points']}")
    # displaying the coordinates on the image window
    font = cv2.FONT_HERSHEY_SIMPLEX
    msg = f"{str(int(x))} , {str(int(y))}"
    cv2.putText(img, msg, (x, y), font, 1, color, 2)
    cv2.imshow(window_name, img)

# driver function
if __name__ == "__main__":
    CWFID_dataset = {
        'annotations': os.path.join(*['CWFID_dataset', 'annotations']),
        'SAM_annotations': os.path.join(*['CWFID_dataset', 'SAM_annotations']),
        'images': os.path.join(*['CWFID_dataset', 'images']),
        'masks': os.path.join(*['CWFID_dataset', 'masks']),
        'SAM_masks': os.path.join(*['CWFID_dataset', 'SAM_masks'])
    }
    img_file_list = os.listdir(CWFID_dataset['images'])
    img_file_list = [os.path.join(*[CWFID_dataset['images'], f]) for f in img_file_list if
                      f.endswith('.png')
                      and not f.startswith('.')
                      and os.path.isfile(os.path.join(*[CWFID_dataset['images'], f]))
                      ]
    img_file_list.sort()

    for filename in img_file_list:
        dictionary = {
            'add_points': [],
            'rem_ponts': []
        }
        window_name = filename
        img = None
        # reading the image
        img = cv2.imread(filename, 1)
        cv2.namedWindow(window_name)
        cv2.setMouseCallback('image', click_event)
        # displaying the image
        cv2.imshow('image', img)
        # setting mouse handler for the image
        # and calling the click_event() function
        # wait for a key to be pressed to exit
        k = cv2.waitKey(0)
        if k == 0 or k == 27:
            break
        # close the window
        cv2.destroyAllWindows()
        jsonfile = os.path.join(*[CWFID_dataset['SAM_annotations'], f"{contraction(path2name(filename))}.json"])
        with open(jsonfile, "w") as outfile:
            json.dump(dictionary, outfile)
        print(f"Iteration on file {filename} completed. File {jsonfile} saved.")
    print("list terminated")
    pass