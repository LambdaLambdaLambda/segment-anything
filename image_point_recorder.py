# importing the module
import cv2
import os
import json
from utils import path2name, contraction

CWFID_dataset = {
        'annotations': os.path.join(*['CWFID_dataset', 'annotations']),
        'SAM_annotations': os.path.join(*['CWFID_dataset', 'SAM_annotations']),
        'images': os.path.join(*['CWFID_dataset', 'images']),
        'masks': os.path.join(*['CWFID_dataset', 'masks']),
        'SAM_masks': os.path.join(*['CWFID_dataset', 'SAM_masks'])
    }
dictionary = {
            'add_points': [],
            'rem_ponts': []
}

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        dictionary['add_points'].extend([[int(x),int(y)]])
        # displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' + str(y), (x, y), font, 1, (255, 0, 0), 2)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        dictionary['rem_points'].extend([[int(x), int(y)]])
        # displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' + str(y), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)

# driver function
if __name__ == "__main__":
    img_file_list = os.listdir(CWFID_dataset['images'])
    img_file_list = [os.path.join(*[CWFID_dataset['images'], f]) for f in img_file_list if
                      f.endswith('.png')
                      and not f.startswith('.')
                      and os.path.isfile(os.path.join(*[CWFID_dataset['images'], f]))
                      ]
    img_file_list.sort()

    cv2.setMouseCallback('image', click_event)

    for filename in img_file_list:
        # reading the image
        img = cv2.imread(filename, 1)
        # displaying the image
        cv2.imshow('image', img)
        # setting mouse handler for the image
        # and calling the click_event() function
        # wait for a key to be pressed to exit
        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()
        jsonfile = os.path.join(*[CWFID_dataset['SAM_annotations'], f"{contraction(path2name(filename))}.json"])
        with open(jsonfile, "w") as outfile:
            json.dump(dictionary, outfile)
    print("list terminated")
    pass