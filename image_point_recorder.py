# importing the module
import cv2
import os
import json

CWFID_dataset = {
        'annotations': os.path.join(*['CWFID_dataset', 'annotations']),
        'SAM_annotations': os.path.join(*['CWFID_dataset', 'SAM_annotations']),
        'images': os.path.join(*['CWFID_dataset', 'images']),
        'masks': os.path.join(*['CWFID_dataset', 'masks']),
        'SAM_masks': os.path.join(*['CWFID_dataset', 'SAM_masks'])
    }

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        # Data to be written
        dictionary = {
            'filename': os.path.join(*['CWFID_dataset', 'SAM_annotations', '001_annotation.json']),
            'add_points': [],
            'rem_ponts': []
        }

        with open("sample.json", "w") as outfile:
            json.dump(dictionary, outfile)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x, y), font, 1,
                    (255, 255, 0), 2)
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

    for filename in img_file_list:
        # reading the image
        img = cv2.imread(filename, 1)
        # displaying the image
        cv2.imshow('image', img)
        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image', click_event)
        # wait for a key to be pressed to exit
        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()