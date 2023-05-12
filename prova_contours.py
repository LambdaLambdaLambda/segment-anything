import numpy as np
import cv2
import matplotlib.pyplot as plt

def draw_superposition(img_filename, msk_filename):
    # ".ome.tiff" masks have 255 inside background pixels and 0 inside foliage pixels
    # ".png" masks instead have 1 inside foliage pixels and 0 inside background pixels
    msk = plt.imread(msk_filename).astype(np.uint8)
    if msk_filename.endswith(".ome.tiff"):
        msk = (msk/255).astype(np.uint8)
        msk = 1-msk
    if msk_filename.endswith(".png"):
        pass #nothing to do
    #plt.imshow(msk)
    #plt.title(f"{msk_filename}")
    #plt.show()
    img = plt.imread(img_filename).astype(np.uint8)
    mskgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    contours, hierarchy = cv2.findContours(mskgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    red = (0, 255, 0)
    thickness = 4
    all = -1
    cv2.drawContours(img, contours, all, red, thickness)
    plt.imshow(img)
    plt.title(f"{img_filename}")
    plt.show()
    rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    max_opacity = 255
    rgba[:, :, 3] = np.where(msk == 0, 0.4*max_opacity, max_opacity).astype(np.uint8)
    #plt.imshow(rgba)
    #plt.title(f"{img_filename}")
    #plt.show()
    #new_name = os.path.join(*['elaborated_pictures', f"{strip_extension(binary_msk_filename)}_over_{strip_extension(img_filename)}.png"])
    #Image.fromarray(rgba.astype(np.uint8)).save(new_name)

img_filename = 'ESCA_dataset/esca/pictures/esca_000_cam1.jpg'
msk_filename = 'ESCA_dataset/esca/SAM_masks/esca_000_cam1/esca_000_cam1_mask.png'
draw_superposition(img_filename, msk_filename)

img_filename = 'ESCA_dataset/esca/pictures/esca_000_cam1.jpg'
msk_filename = 'ESCA_dataset/esca/masks/esca_000_cam1_finalprediction.ome.tiff'
draw_superposition(img_filename, msk_filename)