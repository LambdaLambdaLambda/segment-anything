import numpy as np
import cv2
import matplotlib.pyplot as plt

def draw_superposition(img_filename, msk_filename):
    img = plt.imread(img_filename).astype(np.uint8)
    rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    max_opacity = 255
    # ".ome.tiff" masks have 255 inside background pixels and 0 inside foliage pixels
    # ".png" masks instead have 1 inside foliage pixels and 0 inside background pixels
    msk = plt.imread(msk_filename).astype(np.uint8)
    if msk_filename.endswith(".ome.tiff"):
        msk = (msk/255).astype(np.uint8)
        msk = 1-msk
    if msk_filename.endswith(".png"):
        pass #nothing to do
    rgba[:, :, 3] = np.where(msk == 0, 0.4 * max_opacity, max_opacity).astype(np.uint8)
    #mskgray = cv2.cvtColor(msk, cv2.COLOR_RGB2GRAY)
    contours, hierarchy = cv2.findContours(msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    red = (0, 255, 0)
    thickness = 4
    all = -1
    msk_contours = np.ones_like(msk)
    cv2.drawContours(msk_contours, contours, all, red, thickness)
    # modify red channel to draw contours in red
    rgba[:, :, 0] = np.where(msk_contours == 0, 255, rgba[:, :, 0]).astype(np.uint8)
    # green channel to draw contours in red
    rgba[:, :, 1] = np.where(msk_contours == 0, 0, rgba[:, :, 1]).astype(np.uint8)
    # blue channel to draw contours in red
    rgba[:, :, 2] = np.where(msk_contours == 0, 0, rgba[:, :, 2]).astype(np.uint8)
    plt.imshow(rgba)
    plt.title(f"{img_filename} with transparency and contours")
    plt.show()
    #plt.imshow(msk)
    #plt.title(f"{msk_filename}")
    #plt.show()
    #plt.imshow(msk_contours*255)
    #plt.title(f"Contours of {msk_filename}")
    #plt.show()
    #new_name = os.path.join(*['elaborated_pictures', f"{strip_extension(binary_msk_filename)}_over_{strip_extension(img_filename)}.png"])
    #Image.fromarray(rgba.astype(np.uint8)).save(new_name)

img_filename = 'ESCA_dataset/esca/pictures/esca_000_cam1.jpg'
msk_filename = 'ESCA_dataset/esca/SAM_masks/esca_000_cam1/esca_000_cam1_mask.png'
#img_filename = 'ESCA_dataset/esca/pictures/esca_000_cam1.jpg'
#msk_filename = 'ESCA_dataset/esca/masks/esca_000_cam1_finalprediction.ome.tiff'

draw_superposition(img_filename, msk_filename)