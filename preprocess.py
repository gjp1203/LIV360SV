from PIL import Image
import numpy as np
import scipy.misc
import pandas as pd
import torch
import glob
import os
import cv2

def crop_image(image):
    mask = image > 0

    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)

    # Bounding box of non-black pixels.
    x0, y0, _  = coords.min(axis=0)
    x1, y1, _  = coords.max(axis=0) + 1   # slices are exclusive at the top

    # Get the contents of the bounding box.
    return image[x0:x1, y0:y1]


segmentations = "/segmentations" # Folder containing segmentation masks
originals = "/Originals" # Folder containing street level images

files = glob.glob(segmentations+"/*.pth.tar")
print(files)
redo = []
fileinfo = []
for file in files: 
    labels = torch.load(file, map_location=torch.device('cpu'))['sem_pred'].cpu()
    image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    image[labels == 39] = [40, 40, 40]
    if np.array(labels).max() > 70:
        redo.append({'filename':file})
        print("Redo: " + file)
    else:
        mask = np.zeros_like(image)
        mask[np.all(image == [40,40,40], axis=-1)] = 255

        #find all connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask[:,:,0].astype(np.uint8), connectivity=8)

        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]; nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        min_size = 2000  

        #your answer image
        #for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2 = np.zeros((output.shape))
                img2[output == i + 1] = 255

                # Mask
                maskname = file.replace(".pth.tar", "_"+str(i)+"mask.png").replace('segmentations', 'mask')
                print(maskname)
                cv2.imwrite(maskname, img2) # Expensive step that should be removed.
                im = cv2.imread(maskname) 
                imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(imgray, 127, 255, 0)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                hull = cv2.convexHull(contours[0])
                cv2.fillPoly(im, pts =[hull], color=(255,255,255))
                wContours = cv2.drawContours(im, [hull], 0, (0,255,0), 3)
                cv2.imwrite(maskname, crop_image(wContours))
                wContours_bw = cv2.drawContours(im, [hull], 0, (255,255,255), 3)

                # Masked Image
                print(file.replace(".pth.tar", ".jpg").replace(segmentations, originals))
                original = cv2.imread(file.replace(".pth.tar", ".jpg").replace(segmentations, originals))
                original = cv2.resize(original, (2048,1024))
                #original[(img2 != 255)] = 0.0
                print(wContours_bw.shape)
                original[wContours_bw != 255] = 0.0
                cv2.imwrite(file.replace(".pth.tar", '_'+str(i)+'masked.png').replace('segmentations', 'masked'), crop_image(original))
                fileinfo.append({'size':sizes[i], 'path':file.replace(".pth.tar", '_'+str(i)+'masked.png').replace('segmentations', 'masked')})
                print({'size':sizes[i], 'path':file.replace(".pth.tar", '_'+str(i)+'masked.png').replace('segmentations', 'masked')}) 
df = pd.DataFrame(fileinfo)
df.to_csv('fileinfo.csv')
pd.DataFrame(redo).to_csv('redo.csv')

