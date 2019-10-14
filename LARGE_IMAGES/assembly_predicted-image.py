#riscala e assembla tutte le roi predette dalla NN per ricreare l'immagine grande grande.
#inoltre applico il TH OTSU e il LABELLING direttamente qui perché questo abbiamo ritenuto essere il metodo migliore.

import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage import io
from skimage.morphology import disk
from skimage.filters import median,gaussian
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.morphology import watershed
import matplotlib.colors as mc
import matplotlib.patches as mpatches
from skimage.transform import rescale, resize


cwd = os.getcwd()

#levo il padding
list_names=[]
for element in os.listdir("/Users/alessandropasqui/Desktop/NN+TH+LAB/predicted_masks_0522-2025-day4-ref1-tile"):
    list_names.append(element[:-4])
for element in list_names:
    im = io.imread(str(cwd)+"/NN+TH+LAB/predicted_masks_0522-2025-day4-ref1-tile/"+str(element)+".png")
    if im.dtype == np.float64:
        im = np.uint8(im * (2 ** 8 - 1))
    h, w = im.shape
    im = im[(16+64):h-(16+64),(16+64):w-(16+64)]
    io.imsave("/Users/alessandropasqui/Desktop/NN+TH+LAB/predicted_masks_0522-2025-day4-ref1-tile-CROPPED/"+str(element)+".tif",im)

#assemblo l'immagine
list_names_ordered = []

large_matrix = np.zeros(shape=(33,45,400,400))
for x in range(1,46):
    for y in range(1,34):

        im = io.imread(str(cwd)+"/NN+TH+LAB/predicted_masks_0522-2025-day4-ref1-tile-CROPPED/x"+str(x)+"-y"+str(y)+".tif")
        if im.dtype == np.float64:
            im = np.uint8(im * (2 ** 8 - 1))

        img = np.uint8(rescale(im, 200.0/64.0, anti_aliasing=False, order=1)*255)
        large_matrix[y-1][x-1]=img


assembled_vertical_line = np.zeros(shape=(45,13200,400))
for x in range(0,45):
    assembled=large_matrix[0][x]
    for y in range(1,33):
        assembled=np.vstack((assembled,large_matrix[y][x]))

    assembled_vertical_line[x] = assembled

assembled_image=np.hstack(assembled_vertical_line)

print(assembled_image)

#median filter + thresholding + labelling

image_median = median(assembled_image/np.max(assembled_image), disk(30)) # applico il filtro mediano all'immagine
threshold_otsu_value = threshold_otsu(image_median) # determinaa soglia per otsu thresholding
binary_otsu = image_median > threshold_otsu_value # applicazione del binary binary_otsu
binary_otsu_labels, nuMarkers = ndi.label(binary_otsu)


io.imsave("/Users/alessandropasqui/Desktop/0522-2025-day4-ref1-tile_PREDICTED.tif",binary_otsu_labels)



#
