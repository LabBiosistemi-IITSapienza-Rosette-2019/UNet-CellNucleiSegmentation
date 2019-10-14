#rimuove il padding nelle immagini predette dalla NN precedentemente applicato

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

cwd = os.getcwd()

#LEVO PADDING (64+16 PIXELS oppure solo 16 PIXELS)
for element in os.listdir(str(cwd)+"/predicted_masks_no-padding_smussed0.4_500E_original"):
    im = io.imread(str(cwd)+"/predicted_masks_no-padding_smussed0.4_500E_original/"+str(element))
    if im.dtype == np.float64:
        im = np.uint8(im * (2 ** 8 - 1))

    h, w = im.shape
    #im = im[(16+64):h-(16+64),(16+64):w-(16+64)]       #QUESTO PER BLACK/REFLECT - PADDING
    im = im[(16):h-(16),(16):w-(16)]                    #QUESTO PER NO - PADDING

    io.imsave("/Users/alessandropasqui/Desktop/valutazione_no-padding_smussed0.4_500E_original/"+str(element[:-4])+".tif",im)


#
