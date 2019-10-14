#con questo codice è possibile creare tre differenti database riscalati nell'intensità e nella taglia delle roi a partire dal nostro GT: uno senza padding ma solo riscalato, uno con paddig nero e uno con padding reflect 

import os
import fnmatch
import numpy as np
from skimage import io
from skimage import color
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
import cv2
from skimage.transform import rescale
from skimage.filters import median,gaussian
from skimage.filters import threshold_otsu
from skimage.morphology import disk
#from torchvision.transforms.functional import pad

cwd = os.getcwd() #sapere in che directory siamo

list_file_names=[]
list_masks_names=[]
list_labels_names=[]

#range(2,5)
for num_day in range(2,5):  #Il programma viene eseguito per tutte le giornate in range
    string_in_file_name = "-day"+str(num_day)+"-ref1-"

    #range(1,11)
    for num_roi in range(1,11): #per ogni giornata si possono fare tutte le roi in range
        list_files_tif=[]
        for file_name in os.listdir(cwd):
            if fnmatch.fnmatch(file_name, '*'+string_in_file_name+str(num_roi)+'-*.tif'):
                list_files_tif.append(file_name)
            if fnmatch.fnmatch(file_name, '*'+string_in_file_name+str(num_roi)+'-*-mask.tif'):
                list_masks_names.append(file_name)
            if fnmatch.fnmatch(file_name, '*'+string_in_file_name+str(num_roi)+'-*-labels.tif'):
                list_labels_names.append(file_name)
        list_file_names.append(max(list_files_tif, key=len))


for element in list_file_names:

    os.makedirs("/Users/alessandropasqui/Desktop/my_data_256_factor2_black-padding_original/cells/"+str(element[:-49])+"/images")
    #os.rename(str(cwd)+"/"+str(element),"/Users/alessandropasqui/Desktop/my_data/cells/"+str(element[:-4])+"/images/"+str(element[:-4])+".png")
    im = io.imread(element)
    im = rescale_intensity(im,out_range=(0,255))
    im = np.uint8(im)
    imRGB = color.gray2rgb(im)
    imRGB = rescale(imRGB, 64.0 / 200.0, anti_aliasing=False)
    #imRGB = rescale(imRGB, 1.0 / 2.0 , anti_aliasing=False)
    #print(imRGB.shape)
    """
    black_matrix = np.zeros(shape=(256,256,3))
    for x in range(64,(64+128)):
        for y in range(64,((64+128))):
            black_matrix[y][x]=imRGB[y-64][x-64]
    """

    imRGB = np.pad(imRGB, ((64, 64), (64, 64), (0, 0)), 'constant', constant_values=0)
    #imRGB = np.pad(imRGB, ((64, 64), (64, 64), (0, 0)), 'reflect')



    print(imRGB.shape)

    #image_median_original = median(imRGB, disk(3)) # applico il filtro mediano all'immagine
    threshold_otsu_value_original = threshold_otsu(imRGB) # determinaa soglia per otsu thresholding
    binary_otsu_original = np.uint8((imRGB > threshold_otsu_value_original)*255)

    io.imsave("/Users/alessandropasqui/Desktop/my_data_256_factor2_black-padding_original/cells/"+str(element[:-49])+"/images/"+str(element[:-49])+".png",imRGB)
    io.imsave("/Users/alessandropasqui/Desktop/my_data_256_factor2_black-padding_original/cells/"+str(element[:-49])+"/mask.png",binary_otsu_original)

#
