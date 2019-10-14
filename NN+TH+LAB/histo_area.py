#grafica l'istogramma delle aree delle cellule nel database del tutorial e poi di quelle nel nostro GT e ne calcola le rispettive medie

import os
import fnmatch
import numpy as np
from skimage import io
from skimage import color
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi

cwd = os.getcwd()

list_area=[]

for element in os.listdir(str(cwd)+"/immagini_database_tutorial"):
    im = io.imread(str(cwd)+"/immagini_database_tutorial/"+str(element))
    if im.dtype == np.float64:
        im = np.uint8(im * (2 ** 8 - 1))

    im_labels, nuMarkers = ndi.label(im)
    #print(im_labels)
    labels, counts = np.unique(im_labels, return_counts=True)
    #print(counts)
    list_area = list_area+list(counts)
    #print(list_area)

counts_cropped = [i for i in list(counts) if i <=2000]
mean=sum(list(counts_cropped))/len(list(counts_cropped))
#var_1 = sum([(x)**2 for x in counts_cropped])/len(counts_cropped)
#var = (var_1 - (mean**2))/len(counts_cropped) #CONTROLLARE SE GIUSTA LA VARIANZA
print("mean: "+str(mean))#+"  var: "+str(var))

plt.hist(list_area, bins='auto',range=(0,2000))
plt.show()


for element in os.listdir(str(cwd)+"/immagini_gt_nostro"):
    im = io.imread(str(cwd)+"/immagini_gt_nostro/"+str(element))
    if im.dtype == np.float64:
        im = np.uint8(im * (2 ** 8 - 1))

    #im = (im!=255)*255
    #print(im)
    #im_labels, nuMarkers = ndi.label(im)
    labels, counts = np.unique(im, return_counts=True)
    print(counts)
    list_area = list_area+list(counts)
    #print(list_area)

counts_cropped = [i for i in list(counts) if i <=2000]
mean=sum(list(counts_cropped))/len(list(counts_cropped))
#var_1 = sum([(x)**2 for x in counts_cropped])/len(counts_cropped)
#var = (var_1 - (mean**2))/len(counts_cropped) #CONTROLLARE SE GIUSTA LA VARIANZA
print("mean: "+str(mean))#+"  var: "+str(var))


plt.hist(list_area, bins='auto',range=(0,2000))
plt.show()


#
