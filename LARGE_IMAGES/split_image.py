#divide l'immagine grande grande in tante roi 400x400 ben rinominate con x e y rispettive per poi adare a ricomporre l'immagine.
#inoltre scala le roi in intensità e taglia e crea un database del tipo necessario alla NN del tutorial.

import os
import numpy as np
from skimage import io
from skimage.exposure import rescale_intensity
from skimage import color
from skimage.transform import rescale
from skimage.filters import median
from skimage.morphology import disk
from skimage.filters import threshold_otsu

cwd = os.getcwd()

image_original = io.imread(str(cwd)+"/0522-2025-day4-ref1-tile.tif")
if image_original.dtype == np.float64:
    image_original = np.uint8(image_original * (2 ** 8 - 1))

#image_original = rescale_intensity(image_original,out_range=(0,255))

#tolgo il contorno nero
image_no_black = image_original[50:13629,53:18100]
io.imsave("/Users/alessandropasqui/Desktop/SPLIT/image_no_black.tif",image_no_black)

#conto quante ce ne stanno
y_max = int(image_no_black.shape[0])/400
x_max = int(image_no_black.shape[1])/400
print("x_max = "+ str(x_max))
print("y_max = "+ str(y_max))

#ritaglio l'immagine in modo da farcele stare perfettamente (ho scelto di tagliare a sinistra e in basso perche' mi perdo meno cellule... forse meglio tagliare un po' sopra e un po' sotto ecc...)
#ricordare che numpy_array[y:x]
image_cropped = image_no_black[(image_no_black.shape[0]-(400*y_max)):,(image_no_black.shape[1]-(400*x_max)):]
io.imsave("/Users/alessandropasqui/Desktop/SPLIT/image_cropped.tif",image_cropped)


#taglio le roi e le converto e faccio le maschere
for dy in range(1,y_max+1):
    for dx in range(1,x_max+1):

        os.makedirs("/Users/alessandropasqui/Desktop/SPLIT/my_data/cells/x"+str(dx)+"-y"+str(dy)+"/images")

        roi = image_cropped[400*(dy-1):400*dy,400*(dx-1):400*dx]

        roi = rescale_intensity(roi,out_range=(0,255))
        roi = np.uint8(roi)
        roiRGB = color.gray2rgb(roi)
        roiRGB = rescale(roiRGB, 64.0 / 200.0, anti_aliasing=False)
        imRGB = np.pad(roiRGB, ((64, 64), (64, 64), (0, 0)), 'reflect')

        threshold_otsu_value_original = threshold_otsu(imRGB) # determinaa soglia per otsu thresholding
        binary_otsu_original = np.uint8((imRGB > threshold_otsu_value_original)*255)

        io.imsave("/Users/alessandropasqui/Desktop/SPLIT/my_data/cells/x"+str(dx)+"-y"+str(dy)+"/images/x"+str(dx)+"-y"+str(dy)+".png",imRGB)

        io.imsave("/Users/alessandropasqui/Desktop/SPLIT/my_data/cells/x"+str(dx)+"-y"+str(dy)+"/mask.png",np.uint8(binary_otsu_original))

#
