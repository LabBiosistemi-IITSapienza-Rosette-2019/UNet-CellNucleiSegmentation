import sys
import numpy as np
import cv2
import os

from skimage.filters import threshold_otsu
from skimage.feature import corner_peaks
from scipy import ndimage as ndi

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

Conv = 0.225 #micron/pixel
bs = 3200 # in pixel

xLbnd = 4000  # 500
xHbnd = 18000  # 12500#im.shape[0]#-xLbnd
yLbnd = 500
yHbnd = 10700

picPath = '/Users/alessandropasqui/Desktop/'
dataPath = '/Users/alessandropasqui/Desktop/'


if __name__ == '__main__':
    #tail = dataPath[:-1]
    #anFolder = tail + '-analisi/'
    #if not os.path.isdir(anFolder):
        #os.mkdir(anFolder)

    #pdfName = anFolder + 'segmentation-analisis.pdf'

    #picNames = os.listdir(picPath)
    #print picNames

    #days = ['0520-1436', '0521-0236','0521-0836', '0521-1313','0521-1647','0522-0447', '0522-2025']#, '0523-2032', '0524-1546']#, '0518-1719','0525-1551']
    days = ['0522-2025-day4-ref1-tile_PREDICTED']
    ndays = len(days)

    if ndays == 1: ndays = 2
    fig0, axs0 = plt.subplots(nrows=ndays, ncols=1, sharex=False, figsize=(7, 14))
    fig1, axs1 = plt.subplots(nrows=ndays, ncols=2, sharex=True, sharey=True, figsize=(14, 14))
    fig2, axs2 = plt.subplots(nrows=ndays, ncols=1, sharex=True, sharey=False, figsize=(4, 14))
    fig3, axs3 = plt.subplots(nrows=ndays, ncols=1, sharex=True, sharey=False, figsize=(4, 14))
    fig4, axs4 = plt.subplots(nrows=ndays, ncols=2, sharex=False, sharey=False, figsize=(4, 14))

    for nday,date in enumerate(days):

        print 'date ', date

        """
        picName = [f for f in picNames if date in f][0]
        picName = picName.split('.')[0]
        pointsName = picName + '-data.pkl'
        try:
            data = pd.read_pickle(dataPath+pointsName)#np.load(dataPath+pointsName)
        except IOError:
            print("Wrong choice: a file named " + picName + "-data.pkl do not exist.\n Make sure of running compGdr in correct place.\nExit...\n")
            sys.exit(1)
        """
        data = pd.read_pickle(dataPath+'data.pkl')
        # apre l'mmagine per avere le dimensioni
        imBase = cv2.imread(picPath + '0522-2025-day4-ref1-tile_PREDICTED.tif',cv2.IMREAD_UNCHANGED)
        im = imBase[yLbnd:yHbnd,xLbnd:xHbnd]

        #plt.figure()
        #plt.imshow(im, cmap=plt.cm.gray, interpolation='nearest')
        #print data
        #print data[:,1], data[:,2]
        #print data[:,4]/data[:,3], data[:,6]/data[:,3]
        #plt.scatter(data[:,2], data[:,1], s=1)
        #plt.scatter(data[:,6]/data[:,3], data[:,4]/data[:,3], s=1)
        #plt.show()

        xdim = im.shape[1] * Conv
        ydim = im.shape[0] * Conv
        #Scelta del range di aree utilizzate per l'analisi
        LWbnd = 100 # lower bound
        UPbnd = 100000 # upper bound
        cleaner1 = data['sum pixels'].values #[:,3:4] # la cernita dei segmentati e' effettuata sulla base dell'area
        cleaner1 = (cleaner1 > LWbnd) & (cleaner1 < UPbnd)
        #cleaner2 = np.float32(data[:,4:5]) # la cernita dei segmentati e' effettuata sulla roi
        #print cleaner2,cleaner2.min(),cleaner2.max()
        #cleaner2 = (cleaner2 > xLbnd) & (cleaner2 < xHbnd)
        #print cleaner2
        #cleaner3 = data[:,6:7] # la cernita dei segmentati e' effettuata sulla roi
        #cleaner3 = (cleaner3 > yLbnd) #& (cleaner3 < xUbnd)
        #Acquisizione dei dati dei segmentati sul range scelto
        cleaner = cleaner1#cleaner1 * cleaner2 #& cleaner3
        #print cleaner
        area = np.float64(data['sum pixels'].values[cleaner])
        r = np.float64(data['sum y'].values[cleaner])
        r2 = np.float64(data['sum y**2'].values[cleaner])
        c = np.float64(data['sum x'].values[cleaner])
        c2 = np.float64(data['sum x**2'].values[cleaner])
        rc = np.float64(data['sum x*y'].values[cleaner])

        #Calcolo di tutti i baricentri
        xB = (c / area) * Conv
        yB = (r / area) * Conv
        #print xB,yB
        xB2 = (c2 / area) * (Conv**2)
        yB2 = (r2 / area) * (Conv**2)
        xyB = (rc / area) * (Conv**2)
        areaB2 = area * (Conv**2)

        ##Calcolo di tutte le matrici dei momenti
        #I_xx = xB2 - xB ** 2
        #I_yy = yB2 - yB ** 2
        #I_xy = xyB - xB * yB
        ##calcolo dell'autovalore massimo e del rispettivo autovettore
        #autov = 0.5 * ( I_xx + I_yy + ( ( I_xx - I_yy) ** 2 + 4 * I_xy ** 2 ) ** 0.5 )
        ## WARNING: nonostante questa scelta possa presentare divisioni per zero, e' tale da fornire comunque il risultato giusto
        #dx = ( I_xy != 0 )
        #I_xy += ( I_xy == 0 ) #in questo modo si evita il warning di di divisione per zero (tanto dx=0 corregge la variazione)
        #dy = dx * ( ( autov - I_xx ) / I_xy - 1 ) + 1
        #norm = ( dx ** 2 + dy ** 2 ) ** 0.5
        #dx = dx / norm
        #dy = dy / norm

        axs0[nday].imshow(imBase, cmap=plt.cm.gray, interpolation='nearest')
        rect = patches.Rectangle((xLbnd,yLbnd),xHbnd-xLbnd,yHbnd-yLbnd,linewidth=1,edgecolor='r',facecolor='none')
        axs0[nday].add_patch(rect)

        cl1 = (xB > xLbnd*Conv) & (xB < xHbnd*Conv) & (yB > yLbnd*Conv) & (yB < yHbnd*Conv)

        axs1[nday,0].imshow(im, cmap=plt.cm.gray, interpolation='nearest')
        axs1[nday,0].scatter(xB[cl1]/Conv-xLbnd,yB[cl1]/Conv-yLbnd, s=1)

        # salva i dati su file
        output = pd.DataFrame({'xB': xB, 'yB': yB})
        output.to_csv(picPath + 'analysis.out', sep=';', index=False)

        # HISTOGRAM stampa l'istogramma delle aree
        NOP = len(area)  # number of particles
        print 'NOP',NOP
        # area *= np.power(Conv, 2)
        mean_area = np.mean(area)
        hhh = axs2[nday].hist(area.ravel(), bins=100, alpha=1.0, normed=True)
        dfAreas = pd.DataFrame({'bins': np.delete(hhh[1], 0), 'freq': hhh[0]})
        dfAreas.to_csv(picPath + 'area-hist.out', sep=';', index=False)
        axs2[nday].set_xlabel('$M\;[\mu m^2]$')
        axs2[nday].set_ylabel('frequency')
        axs2[nday].axvline(mean_area, color='r', label=r'$\overline{A} = %i\;\mu m^2$' % mean_area, linestyle='dashed')
        axs2[nday].legend(loc='upper right', shadow=True, fontsize='x-large')

        # DENSITA' DI SINGOLA PARTICELLA
        binsSize = 400.0
        w = xHbnd - xLbnd
        h = yHbnd - yLbnd
        res = float(binsSize * Conv)
        density, yedges, xedges = np.histogram2d(yB[cl1] / Conv - yLbnd, xB[cl1] / Conv - xLbnd, ( w/binsSize, h/binsSize))
        densityNorm = np.float32(density) / (res ** 2)
        #axs1[nday,1].imshow(density, cmap=plt.cm.gray, interpolation='nearest')
        axs1[nday,1].imshow(density, interpolation='nearest', extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]])

        print 'density.dtype',density.dtype,density.max()
        # HISTOGRAM stampa l'istogramma delle densita'
        # plt.figure(figsize=(8, 6))
        hhh = axs3[nday].hist(density.ravel(), bins=20, alpha=1.0, color="C1", normed=True)
        #hhh = axs3[nday].hist(densityNorm.ravel(), bins=100, alpha=0.85, color="C1", density=True)
        dfDensities = pd.DataFrame({'bins': np.delete(hhh[1], 0), 'freq': hhh[0]})
        dfDensities.to_csv(picPath + 'loc_dens-hist.out', sep=';',index=False)
        axs3[nday].set_xlabel(r'$\rho\;[centroids / \mu m^2]$')
        axs3[nday].set_ylabel('frequency')

        # ------GDR_TOT-------
        # g_tot: gdr calcolata con la strategia di un box piu' piccolo dell'immagine
        box = (xB > (xLbnd + bs)*Conv ) & (xB < (xHbnd - bs)*Conv ) & (yB > (yLbnd + bs)*Conv ) & (yB < (yHbnd - bs)*Conv )

        xTj = xB[box]  # selezione dei baricentri nel box
        yTj = yB[box]
        NOP_box = len(xTj)  # numero di particelle nel box
        print 'NOP_box',NOP_box
        axs1[nday,0].scatter(xTj/Conv-xLbnd,yTj/Conv-yLbnd, s=1)

        rs = []
        for i in range(NOP_box):
            if i%100==0: print i
            Dx = np.float64(xB - xTj[i])
            Dy = np.float64(yB - yTj[i])
            if (Dx ** 2 < 0).any(): print 'Dx min 0', Dx
            if (Dy ** 2 < 0).any(): print 'Dx min 0', Dy
            if ((Dx ** 2 + Dy ** 2) < 0).any(): print 'SumofSq min 0', (Dx ** 2 + Dy ** 2)
            distances = np.sqrt(Dx ** 2 + Dy ** 2)
            rs = rs + list(distances[distances < bs*Conv])
        dr = 8*Conv
        hist, bin_edges = np.histogram(rs, bins= np.arange(1,(bs+1)*Conv, dr),normed=False)
        rho = float(NOP_box) / float((w - 2 * bs) * (h - 2 * bs) * Conv**2 ) #densita' di particelle uniforme sul box
        print 'rho', rho
        #print 'hist', hist
        #print 'r', bin_edges
        r = bin_edges[:-1] + 0.5 * dr
        r0 = bin_edges[:-1]
        r1 = bin_edges[1:]
        deltaV = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)
        g_tot = hist / ( NOP_box * rho * deltaV )
        axs4[nday,0].plot(r,g_tot, '-', c='green', label='$g_{tot}(r)$')
        axs4[nday,1].plot(r,g_tot, '-', c='green', label='$g_{tot}(r)$')
        #axs4[nday,0].plot(bin_edges[:-1]+0.5, g_tot, '-', c='green', label='$g_{tot}(r)$')
        #axs4[nday,0].plot(bin_edges[:-1]+0.5, g_tot, '-', c='green', label='$g_{tot}(r)$')

        #salva i dati su file
        df = pd.DataFrame({'r': r, 'r0': r0, 'r1':r1, 'g_tot': g_tot,'segmented pics folder': [dataPath+ 'data.pkl']*len(r)})
        df.to_csv(dataPath + 'gdr.out', sep=';', index = False)

        #DISPLAY gdr
        #plt.figure(figsize=(4.5, 4))
        axs4[nday,0].set_title(str(nday)+' '+date)
        axs4[nday,0].plot(r, g_tot, '-', c = 'green', label='$g_{tot}(r)$')
        axs4[nday, 0].set_xlim((0,200))
        axs4[nday,0].axhline(1)
        #plt.plot(r, g_ros, '-', c = 'yellow', label='$g_{ros}(r)$')
        axs4[nday,0].set_xlabel('$r\;[\mu m]$')
        axs4[nday,1].plot(r, g_tot, '-', c = 'green', label='$g_{tot}(r)$')
        #plt.plot(r, g_ros, '-', c = 'yellow', label='$g_{ros}(r)$')
        axs4[nday,1].set_xlabel('$r\;[\mu m]$')
        axs4[nday,1].axhline(1)


    fig2.savefig(dataPath + 'area-hist_0522-2025-day4-ref1-tile_PREDICTED.pdf', bbox_inches='tight')
    fig3.savefig(dataPath + 'loc_dens-hist_0522-2025-day4-ref1-tile_PREDICTED.pdf', bbox_inches='tight')
    fig4.savefig(dataPath + 'gdr_0522-2025-day4-ref1-tile_PREDICTED.pdf', bbox_inches='tight')

    plt.show()
