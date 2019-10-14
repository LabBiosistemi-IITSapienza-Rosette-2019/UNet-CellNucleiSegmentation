#!/usr/bin/env python

import sys
import os
import matplotlib
from skimage import io
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu,threshold_yen,threshold_mean
from skimage.morphology import binary_closing, binary_erosion, binary_opening, disk
import cv2
import numpy as np

#from skimage import measure
#import cupy as cp
#import tifffile as tfile
import pandas as pd

import matplotlib.colors as mc

#import tilesSegUtils as tsu

import seaborn as sns

#checkCpuVer = False
"""

# Set random colors -------------
randCmap = np.random.rand(256, 3)
randCmap[0, :] = 0
cmap = mc.ListedColormap(randCmap)
#----------------------------------

verbose = True

GTpath = '/media/gosti/Data02/20180518-Exp8-post/ref1/GT/'#crop-segGroundTruth-sample/'
GT = os.listdir(GTpath)
GT.sort()

tail='stack-deconv-fab1'#'stack-1-deconv2-Denoice002-BkgSub050'#'stack-1-bgsub'
dd = 2#2
od = 0#4
BkgSub = 0
#maxArea = 640000
minArea = 300
minTh = 15

featuresList = ['area','centroid','convex_area','eccentricity','perimeter','solidity','image','intensity_image','bbox']

#openR = 2
#closeR = 2
otsuPars = [20]#[30]#[40]
resolutionDownScale = 1
startRowz = 0
rowz = 7
"""
if __name__ == '__main__':

    """
    if verbose:
        fig, axs =plt.subplots(nrows=rowz,ncols=3,sharex=True,sharey=True)
        fig, axs1 =plt.subplots(nrows=rowz,ncols=2,sharex=False,sharey=False)
    #fig, axs3 =plt.subplots(nrows=rowz,ncols=len(otsuPars),sharex=False,sharey=False)
    #fig, axs4 =plt.subplots(nrows=2,ncols=len(otsuPars),sharex=False,sharey=False)
    #fig, axs5 =plt.subplots(nrows=2,ncols=len(otsuPars),sharex=True,sharey=True)

    for j,ofr in enumerate(otsuPars):#20
        fig2, axs2 = plt.subplots(nrows=rowz, ncols=11, sharex=False, sharey=False, figsize=(14, 14))
        fig2.suptitle('GT')
        fig3, axs3 = plt.subplots(nrows=rowz, ncols=11, sharex=False, sharey=False, figsize=(14, 14))
        fig3.suptitle('Auto')
        fig4, axs4 = plt.subplots(nrows=rowz, ncols=11, sharex=False, sharey=False, figsize=(14, 14))
        fig4.suptitle('Errors')
        Art = {}
        Bl = {}
        TrueS = {}
        for feature in featuresList:
            Art[feature] = []
            Bl[feature] = []
            TrueS[feature] = []
        if (dd > 0) and (od == 0) and (BkgSub>0):
            tail = tail+'-Denoise'+'{:03d}'.format(dd)+'-BkgSub'+'{:03d}'.format(BkgSub)
            filterOtsuFolder = tail+'-filterOtsu'+'{:03d}'.format(ofr)+ '-regionSizeFilter'+'{:09d}'.format(minArea)+'-AreaIntensityMap/'
        elif (dd == 0) and (od == 0) and (ofr == None):
            filterOtsuFolder = tail+ '-globalOtsu/'
        elif (dd == 0) and (od == 0):
            filterOtsuFolder = tail+ '-filterOtsu' + '{:03d}'.format(ofr) + '-regionSizeFilter'+'{:09d}'.format(minArea)+'-AreaIntensityMap/'
        elif od == 0:
            tail = tail + '-Denoise' + '{:03d}'.format(dd)
            filterOtsuFolder = tail+'-filterOtsu'+'{:03d}'.format(ofr) + '-regionSizeFilter'+'{:09d}'.format(minArea)+'-AreaIntensityMap/'
        else:
            filterOtsuFolder = tail + '-Denoise' + '{:03d}'.format(dd) + 'filterOtsu' + '{:03d}'.format(ofr) + '-open' + '{:03d}'.format(od) +  '-regionSizeFilter' + '{:09d}'.format(minArea) + '-AreaIntensityMap/'

        #filterGrayFolder = tail+'/'
        print 'folder ',filterOtsuFolder
        otsuPics = os.listdir(filterOtsuFolder)
        otsuPics.sort()
        qualityMeasures = {'days': [], 'roi x0': [], 'roi y0': [], 'mean J': [], 'Center of Mass Error': [],
                           'missed GT': [], 'artefactSeg': [], 'blobbed': [], 'GTsegs': []}
        for i,p in enumerate(otsuPics[startRowz:startRowz+rowz]):
            print i,j,p,'ofr',ofr
            areaInt = tfile.imread(filterOtsuFolder + p)[::resolutionDownScale,::resolutionDownScale]
            #im = cv2.imread(filterGrayFolder + p, cv2.IMREAD_UNCHANGED)[::resolutionDownScale,::resolutionDownScale]
            if verbose:
                axs[i,0].imshow(areaInt[::8/resolutionDownScale, ::8/resolutionDownScale])
                axs1[i,0].hist(areaInt.flatten(),bins=100)
            if (ofr == None):
                thOtsu = areaInt
                otsuTh = None
            else:
                otsuTh = threshold_otsu(areaInt)
                print otsuTh
                #plot([otsuTh,otsuTh],[0.0,1.0],'-r')
                thOtsu = areaInt[::resolutionDownScale, ::resolutionDownScale] > otsuTh
            if verbose:
                axs1[i,0].axvline(otsuTh, color='r', linestyle='dashed', linewidth=1)
                axs[i,1].imshow(thOtsu[::8/resolutionDownScale, ::8/resolutionDownScale])
            print p[:-4]
            boxs,fs,oims = tsu.getGTforPic(GT, p[:-6],GTpath)
            for k, (box,f,oim) in enumerate(zip(boxs,fs,oims)):
                print ' -> f', f,box
                x0, y0, w, h = box
                oimLarge = tfile.imread(tail + '/' + p)[y0:y0 + h, x0:x0 + w]
                gtLab = tfile.imread(GTpath + f)
                thSmall = thOtsu[y0:y0 + h, x0:x0 + w]
                #if openR >0: thSmall = binary_opening(thSmall,disk(openR))
                labelSmall = label(thSmall)
                labelSmall = tsu.fillHoles(labelSmall)
                #Graundtruth segments plot
                tsu.drawSegments(axs2[i, k], f[22:], oim, gtLab,cmap)
                # Automatic segments plot
                tsu.drawSegments(axs3[i, k], f[22:], oim, labelSmall,cmap)
                mean_J, sigma1, missedGT, artefactSeg, numBlobbed, numGTSeg,artefacts,blobbed,missed,trueSegs = tsu.measureQuality(labelSmall, gtLab)
                tsu.drawErrors(axs4[i, k],oim,labelSmall,gtLab,artefacts,blobbed,missed,trueSegs)
                Art,Bl,TrueS = tsu.getConMeasures(oimLarge,labelSmall, gtLab, artefacts, blobbed, missed,trueSegs,Art,Bl,TrueS,axs4[i, k])
                qualityMeasures['days'].append(int(p[:4]))
                qualityMeasures['roi x0'].append(x0)
                qualityMeasures['roi y0'].append(y0)
                qualityMeasures['mean J'].append(mean_J)
                qualityMeasures['Center of Mass Error'].append(sigma1)
                qualityMeasures['missed GT'].append(missedGT)
                qualityMeasures['artefactSeg'].append(artefactSeg)
                qualityMeasures['blobbed'].append(numBlobbed)
                qualityMeasures['GTsegs'].append(numGTSeg)
        #plt.figure()
        #pp = sns.pairplot(pd.DataFrame(Art), size=1.8, aspect=1.8,
        #          plot_kws=dict(edgecolor="k", linewidth=0.5),
        #          diag_kind="kde", diag_kws=dict(shade=True))
        #pp = sns.pairplot(concatenated, hue='type', size=1.8, aspect=1.8,
        #                  #palette={"Art": "#FF9999", "Bl": "#FFE888", "TrueS": "#0000FF"},
        #                  plot_kws=dict(edgecolor="black", linewidth=0.5),
        #                  diag_kind="kde", diag_kws=dict(shade=True))
        #Preapare data
        filterFolder2 = filterOtsuFolder[:-4] + 'OtsuTh/'
        if not os.path.isdir(filterFolder2):
            os.mkdir(filterFolder2)
        fig4.savefig(filterFolder2 + filterOtsuFolder[:-1] + '-AIT-th-OTSU-errors.pdf')
        dfArt = pd.DataFrame(Art)
        dfBl = pd.DataFrame(Bl)
        dfTrueS = pd.DataFrame(TrueS)
        dfArt['type'] = 'Art'
        dfBl['type'] = 'Bl'
        dfTrueS['type'] = 'TrueS'
        concatenated = pd.concat([dfArt, dfBl, dfTrueS], ignore_index=True)
        concatenated.to_pickle(filterFolder2 + filterOtsuFolder[:-1] + '-roi-segs.pkl')
        dfQM = pd.DataFrame(qualityMeasures)
        dfQM['globalOtsuSize'] = ofr
        dfQM['dd'] = dd
        dfQM['od'] = od
        dfQM['Intensity Area Th'] = otsuTh
        #dfQM.to_csv(filterOtsuFolder[:-1] +'-open-'+'{:03d}'.format(openR)+'-AIT-th-OTSU-segDiagnostics.csv')
        dfQM.to_csv(filterFolder2 + filterOtsuFolder[:-1] + '-AIT-th-OTSU-segDiagnostics.csv')

        if verbose:
            fig, axs = plt.subplots(nrows=1, ncols=6, sharex=True, sharey=False, figsize=(14, 14))
        temp = dfQM
        temp2 = temp.groupby('days').mean()
        tempStd = temp.groupby('days').sem()
        print temp2
        if verbose:
            for i, labelSt in enumerate(['mean J', 'Center of Mass Error', 'missed GT', 'artefactSeg', 'blobbed', 'GTsegs']):
                axs[i].set_title(labelSt)
                axs[i].scatter(temp['days'], temp[labelSt], alpha=0.4)
                axs[i].errorbar(temp2.index, temp2[labelSt].values, yerr=tempStd[labelSt].values)
            plt.legend()
            #plt.savefig(filterOtsuFolder[:-1] + '-open-'+'{:03d}'.format(openR)+'-AIT-th-OTSU-segDiagnostics.pdf')
            plt.savefig(filterFolder2 + filterOtsuFolder[:-1] + '-AIT-th-OTSU-segDiagnostics.pdf')

        fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=False, figsize=(14, 14))
        temp = dfQM[['days', 'missed GT', 'artefactSeg', 'blobbed']]
        print 'temp', temp
        for labelIn in ['missed GT', 'artefactSeg', 'blobbed']:
            temp[labelIn] = temp[labelIn] / dfQM['GTsegs']
        print 'temp ratio ', temp
        temp2 = temp.groupby('days').mean()
        tempStd = temp.groupby('days').sem()
        print temp2
        if verbose:
            for i, labelSt in enumerate(['missed GT', 'artefactSeg', 'blobbed']):
                axs[i].set_title(labelSt + ' ratio')
                axs[i].scatter(temp['days'], temp[labelSt], alpha=0.4)
                axs[i].errorbar(temp2.index, temp2[labelSt].values, yerr=tempStd[labelSt].values)
            plt.legend()
            #plt.savefig(filterOtsuFolder[:-1] + '-open-'+'{:03d}'.format(openR)+'-AIT-th-OTSU-segRatioDiagnostics.pdf')
            plt.savefig(filterFolder2 + filterOtsuFolder[:-1] + '-AIT-th-OTSU-segRatioDiagnostics.pdf')
        if verbose: plt.show()

    for j, ofr in enumerate(otsuPars):  # 20
        #if verbose:
        #    fig2, axs2 = plt.subplots(nrows=rowz, ncols=5, sharex=False, sharey=False, figsize=(14, 14))
        #    fig3, axs3 = plt.subplots(nrows=rowz, ncols=5, sharex=False, sharey=False, figsize=(14, 14))
        if (dd > 0) and (od == 0) and (BkgSub>0):
            #tail = tail+'-Denoise'+'{:03d}'.format(dd)+'-BkgSub'+'{:03d}'.format(BkgSub)
            filterOtsuFolder = tail+'-filterOtsu'+'{:03d}'.format(ofr)+ '-regionSizeFilter'+'{:09d}'.format(minArea)+'-AreaIntensityMap/'
        elif (dd == 0) and (od == 0) and (ofr == None):
            filterOtsuFolder = tail + '-globalOtsu/'
        elif (dd == 0) and (od == 0):
            filterOtsuFolder = tail + '-filterOtsu' + '{:03d}'.format(ofr) + '-regionSizeFilter' + '{:09d}'.format(minArea) + '-AreaIntensityMap/'
        elif od == 0:
            #tail = tail + '-Denoise' + '{:03d}'.format(dd)
            filterOtsuFolder = tail+'-filterOtsu'+'{:03d}'.format(ofr)+ '-regionSizeFilter' + '{:09d}'.format(minArea) + '-AreaIntensityMap/'
        else:
            filterOtsuFolder = tail + '-Denoise' + '{:03d}'.format(dd) + 'filterOtsu' + '{:03d}'.format(
                ofr) + '-open' + '{:03d}'.format(od) + '-regionSizeFilter' + '{:09d}'.format(
                minArea) + '-AreaIntensityMap/'


        for i,p in enumerate(otsuPics[startRowz:startRowz+rowz]):
            print i,j,p,'ofr',ofr
            areaInt = tfile.imread(filterOtsuFolder + p)[::resolutionDownScale,::resolutionDownScale]
            otsuTh = threshold_otsu(areaInt)
            print otsuTh
            thOtsu = areaInt[::resolutionDownScale, ::resolutionDownScale] > otsuTh
            #if openR >0: thOtsu = binary_opening(thOtsu,disk(openR))
            labels = label(thOtsu)
            print 'labels',labels.dtype,labels.min(),labels.max()
            #filterFolder2 = tail + '-Denoise' + '{:03d}'.format(dd) + 'filterOtsu' + '{:03d}'.format(
            #    ofr) + '-regionSizeFilter' + '{:09d}'.format(minArea) + '-open' + '{:03d}'.format(openR)+'-AreaIntensityOtsuTh/'
            filterFolder2 = filterOtsuFolder[:-4]+'OtsuTh/'
            if not os.path.isdir(filterFolder2):
                os.mkdir(filterFolder2)
            tfile.imsave(filterFolder2 + p[:-4] + '.tif',np.uint32(labels))
            coords = np.transpose(labels.nonzero())
            # calcola: [0] l'area; [1] sum_x; [3] sum_(x^2); [4] sum_y; [5] sum_(y^2); [6] sum_(x*y);
            results = np.zeros((labels.max(), 7), dtype=np.uint64)
            for r, c in coords:
                labe = labels[r, c]
                results[labe - 1] += np.uint64([0, 1, r, r ** 2, c, c ** 2, r * c])
            results[:,0] = np.arange(1,labels.max()+1)
            dfPoints = pd.DataFrame(results, columns=['label','sum pixels', 'sum y', 'sum y**2', 'sum x', 'sum x**2', 'sum x*y'])
            dfPoints.to_csv(filterFolder2 + p[:-4] + '-data.csv')
            dfPoints.to_pickle(filterFolder2 + p[:-4] + '-data.pkl')
            #np.save(filterFolder2 + p[:-4] + '-data.npy', results)
            #np.savetxt(filterFolder2 + p[:-4] + '-data.out', results, fmt='%d')
    #plt.savefig('fig.eps')


    """

    #for i,p in enumerate(otsuPics[startRowz:startRowz+rowz]):
        #print i,j,p,'ofr',ofr
    areaInt = io.imread("/Users/alessandropasqui/Desktop/0522-2025-day4-ref1-tile_PREDICTED.tif")#[::resolutionDownScale,::resolutionDownScale]
    #otsuTh = threshold_otsu(areaInt)
    #print otsuTh
    #thOtsu = areaInt[::resolutionDownScale, ::resolutionDownScale] > otsuTh
    #if openR >0: thOtsu = binary_opening(thOtsu,disk(openR))
    #labels = label(thOtsu)
    #print 'labels',labels.dtype,labels.min(),labels.max()
    #filterFolder2 = tail + '-Denoise' + '{:03d}'.format(dd) + 'filterOtsu' + '{:03d}'.format(
    #    ofr) + '-regionSizeFilter' + '{:09d}'.format(minArea) + '-open' + '{:03d}'.format(openR)+'-AreaIntensityOtsuTh/'
    #filterFolder2 = filterOtsuFolder[:-4]+'OtsuTh/'
    #if not os.path.isdir(filterFolder2):
        #os.mkdir(filterFolder2)
    #tfile.imsave(filterFolder2 + p[:-4] + '.tif',np.uint32(labels))
    coords = np.transpose(areaInt.nonzero())
    # calcola: [0] l'area; [1] sum_x; [3] sum_(x^2); [4] sum_y; [5] sum_(y^2); [6] sum_(x*y);
    results = np.zeros((areaInt.max(), 7), dtype=np.uint64)
    for r, c in coords:
        labe = areaInt[r, c]
        results[labe - 1] += np.uint64([0, 1, r, r ** 2, c, c ** 2, r * c])
    results[:,0] = np.arange(1,areaInt.max()+1)
    dfPoints = pd.DataFrame(results, columns=['label','sum pixels', 'sum y', 'sum y**2', 'sum x', 'sum x**2', 'sum x*y'])
    dfPoints.to_csv("/Users/alessandropasqui/Desktop/data.csv")
    dfPoints.to_pickle("/Users/alessandropasqui/Desktop/data.pkl")
