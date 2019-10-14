#dopo aver predetto le immagini con la NN, questo codice applica il metodo otsu e il labelling, mostra le associazioni tra "auto" e "thruth" con le frecce, calcola i valori medi e gli errori per i parametri di valutazione dell'accuratezza

#NOTA: questo file deve stare in una cartella contenente tutti i file del GT (<name>.tif e <name>-labels.tif)



import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import matplotlib.axes as ax
from skimage import io
from skimage.morphology import disk
from skimage.filters import median,gaussian
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.morphology import watershed
import matplotlib.colors as mc
import matplotlib.patches as mpatches
from skimage.transform import rescale


#funzione per le frecce da Giorgio
def drawMatch(match,finalLabels,autoLabels):
    randCmap = np.random.rand(256, 3)
    randCmap[0, :] = 0
    cmap = mc.ListedColormap(randCmap)
    f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[1].set_title('truth')
    ax[1].imshow(finalLabels, cmap=cmap)
    ax[0].set_title('auto')
    ax[0].imshow(autoLabels, cmap=cmap)
    for a, b in match.items():
        #print a,b
        xya = np.mean(np.where(finalLabels == a), axis=1)
        xyb = np.mean(np.where(autoLabels == b), axis=1)
        #print xya,xyb
        coordsA = "data"
        coordsB = "data"
        con = mpatches.ConnectionPatch(xyA=xya[::-1], xyB=xyb[::-1], coordsA=coordsA, coordsB=coordsB,
                              axesA=ax[1], axesB=ax[0],
                              arrowstyle="->", shrinkB=5, color=cmap(a))
        ax[1].add_artist(con)
    return f, ax


cwd = os.getcwd() #sapere in che directory siamo

list_file_names=[]

day2_list_J = []
day3_list_J = []
day4_list_J = []

day2_list_missed = []
day3_list_missed = []
day4_list_missed = []

day2_list_artefacts = []
day3_list_artefacts = []
day4_list_artefacts = []

day2_list_blobbed = []
day3_list_blobbed = []
day4_list_blobbed = []

J_totale_mean_list=[]
J_totale_std_list=[]
day2_list_j_totale=[]
day3_list_j_totale=[]
day4_list_j_totale=[]


#range(2,5)
for num_day in range(2,5):  #Il programma viene eseguito per tutte le giornate in range
    string_in_file_name = "-day"+str(num_day)+"-ref1-"

    #range(1,11)
    for num_roi in range(1,11): #per ogni giornata si possono fare tutte le roi in range
        list_files_tif=[]
        for file_name in os.listdir(cwd):
            if fnmatch.fnmatch(file_name, '*'+string_in_file_name+str(num_roi)+'-*.tif'):
                list_files_tif.append(file_name)
        list_file_names.append(min(list_files_tif, key=len))

for radius_input in range(3,4): #Viene eseguito il programma per i valori del raggio (della mediana) per i valori in range
    for min_distance_input in range(30,31): #Viene eseguito il programma per i valori della distanza (della ricerca del minimo) per i valori in range

        for element in list_file_names:
            print(element)

            path = cwd + "/" + element # creazione del path finale
            print path
            image = io.imread(path, as_gray=True) # carico l'immagine originale da analizzare
            if image.dtype == np.float64:
                image = np.uint8(image * (2 ** 8 - 1))

            image = rescale(image, 200.0 / 64.0, anti_aliasing=False)

            print(np.max(image))
            print(image.dtype)

            #WATERSHED

            image_median = median(image, disk(radius_input)) # applico il filtro mediano all'immagine

            threshold_otsu_value = threshold_otsu(image_median) # determinaa soglia per otsu thresholding

            binary_otsu = image_median > threshold_otsu_value # applicazione del binary binary_otsu

            binary_otsu_labels, nuMarkers = ndi.label(binary_otsu)

            #MATCHING

            path_R=path[:-4]+'-labels.tif' # creo il path giusto della immagine labellata R con programma Giorgio
            R = io.imread(path_R) # carico l'immagine R
            S = binary_otsu_labels # definisco S come l'immagine labellata watershed

            match_diz=dict() #dizionario key=label di R e value=label di S che copre piu' del 50percento
            J_list=[] #Lista di J per ogni roi ciascuna contenente tutti valori di j per ogni cellula presente nella roi


            #CALCOLO J TOTALE
            TH_labelled = (R > 0)
            matrice_moltiplicazione = (TH_labelled * binary_otsu)

            numero_di_labels, numero_di_pixel_bianchi_contati = np.unique(matrice_moltiplicazione*255, return_counts=True)
            numero_di_labels_per_normalizzare, numero_di_pixel_bianchi_contati_per_normalizzare = np.unique(TH_labelled*255, return_counts=True)
            print(numero_di_labels)
            print(numero_di_pixel_bianchi_contati)
            print(numero_di_labels_per_normalizzare)
            print(numero_di_pixel_bianchi_contati_per_normalizzare)

            j_totale = float(numero_di_pixel_bianchi_contati[1]) / float(numero_di_pixel_bianchi_contati_per_normalizzare[1])



            print "\n"

            for i in range(1,np.max(R)+1): # ciclo che scorre su tutte le cellule labellate nel file R

                if(np.count_nonzero(R==i)!=0): #Condizione affinche' si consideri una cellula esistente e non sfondo

                    print "cell label: " + str(i) #Label della cellula i-esima

                    num_pixels_R = np.count_nonzero(R==i) # numero di pixel che non sono False della matrice R==1 (arancione nel disegne)

                    intersection = (R==i)*S # matrice di intersezione tra la cellula iesima di R e S

                    labels, counts = np.unique(intersection, return_counts=True) # Per ogni label che compare nella matrice intersezione conto quanti pixel ci sono per quel label
                    num_pixels_intersection = dict(zip(labels, counts)) #dizionario che ha come key i labels e come value i conteggi di pixel corrispondenti
                    #print("dizionario cellula iesima: ",num_pixels_intersection)

                    #inizializzazione
                    num_pixels_S=0 # numero di pixel contenuti nelle cellule individuate dal Watershed che coprono almeno parzialmente la cellula iesima di R (le palloche blu disegno)
                    count_intersec_tot=0 # numero di pixel contenuti nell'intersezione tra la cellula iesima di R e le cellule separate con watershed che si sovrappongono almeno parzialmente con lei (zone gialle)
                    j_best=0
                    for j in range(1,labels.size): #ciclo sui labels che compaiono nell'array dell'intersezione (per la cellula iesima che stiamo considerando)

                        if(counts[j]>(num_pixels_R*0.5)): #condizione che controlla la percentuale di cellula iesmia (di R) sovrapposta alla cellula segmentata (di S) per ogni label (ciclo for j)
                            percentuale= (float(counts[j])/float(num_pixels_R)) # se condizione soddisfatta calcolo percentuale sovrapposizione
                            #print("percentuale sovrapposizione: ", percentuale) #stampo percentuale sovraposizione solo nel caso in cui sovrapp maggiore 50 perc
                            j_best= j # mi dice il j corrispondente al count maggiore che copre + del 50 percento
                            label_best = labels[j] #il label della S corrispondente al count maggiore che copre piu del 50 percento
                            match_diz[i]=label_best #Creiamo un dizionario in cui abbiamo come keys i label di R e come value l'indice di S corrispondente a quelli che coprono piu del 50%

                        num_pixels_S += np.count_nonzero(S==labels[j]) # aggiornamento di num_pixels_S per ogni label trovato dalla sovrapposizione
                        count_intersec_tot += counts[j] #aggiornamento del numero totale di pixel in count inter tot

                    if(labels.size==1): #label.size e la quantita di label di S che compaiono nella array di interesezione per la cellula i-esima_ quindi quando questo valore coincide con 1 il label nella matrice S corrisponde a 0 (sfondo)
                        match_diz[i]=-1 #A quel valore nel dizionario di matching assegno -1 (missed)

                    if(labels.size==1 or j_best==0): #se non c'e per niente sovrapposizone tra R e S cioe R matcha con lo sfondo di s oppure la cellula viene spezzettata in cosi tante parti che nessun j copre piu del 50% (quindi viene esclusa dalla condizione precedente)
                        count_intersec = 0 #allora l'intersezione del label che copre piu del 50 perc e' 0
                    else:
                        count_intersec = counts[j_best] #altrimenti mi salvo il numero di pixel corrispondenti al label che copre piu del 50 percento

                    #print(num_pixels_R)
                    #print(num_pixels_S)
                    #print(count_intersec_tot)

                    union = num_pixels_R + num_pixels_S - count_intersec_tot # unione tra cellula iesima di R e tutte le cellule di S che si sovrappongono almeno in parte con lei

                    J = (float(count_intersec)/float(union)) #calcolo J come rapporto tra intersezione /unione
                    print "J = " + str(J)
                    J_list.append(J) #aggiungo il valore di J iesimo alla lista corrispondente
                    #print("J = ", J)

                else: #Se R==i e vuoto non viene eseguita la condizione di sopra (non ci sono pixel corrispondenti a quel label)
                    print("zero pixels for label " + str(i))

            print "\n"

            J_mean=sum(J_list)/len(J_list) #Il valor medio di J per la roi consederata
            print "J_mean = " + str(J_mean)

            print "match_diz = " + str(match_diz) #stampo il dizionario del match

            #drawMatch(match_diz,R,S)
            #plt.show()

            num_missed = match_diz.values().count(-1) #Numero di cellule missed (quelle associate al valore -1)
            print "num_missed = " + str(num_missed)

            labels_S = np.unique(S).tolist()  #Consideriamo tutti i label di S e li inseriamo nella lista labels_S perche ora abbiamo bisogno di tutti i label di S e non solo quelli associati ad una cellula di R per conoscere quelli inventati (artefatti)

            labels_artefacts = set(labels_S) ^ set(match_diz.values()) #Confronto i valori delle due liste, cioe quella dei labels di S totali (definita sopra) e quelli all'intenrno del dizionario che matcha R con S. Dopodiche inserisce i valori non in comune in una lista
            num_artefacts = len(labels_artefacts) #Voglio sapere quanti sono i valori differenti confrontati prima e quindi considero la lunghezza della lista labels_artefacts

            print "num_artefacts = "+ str(num_artefacts)

            #Voglio contare i blob: quindi devo capire quanti keys puntano allo stesso valore
            dict_blobbed=dict((x,match_diz.values().count(x)) for x in set(match_diz.values())) #crea un dizionario in cui le keys sono un certo valore di match e i values sono quante volte compariva quel valore nel vecchio dizionario
            num_blob=0
            for key in dict_blobbed.keys(): #ciclo che permette di contare il numero di blob
                if(key!=-1 and dict_blobbed[key]!=1): #condizione necessaria per escludere i missed (quelle con key = -1) e i valori non ripetuti nel dizionario (che sono banalmente non blobbed)
                    num_blob = num_blob+1
            print "num_blob = " + str(num_blob)

            print "\n"

            #NORMALIZZAZIONE

            labels_R = np.unique(R).tolist()

            numtot_groundtruth = len(labels_R)-1
            print "numtot_groundtruth = " + str(numtot_groundtruth)

            num_blob_norm = float(num_blob)/float(numtot_groundtruth)
            num_artefacts_norm = float(num_artefacts)/float(numtot_groundtruth)
            num_missed_norm = float(num_missed)/float(numtot_groundtruth)
            print "num_blob_norm = " + str(num_blob_norm)
            print "num_artefacts_norm = " + str(num_artefacts_norm)
            print "num_missed_norm = " + str(num_missed_norm)

            #LISTE DI MISSED, ARTEFATTI, BLOBBED

            print "\n"

            if (fnmatch.fnmatch(element, '*-day2-*')):
                day2_list_J.append(J_mean)
                day2_list_missed.append(num_missed_norm)
                day2_list_artefacts.append(num_artefacts_norm)
                day2_list_blobbed.append(num_blob_norm)
                day2_list_j_totale.append(j_totale)

            if (fnmatch.fnmatch(element, '*-day3-*')):
                day3_list_J.append(J_mean)
                day3_list_missed.append(num_missed_norm)
                day3_list_artefacts.append(num_artefacts_norm)
                day3_list_blobbed.append(num_blob_norm)
                day3_list_j_totale.append(j_totale)


            if (fnmatch.fnmatch(element, '*-day4-*')):
                day4_list_J.append(J_mean)
                day4_list_missed.append(num_missed_norm)
                day4_list_artefacts.append(num_artefacts_norm)
                day4_list_blobbed.append(num_blob_norm)
                day4_list_j_totale.append(j_totale)


        J_mean_day2 = sum(day2_list_J)/float(len(day2_list_J))
        J_std_day2 = np.std(np.asarray(day2_list_J))/np.sqrt(len(day2_list_J))
        J_mean_day3 = sum(day3_list_J)/float(len(day3_list_J))
        J_std_day3 = np.std(np.asarray(day3_list_J))/np.sqrt(len(day3_list_J))
        J_mean_day4 = sum(day4_list_J)/float(len(day4_list_J))
        J_std_day4 = np.std(np.asarray(day4_list_J))/np.sqrt(len(day4_list_J))


        missed_mean_day2 = sum(day2_list_missed)/float(len(day2_list_missed))
        missed_std_day2 = np.std(np.asarray(day2_list_missed))/np.sqrt(len(day2_list_missed))
        missed_mean_day3 = sum(day3_list_missed)/float(len(day3_list_missed))
        missed_std_day3 = np.std(np.asarray(day3_list_missed))/np.sqrt(len(day3_list_missed))
        missed_mean_day4 = sum(day4_list_missed)/float(len(day4_list_missed))
        missed_std_day4 = np.std(np.asarray(day4_list_missed))/np.sqrt(len(day4_list_missed))

        artefacts_mean_day2 = sum(day2_list_artefacts)/float(len(day2_list_artefacts))
        artefacts_std_day2 = np.std(np.asarray(day2_list_artefacts))/np.sqrt(len(day2_list_artefacts))
        artefacts_mean_day3 = sum(day3_list_artefacts)/float(len(day3_list_artefacts))
        artefacts_std_day3 = np.std(np.asarray(day3_list_artefacts))/np.sqrt(len(day3_list_artefacts))
        artefacts_mean_day4 = sum(day4_list_artefacts)/float(len(day4_list_artefacts))
        artefacts_std_day4 = np.std(np.asarray(day4_list_artefacts))/np.sqrt(len(day4_list_artefacts))

        blob_mean_day2 = sum(day2_list_blobbed)/float(len(day2_list_blobbed))
        blob_std_day2 = np.std(np.asarray(day2_list_blobbed))/np.sqrt(len(day2_list_blobbed))
        blob_mean_day3 = sum(day3_list_blobbed)/float(len(day3_list_blobbed))
        blob_std_day3 = np.std(np.asarray(day3_list_blobbed))/np.sqrt(len(day3_list_blobbed))
        blob_mean_day4 = sum(day4_list_blobbed)/float(len(day4_list_blobbed))
        blob_std_day4 = np.std(np.asarray(day4_list_blobbed))/np.sqrt(len(day4_list_blobbed))

        J_totale_mean_day2 = sum(day2_list_j_totale)/float(len(day2_list_j_totale))
        J_totale_std_day2 = np.std(np.asarray(day2_list_j_totale))/np.sqrt(len(day2_list_j_totale))
        J_totale_mean_day3 = sum(day3_list_j_totale)/float(len(day3_list_j_totale))
        J_totale_std_day3 = np.std(np.asarray(day3_list_j_totale))/np.sqrt(len(day3_list_j_totale))
        J_totale_mean_day4 = sum(day4_list_j_totale)/float(len(day4_list_j_totale))
        J_totale_std_day4 = np.std(np.asarray(day4_list_j_totale))/np.sqrt(len(day4_list_j_totale))


        J_mean_list = [J_mean_day2, J_mean_day3,  J_mean_day4]
        J_std_list = [J_std_day2, J_std_day3, J_std_day4]
        missed_mean_list = [missed_mean_day2, missed_mean_day3,  missed_mean_day4]
        missed_std_list = [missed_std_day2, missed_std_day3, missed_std_day4]
        artefacts_mean_list = [artefacts_mean_day2, artefacts_mean_day3,  artefacts_mean_day4]
        artefacts_std_list = [artefacts_std_day2, artefacts_std_day3, artefacts_std_day4]
        blobbed_mean_list = [blob_mean_day2, blob_mean_day3,  blob_mean_day4]
        blobbed_std_list = [blob_std_day2, blob_std_day3, blob_std_day4]
        J_totale_mean_list = [J_totale_mean_day2, J_totale_mean_day3,  J_totale_mean_day4]
        J_totale_std_list = [J_totale_std_day2, J_totale_std_day3, J_totale_std_day4]


        list_days = range(2,5)

        #GRAFICI

        x1 = list_days
        y1 = J_mean_list
        #yerr1=np.std(np.asarray(y1))/np.sqrt(len(y1))

        x2 = list_days
        y2 = missed_mean_list
        #yerr2=np.std(np.asarray(y2))/np.sqrt(len(y2))

        x3 = list_days
        y3 = artefacts_mean_list
        #yerr3=np.std(np.asarray(y3))/np.sqrt(len(y3))

        x4 = list_days
        y4 = blobbed_mean_list
        #yerr4=np.std(np.asarray(y4))/np.sqrt(len(y4))


        x5 = list_days
        y5 = J_totale_mean_list


        plt.subplot(1,5,1)
        #plt.plot(x1, y1, 'o-')
        plt.errorbar(x1, y1, yerr=J_std_list, label='radius: ')
        plt.title ('J mean')
        plt.xlabel('day')


        plt.subplot(1,5,2)
        #plt.plot(x2, y2, 'o-')
        plt.errorbar(x2, y2, yerr=missed_std_list, label='radius: ')
        plt.title ('missed')
        plt.xlabel('day')

        plt.subplot(1,5,3)
        #plt.plot(x3, y3, 'o-')
        plt.title ('artefacts')
        plt.errorbar(x3, y3, yerr=artefacts_std_list, label='radius: ')
        plt.xlabel('day')

        plt.subplot(1,5,4)
        #plt.plot(x4, y4, 'o-')
        plt.errorbar(x4, y4, yerr=blobbed_std_list, label='radius: ')
        plt.title ('blobbed')
        plt.xlabel('day')

        plt.subplot(1,5,5)
        #plt.plot(x1, y1, 'o-')
        plt.errorbar(x5, y5, yerr=J_totale_std_list, label='radius: ')
        plt.title ('J mean tot')
        plt.xlabel('day')

        plt.tight_layout()

        plt.savefig('NN+TH+LAB_valutazione_reflect_padding_smussed_0.4_original.png')
