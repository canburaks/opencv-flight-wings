"""
@author:
"""
# This python script reads undistorted thermal images, crops and enhances,
# finds transition and separation locations, marks on the image and saves the data
import cv2
import numpy as np
#from matplotlib import pyplot as plt
#import math
#import pylab
import os
#from pylab import arange, plot, sin, ginput, show
#import win32api
#import ctypes
import tkinter as tk
from tkinter import filedialog
#import peakutils
#import scipy
#import time
from scipy.signal import find_peaks_cwt
import sys
#import scipy.stats.stats as st


def main():

    global filename, path, fileout, counter, fint, inputfile
    global xOcList, curveFit_mList, curveFit_nList
    global left, right, top, bottom
    global thresh01s, thresh01p, threshSigmaTr, threshSigmaSp, threshPixelTr, threshPixelSp, threshStddevTr, threshStddevSp

    # Select the folder with thermography images
    Mbox('Image Location', 'Please select the folder where images are located', 0)
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory()

   ##  PAREMETERS TO CHANGE ACCORDING TO IMAGE     #####

    # Parameters to check the peaks of locally found transition and separation points.
    # Changes between 0 and 1.
    # Higher value means more clean data but may cause loss of information
    # such as the locations of transition and separation locations.
    thresh01s = 0.3  # suction side (upper)
    thresh01p = 0.5  # pressure side (lower)

    # Parameters for validity check around average (transition and separation),
    # localTrx-spanwise avg
    # as a factor of respective standard deviation values
    threshSigmaTr = 0.5
    threshSigmaSp = 1  # b
    # 1 den 0.6 ya indirince kaybettik bazi imajlard abuldugu yeri
    # 1 den 0.8 yapinca scattered olan line geri geldi .
    # artirinca dhaa scatter oluyor ama cok scatter oldugundan ikinci kriter std ye takiliyor ondan cikti vermiyor olabilir.
   # o yiuzden 1 yapinca threshStddevSp kriteria yi da 6 yaptim

   # Yukaridan gecen noktalar

   # Parameters to check the std deviation levels of spanwise transition/separation distributions
    threshStddevTr = 12
    threshStddevSp = 12

    # Parameters to check number of valid spanwise points
    threshPixelTr = 8
    threshPixelSp = 8

   #########################################################################################

    # Open the output data file
    fileout = open('tr-out.dat', 'w')
    fileout.write(
        'Run No \t AoA \t Tr Pixel \t \t Xtr/C \t \t Sp Pixel \t Xsp/C \t \t Up/Down \n')

    # Read the commandline input (name of the input file that contains calibration data and crop boundary information)
    inputfile = sys.argv[1]
    filein = open(inputfile, 'r')
    # Read the file line by line and fill all data into the list "curveFit"
    curveFit = filein.readlines()
    ccf1 = np.array(curveFit)  # convert the list to an numpy array
    ccf2 = ccf1.astype(np.float)  # convert to float
    print(ccf2, "ccf2")
    # Number of chordwise position data in the calibration images
    noChordPos = ccf2[0]

    xOcList = []  # List of chordwise calibration point positions
    curveFit_mList = []  # List of m coefficients for the linear curvefits for each chordwise position that calibration is performed
    curveFit_nList = []  # List of n coefficients for the linear curvefits for each chordwise position that calibration is performed

    # fill up the above lists from the data read from the input file 1 den chord sayisi kadar herbirine 1inci eleman 4uncu eleman gibi giditor
    # ilk elamni secmiyor o 7 zaten
    for inoChordPos in range(0, int(noChordPos)):
        xOc = ccf2[3*inoChordPos+1]
        curveFit_m = ccf2[3*inoChordPos+2]
        curveFit_n = ccf2[3*inoChordPos+3]
        print(curveFit_m, "Curvefit_m")
        print(curveFit_m, "Curvefit_n")
        xOcList.append(xOc)
        curveFit_mList.append(curveFit_m)
        curveFit_nList.append(curveFit_n)

    # Create the crop boundaries as read from the input file
    # SINEM THEREIS ERROR HERE 16 out of boundas por axcis 0 witth size 13 13 column var cunku elimdeki datada ama 13 desend e calsimiyor
    left = int(ccf2[-4])
    right = int(ccf2[-3])
    top = int(ccf2[-2])
    bottom = int(ccf2[-1])

    # the order of crop positions are reversed in the inputfile if it is the pressure side data (lower), so take care of that...
    if 'lower' in inputfile:
        temp = left
        left = right
        right = temp

    # Scan all images in the selected folder and analyze them one by one.
    counter = 0  # counter for analyzed number of images
    for filename in os.listdir(path):
        if filename.endswith('.tiff'):  # only works on tiff images
            counter += 1
            print(filename)
            imageSelect = os.path.join(path, filename)
            findTrans(imageSelect)

#########################################################################################


def findTrans(imageSelect):

    global min_value, max_value, min_value2, max_value2

    # Read input image
    img = cv2.imread(imageSelect)
    # s_img = cv2.imread('Suzlon_logo.png')

    # Convert to grayscale
    imgGray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Below lines maybe used if one would need to display the image for checking
    # cv2.imshow('test', imgGray1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Crop image using the crop boundaries read from the input file
    imgGray2 = imgGray1[top:bottom, left:right]

    # Enhance cropped image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(2, 2))
    imgGray = clahe.apply(imgGray2)
    # cv2.imshow('test', imgGray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # clahe_img1 = cv2.createCLAHE(clipLimit=1, tileGridSize=(2, 2)).apply(imgGray2)
    # clahe_img5 = cv2.createCLAHE(clipLimit=5, tileGridSize=(2, 2)).apply(imgGray2)
    # clahe_img10 = cv2.createCLAHE(clipLimit=10, tileGridSize=(2, 2)).apply(imgGray2)
    # clahe_img50 = cv2.createCLAHE(clipLimit=50, tileGridSize=(2, 2)).apply(imgGray2)
    # clahe_img100 = cv2.createCLAHE(clipLimit=100, tileGridSize=(2, 2)).apply(imgGray2)

    # cv2.imshow('clahe_img1', clahe_img1)
    # cv2.imshow('clahe_img5', clahe_img5)
    # cv2.imshow('clahe_img10', clahe_img10)
    # cv2.imshow('clahe_img50', clahe_img50)
    # cv2.imshow('clahe_img100', clahe_img100)
    # cv2.waitKey(30000)
    # cv2.destroyAllWindows()

    # Get row and column information from the image
    # Transition finder works on this cropped image
    rows, cols = imgGray.shape

    import operator  # needed later below for local maxima finding.

    # Range for scanning pixel columns
    cStart = 0
    cEnd = cols

    # Range for scanning pixel rows
    rowBegin = 0
    rowEnd = rows

    # Number of local pixel rows to be analyzed and assigned one transition location
    step = 3

    # Keep the original image as newImage_color. Later it will marked with transition positions.
    newImage_color = img

    # Parameters for spanwise average pixel location for transition and separation
    avelocTrx = []
    avelocSpx = []

    avelocTry = []
    avelocSpy = []

    # Use below if timing of the loop below is needed. This markes the start time.
    # t0 = time.clock()

    # Main loop that scans the pixel rows.
    # Scans every "step" pixel rows and assigns one value of
    # transition/separation location to that "step" number of pixel rows.
    for rr in range(rowBegin, rowEnd-step, step):
        # Parameters to hold the found transition/separation locations on the -to be scanned- "step" number of pixel rows
        ffTr = []
        ffSp = []

        # Start and end ranges for the "step" number of pixel rows
        rStart = rr
        rEnd = rr+step-1

        # Start scanning the "step" number of pixel rows
        for jj in range(rStart, rEnd):

            # f is the variation of pixel intensity along the columns of the currently scanned pixel row
            scanList_ss = imgGray[jj][cStart:cEnd]
            scanList_ps = scanList_ss[::-1]

            # Scan from left to right if it is upper (suction) side, else scan from right to left
            if 'upper' in inputfile:
                f = np.array(scanList_ss, dtype=np.float)
            if 'lower' in inputfile:
                f = np.array(scanList_ps, dtype=np.float)

            # MOVING AVERAGE PLOT
            # plt.plot(moving_average(f, n=10))
           # plt.show()

            # p is the moving average of the gradient of the moving average of f.
            # (i.e. first f is moving averaged. Then its gradient is found and then moving averaged)
            # This moving averaging is needed because the variations of f and p -if not averaged- are very patchy
            # 5 is the moving averaging range. Smaller values make it closer to actual visualized transition line.
            # However since it is less averaged, finding peaks becomes harder.
            p = moving_average(np.gradient(moving_average(f, n=5)), n=20)

            # Absolute value of p is needed to detect separation
            # Because the gradient is high negative if there is separation whereas it is high positive for transition.
            p = abs(p)

            p = p**5  # WHY WHY WHY WHY

            # plt.plot(p)
            # plt.show()

            # Find the pixel locations (indices) of the peaks in p within a local pixel neighborhood of 50
            indexes = find_peaks_cwt(p, np.arange(1, 50))

            # Find the actual value of p where it has peaks.
            pp = [p[i] for i in indexes]

            # Find the first maximum among the list of peaks, which can be the transition location depending on pixel position
            max1_index, max1_value = max(
                enumerate(pp), key=operator.itemgetter(1))

            # Set the max value to -999 in order to find the second maximum below.
            pp[max1_index] = -999

            # Find the second maximum among the list of peaks, which can be the separation location depending on pixel position
            max2_index, max2_value = max(
                enumerate(pp), key=operator.itemgetter(1))

            # Put the maximum 1 value back into its original position. Just to be neat :)
            pp[max1_index] = max1_value

            # print (indexes[max1_index],indexes[max2_index])
            # Assign the local transition/separation locations on the currently scanned pixel row
            # First check for the ratio of the first peak to second peak. When there is both transition and separation this ratio is high.
            # 0.5 is the current value of the criteria and it can be tuned depending on the image quality chenging the thresh01p and/or thresh01s parameters in the beginning of the code.
            if 'upper' in inputfile:
                # meaning there are two relatively strong peaks pointing to the existence of both transition and separation.
                # meaning if the first maximum occurs earlier on the pixel row, when scanned from left to right (i.e. SS).
                if indexes[max1_index] < indexes[max2_index]:
                    # Transition pixel on the cropped image + the left boundary pixel gives the actual pixel location on the original image
                    xTr = indexes[max1_index] + left
                else:
                    # transition still comes first when scanned from left to right on the suction side.
                    xTr = indexes[max2_index] + left

            if 'lower' in inputfile:
                # meaning there are two relatively strong peaks pointing to the existence of both transition and separation.
                # meaning if the first maximum occurs earlier on the pixel row, when scanned from right to left (i.e. PS).
                if indexes[max1_index] < indexes[max2_index]:
                    # Transition pixel on the cropped image is subtracted from the right boundary pixel gives the actual pixel location on the original image
                    xTr = -indexes[max1_index] + right
                else:
                    # transition still comes first when scanned from right to left on the pressure side.
                    xTr = -indexes[max2_index] + right

            """
            if 'upper' in inputfile:
                # meaning there are two relatively strong peaks pointing to the existence of both transition and separation.
                if max2_value/max1_value > thresh01s:
                    # meaning if the first maximum occurs earlier on the pixel row, when scanned from left to right (i.e. SS).
                    if indexes[max1_index] < indexes[max2_index]:
                        # Transition pixel on the cropped image + the left boundary pixel gives the actual pixel location on the original image
                        xTr = indexes[max1_index] + left
                        # Same as above but for separation
                        xSp = indexes[max2_index] + left
                    else:
                        # if the maximum level of transition is smaller than the maximum level of separation,
                        xSp = indexes[max1_index] + left
                        # transition still comes first when scanned from left to right on the suction side.
                        xTr = indexes[max2_index] + left
                else:
                    # if there are no two peaks but only one peak, it is marked as transition (for free transition cases)
                    xTr = indexes[max1_index] + left
                    xSp = 0  # and separation is marked as zero.
            if 'lower' in inputfile:
                # meaning there are two relatively strong peaks pointing to the existence of both transition and separation.
                if max2_value/max1_value > thresh01p:
                    # meaning if the first maximum occurs earlier on the pixel row, when scanned from right to left (i.e. PS).
                    if indexes[max1_index] < indexes[max2_index]:
                        # Transition pixel on the cropped image is subtracted from the right boundary pixel gives the actual pixel location on the original image
                        xTr = -indexes[max1_index] + right
                        xSp = -indexes[max2_index] + \
                            right  # Same for separation
                    else:
                        # if the maximum level of transition is smaller than the maximum level of separation,
                        xSp = -indexes[max1_index] + right
                        # transition still comes first when scanned from right to left on the pressure side.
                        xTr = -indexes[max2_index] + right
                else:
                    # if there are no two peaks but only one peak, it is marked as transition (for free transition cases)
                    xTr = -indexes[max1_index] + right
                    xSp = 0  # and separation is marked as zero.
            """

            # ffTr is a list that contains local transition locations along each pixel row within the currently scanned 'step' number of rows
            ffTr.append(xTr)
            """
            # ffSp is a list that contains local separation locations along each pixel row within the currently scanned 'step' number of rows
            ffSp.append(xSp)
            """
        # Generate the histogram of transition locations using the step number of pixel rows
        # and find the pixel location that has the maximum of the histogram
        # then assign this pixel location as the local transition location found for this row block that consists of step number of pixel rows
        nTr, bTr = np.histogram(ffTr, bins=100)
        elemTr = np.argmax(nTr)
        bin_maxTr = bTr[elemTr]
        localTr = int(bin_maxTr)  # local transition location

        # Generate the histogram of separation locations using the step number of pixel rows
        # and find the pixel location that has the maximum of the histogram
        # then assign this pixel location as the local separation location found for this row block that consists of step number of pixel rows
        """
        nSp, bSp = np.histogram(ffSp, bins=100)
        elemSp = np.argmax(nSp)
        bin_maxSp = bSp[elemSp]
        localSp = int(bin_maxSp)  # local separation location
        """
        # These two lists contain local transition locations in pixels at each
        # x (chordwise, or pixel columns) and y (spanwise, or pixel rows) location
        avelocTrx.append(localTr)
        avelocTry.append(rStart+top)

        # These two lists contain local separation locations in pixels at each
        # x (chordwise, or pixel columns) and y (spanwise, or pixel rows) location
        # Note that if there is no separation the location is not added to the list.
        """
        if localSp != 0:
            avelocSpx.append(localSp)
            avelocSpy.append(rStart+top)
        """
  # ROWLAR ICIN YAPTIRDIGI FOR LOOPTAN CIKTIK

 ########################## TRANSITION #####################################

    # Calculate the average and std dev of estimated spanwise transition locations
    Trx = np.array(avelocTrx, dtype=np.float)
    averageTrx = np.mean(Trx)
    stddevTrx = np.std(Trx)

    # Set the sigma value for validity check around average (transition)
    # bunu artirinsan daha scattered olur saga sola ama alttaki std threshoduna takilip hic dege rvermeyebilir onun icin
    sigmaTr = threshSigmaTr*stddevTrx
    # bunu artirinca stdthresholduna da artirdim bakalim nolcak

    # Calculate the histogram of spanwise distribution of the local transition locations
    # then find the maximum of the histogram
    # then assign the maximum value as the temporary spanwise average location of the transition location
    nTr1, bTr1 = np.histogram(avelocTrx, bins=100)
    elemTr1 = np.argmax(nTr1)
    bin_maxTr1 = bTr1[elemTr1]
    # spanwise average transition location (temporary)
    aveTr1 = int(bin_maxTr1)
    print("aveTr", aveTr1, "\n\n")

    # Now check how far each of the local transition locations away from the spanwise average
    avelocTrx2 = []
    avelocTry2 = []
    for k1 in range(0, len(avelocTrx)):
        # if the local transition location is more than sigma away then disregard that point. This provides some filtering of the data
        if abs(avelocTrx[k1]-aveTr1) < sigmaTr:
            avelocTrx2.append(avelocTrx[k1])
            avelocTry2.append(avelocTry[k1])

    # Calculate the histogram of filtered spanwise distribution of the local transition locations
    # then find the maximum of the histogram
    # then assign the maximum value as the actual spanwise average location of the transition location
    nTr2, bTr2 = np.histogram(avelocTrx2, bins=100)
    elemTr2 = np.argmax(nTr2)
    bin_maxTr2 = bTr2[elemTr2]
    aveTr = int(bin_maxTr2)  # spanwise average transition location (actual)
    Trx2 = np.array(avelocTrx2, dtype=np.float)
    stddevTrx2 = np.std(Trx2)

    ############################## SEPARATION  #################################################
    """
    # Calculate the average and std dev of estimated spanwise separation locations
    Spx = np.array(avelocSpx, dtype=np.float)
    averageSpx = np.mean(Spx)
    stddevSpx = np.std(Spx)

    # Set the sigma value for validity check around average (separation)
    sigmaSp = threshSigmaSp*stddevSpx

    # Calculate the histogram of spanwise distribution of the local separation locations
    # then find the maximum of the histogram
    # then assign the maximum value as the temporary spanwise average location of the separation location
    nSp1, bSp1 = np.histogram(avelocSpx, bins=100)
    elemSp1 = np.argmax(nSp1)
    bin_maxSp1 = bSp1[elemSp1]
    # spanwise average separation location (temporary)
    aveSp1 = int(bin_maxSp1)

    # Now check how far each of the local separation locations away from the spanwise average
    avelocSpx2 = []
    avelocSpy2 = []
    for k2 in range(0, len(avelocSpx)):
        # if the local separation location is more than sigma away then disregard that point. This provides some filtering of the data
        if abs(avelocSpx[k2]-aveSp1) < sigmaSp:
            avelocSpx2.append(avelocSpx[k2])
            avelocSpy2.append(avelocSpy[k2])

    # Calculate the histogram of filtered spanwise distribution of the local separation locations
    # then find the maximum of the histogram
    # then assign the maximum value as the actual spanwise average location of the separation location
    nSp2, bSp2 = np.histogram(avelocSpx2, bins=100)
    elemSp2 = np.argmax(nSp2)
    bin_maxSp2 = bSp2[elemSp2]
    aveSp = int(bin_maxSp2)  # spanwise average transition location (actual)

    Spx2 = np.array(avelocSpx2, dtype=np.float)
    stddevSpx2 = np.std(Spx2)
    """
    ############################# THRESHOLDS ##################################################

    if stddevTrx2 > threshStddevTr:
        aveTr = 0
    """
    if stddevSpx2 > threshStddevSp:  # <----- 3. cause
        print("cause 3: stddevSpx2 > threshStddevSp")
        print("stddevSpx2: ", stddevSpx2)
        print("threshStddevSp: ", threshStddevSp, "\n\n")
        aveSp = 0
    """

    # If number of filtered data points are less than a certain value probably there is no valid transition/separation location
    if len(avelocTrx2) < threshPixelTr:  # eleman sayisi
        aveTr = 0
    """
    if len(avelocSpx2) < threshPixelSp:  # <----- 2. cause
        print("cause 2: len(avelocSpx2) < threshPixelSp")
        print("len(avelocSpx2): ", len(avelocSpx2))
        print("threshPixelSp: ", threshPixelSp, "\n\n")
        aveSp = 0
    """
    # print (stddevTrx2)
    # print (stddevSpx2,threshStddevSp)

   ###########################################################################
    # Mark the local transition locations on the original but undistorted image
    if aveTr != 0:
        for k3 in range(0, len(avelocTrx2)):
            cv2.line(newImage_color,
                     (int(avelocTrx2[k3]), int(avelocTry2[k3])),
                     (int(avelocTrx2[k3]), int(avelocTry2[k3]+step)),
                     (0, 0, 255), 2)
            cv2.line(newImage_color,
                     (aveTr, int(cStart)),
                     (aveTr, int(cEnd)),
                     (255, 0, 0), 2)
    """
    # Mark the local separation locations on the original but undistorted image
    if aveSp != 0:
        for k4 in range(0, len(avelocSpx2)):
            cv2.line(newImage_color, (int(avelocSpx2[k4]), int(avelocSpy2[k4])), (int(
                avelocSpx2[k4]), int(avelocSpy2[k4]+step)), (0, 255, 0), 2)
    """
    # Following line can be used to track timing of the calculations up to this point
    # print (time.clock()-t0)

    # Find the AoA value by scanning the names of the files that are currently being analyzed.
    if filename[5] != '0':
        if filename[5] == 'p':
            AoA = filename[6]
            if filename[7] == 'p':
                AoA += '.%s' % (filename[8])
                UDflag = filename[10]
            if filename[7] != 'p':
                # AoA += filename[7]
                AoA += '%s' % (filename[7])
                UDflag = filename[9]
        elif filename[5] == 'm':
            AoA = '-%s' % (filename[6])
            if filename[7] == 'p':
                AoA += '.%s' % (filename[8])
                UDflag = filename[10]
            if filename[7] != 'p':
                AoA += filename[7]
                AoA += '.%s' % (filename[9])
                UDflag = filename[11]
    else:
        AoA = '0.0'
        UDflag = filename[9]

    AoA2 = float(AoA)
    print(AoA2, 'AoA2')
    pixelList = []

    # Create the pixel list corresponding to the chordwise positions from the read calibration data
    for iChordList in range(0, len(xOcList)):
        P = curveFit_mList[iChordList]*AoA2 + curveFit_nList[iChordList]
        pixelList.append(P)

    # Find the curvefit coefficients for the current angle of attack...pixel vs x/c
    curveFit6_m, curveFit6_n = np.polyfit(pixelList, xOcList, 1)

    # Using the calculated curvefit coefficients calculate the x/c location of the transition location
    if aveTr == 0:
        xTrChord = 0
    else:
        xTrChord = curveFit6_m*aveTr+curveFit6_n

    # Using the calculated curvefit coefficients calculate the x/c location of the separation location
    # only if there "is" separation
    """
    if aveSp == 0:
        xSpChord = 0
    else:
        xSpChord = curveFit6_m*aveSp+curveFit6_n
    """
    # Write the found spanwise average pixel and x/c locations for each image to a text data file. 
    
    fileout.write('%s \t %s \t %f \t %f \t %s \n' % (
        filename[0:4], AoA, aveTr, xTrChord, UDflag))

    """
    fileout.write('%s \t %s \t %f \t %f \t %f \t %f \t \t %s \n' % (
        filename[0:4], AoA, aveTr, xTrChord, aveSp, xSpChord, UDflag))
    """
    # Write the found pixel and x/c locations for each image on the images as well
    def print_angle_of_attack_label():
        cv2.putText(newImage_color, "AoA= %s" % (str(round(AoA2, 2))),
                    (380, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (61, 241, 228), 1)

    print_angle_of_attack_label()

    def print_transition_label():
        if xTrChord == 0:
            cv2.putText(newImage_color, "Transition not detected",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.putText(newImage_color, "Transition: %s%%" % (
                str(round(xTrChord, 2))), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    print_transition_label()

    def print_separation_label():
        if xSpChord == 0:
            cv2.putText(newImage_color, "Separation not detected",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(newImage_color, "Separation: %s%%" % (
                str(round(xSpChord, 2))), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    #print_separation_label()

    # Put Suzlon logo on the marked image
   # small = cv2.resize(s_img, (0,0), fx=0.1, fy=0.1)
    # x_offset=380
   # y_offset=600
   # newImage_color[y_offset:y_offset+small.shape[0], x_offset:x_offset+small.shape[1]] = small

    # cv2.putText(newImage_color, "TL",
    #             (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # cv2.putText(newImage_color, "TR",
    #             (right, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # cv2.putText(newImage_color, "BL",
    #             (left, bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # cv2.putText(newImage_color, "BR",
    #             (right, bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Save the marked images in "/tr" folder, also add "-tr" to the marked image name
    filename2 = filename.partition(".")[0]
    filename2 = filename2 + "-tr.tiff"
    path2 = path.replace('processed', 'tr')
    if not os.path.exists(path2):  # if the "tr" folder does not exist create it
        os.makedirs(path2)

    filename3 = os.path.join(path2, filename2)
    cv2.imwrite(filename3, newImage_color)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def Mbox(title, text, style):
    print(title, "\n", text, "\n", style, "\n\n")


if __name__ == "__main__":
    main()
