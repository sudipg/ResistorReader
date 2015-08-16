# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------
Created on Sat Jul 11 20:45:07 2015

Find the region of interest in the sample image using a template from file.

@author: sudipguha and Charles (XiaRui) Zhang
-------------------------------------------------------------------------------
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

def main():
    """
    ---------------------------------------------------------------------------

    TODO: Consider using command line arguments for image names?
    Something along the lines of sys.argv?

    ---------------------------------------------------------------------------
    """

    template = cv2.imread('images/rs9.png',0)
    template = cv2.blur(template, (2,2))
    # template2 = cv2.imread('images/T.png',0)

    img = cv2.imread('images/test_res.png',0)

    # img_blurred = cv2.blur(img, (1,1))
    # template = cv2.blur(template,(1,1))
    # # create ORB object for detecting features
    # orb = cv2.ORB_create()
    # kpt, dest = orb.detectAndCompute(template, None)
    # kpt2, dest2 = orb.detectAndCompute(template2, None)
    # kpi, desi = orb.detectAndCompute(img_blurred, None)

    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # matches = bf.match(dest, desi)
    # matches = sorted(matches, key = lambda x:x.distance)
    # img3 = img
    # img3 = cv2.drawMatches(template,kpt,img,kpi,matches, img3,flags=2)

    # matches2 = bf.match(dest, desi)
    # matches2 = sorted(matches2, key = lambda x:x.distance)
    # matches = matches+matches2
    # # now we need to extract the location of the matched features
    # matched_keypoints = set()

    # for match in matches:
    #     matched_keypoints.add(kpi[match.trainIdx].pt)

    # # set cointains points to make line of BF through
    # print matched_keypoints
    # matched_keypoints = list(matched_keypoints)
    # matched_keypoints_x = [match[0] for match in matched_keypoints]
    # matched_keypoints_y = [match[1] for match in matched_keypoints]

    #line_of_BF = np.polyfit(matched_keypoints_x, matched_keypoints_y, 1)

    # now draw them on top of the image
    matchedKeypointsX, matchedKeypointsY, img3 = findMatches(template, img)

    # plt.clf()
    # plt.imshow(matchesImg)
    # plt.show()

    #x_range = [sorted(matched_keypoints_x)[0]+x for x in range(int(sorted(matched_keypoints_x)[len(matched_keypoints_x)-1] - sorted(matched_keypoints_x)[0]))]

    #y_range = [line_of_BF[1] + line_of_BF[0]*x for x in x_range]

    plt.clf()
    plt.imshow(img, cmap = 'gray')
    plt.plot(matchedKeypointsX, matchedKeypointsY, 'ro')
    #plt.plot(x_range, y_range, 'ro')
    lowerBoundX, upperBoundX, lowerBoundY, upperBoundY = findBoxAroundNthPercentile(matchedKeypointsX, matchedKeypointsY, 0.50, 40)
    #boxX = [lowerBoundX, lowerBoundX, upperBoundX, upperBoundX]
    #boxY = [lowerBoundY, upperBoundY, lowerBoundY, upperBoundY]
    #plt.plot(boxX, boxY, 'go')
    plt.show()
    
    ROI = img [lowerBoundY:upperBoundY, lowerBoundX:upperBoundX]
    plt.clf()
    plt.imshow(ROI, cmap = 'gray')
    plt.show()

    # band = cv2.imread('images/Band.png', 0)
    # plotMatches(band, ROI)
    bands = cv2.imread('images/Bands.png', 0)
    bands = cv2.blur(bands, (2,2))
    plotMatches(bands, ROI)

    # dft = discreteFourierTransform(ROI)
    # plt.clf()
    # plt.imshow(dft, cmap = 'gray')
    # plt.show()

def findMatches(template, img):
    """
    ---------------------------------------------------------------------------
    Finds keypoints that matches the TEMPLATE in IMG.
    
    Assumes template and img are 2D arrays (i.e. Pictures)
    
    RETURNS the coordinates of all matched keypoints in IMG.

    The return format is: xCoordinates, yCoordinates.
    ---------------------------------------------------------------------------
    """

    orb = cv2.ORB_create()
    kpTemplate, desTemplate = orb.detectAndCompute(template, None)
    kpImg, desImg = orb.detectAndCompute(img, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(desTemplate, desImg)
    matches = sorted(matches, key = lambda x:x.distance)

    matchedKeypoints = set()

    img3 = img
    img3 = cv2.drawMatches(template,kpTemplate,img,kpImg,matches[:15],img3,flags=2)

    for match in matches:
        matchedKeypoints.add(kpImg[match.trainIdx].pt)

    matchedKeypoints = list(matchedKeypoints)
    matchedKeypointsX = [match[0] for match in matchedKeypoints]
    matchedKeypointsY = [match[1] for match in matchedKeypoints]

    return matchedKeypointsX, matchedKeypointsY, img3
    
def plotMatches(template, img):
    x, y, img3 = findMatches(template, img)

    plt.clf()
    plt.imshow(img3)
    plt.show()

def findLowerAndUpperPercentile(arr, lowerPercentile, upperPercentile):
    """
    ---------------------------------------------------------------------------

    TODO: Algorithmitically, this may not be ideal.
    Currently, it takes O(n log(n)) time.
    Maybe use better algorithm? For now, shouldn't be too bad?

    Given an array ARR, it returns 2 values in the array. 

    The first value's percentile in the ARR is = lowerPercentile.

    The second value's percentile in the ARR is = upperPercentile

    ---------------------------------------------------------------------------
    """

    if (lowerPercentile > 1 or upperPercentile > 1 or lowerPercentile < 0 or upperPercentile < 0):
        print("findLowerAndUpperPercentile Error: One (or more) percentile is (are) invalid.\n")
        print("lowerPercentile: " + str(lowerPercentile) + "upperPercentile: " + str(upperPercentile))
    else:
        temp = sorted(arr, key = lambda x:x)
        return temp[int(len(temp) * lowerPercentile)], temp[int(len(temp) * upperPercentile)]

def findBoxAroundNthPercentile(keypointsX, keypointsY, percentile, border):
    """
    ---------------------------------------------------------------------------

    Given the X and Y coordinates of all keypoints, this returns 4 values.

    These 4 values, represent the lower and upper bound on X and Y coordinates 
    such that PERCENTILE percent of all key points are inside that box.

    Also adds some padding to the box in the form of BORDER

    ---------------------------------------------------------------------------
    """
    
    if (percentile < 0 or percentile > 1):
        print("findBoxAroundNthPercentile Error: Percentile not valid.\n")
        print("Percentile is: " + str(percentile))
    else:
        lowerPercentile = 0.5 - percentile / 2
        upperPercentile = 0.5 + percentile / 2
        lowerBoundX, upperBoundX = findLowerAndUpperPercentile(keypointsX, lowerPercentile, upperPercentile)
        lowerBoundY, upperBoundY = findLowerAndUpperPercentile(keypointsY, lowerPercentile, upperPercentile)
        #TODO: Perform bound checking before returning
        return lowerBoundX - border, upperBoundX + border, lowerBoundY - border, upperBoundY + border    

    # medianX = median(keypointsX)
    # medianY = median(keypointsY)
    # distanceX = computeOneDimensionalDistance(keypointsX, medianX)
    # distanceY = computeOneDimensionalDistance(keypointsY, medianY)
    # lowerBoundX = findNthPercentile(distanceX, 1 - percentile)
    # upperBoundX = findNthPercentile(distanceX, percentile)
    # lowerBoundY = findNthPercentile(distanceY, 1 - percentile)
    # upperBoundY = findNthPercentile(distanceY, percentile)

def discreteFourierTransform(img):
    """
    ---------------------------------------------------------------------------
    
    Given an image in the form of IMG, returns the DFT "img"

    ---------------------------------------------------------------------------
    """
    dft = np.fft.fft2(img)
    dft = np.fft.fftshift(dft)
    dft  = 20 * np.log(np.abs(dft))
    return dft


if __name__ == '__main__':
    main()

