"""
------------------------------------------------------------------------------
Created on Sat Jul 11 20:45:07 2015

Find the region of interest in the sample image using a template from file.

@author: Charles (XiaRui) Zhang and Sudip Guha
-------------------------------------------------------------------------------
"""

import math

colorsRGB = {'red': (255,0,0), 'blue': (234, 182, 122), 'black': (255, 255, 255), 'brown': (99,142,197)}

def RGBColorMatch(r, g, b):
    """
    Returns a string that best describes the color of the RGB.
    """
    
    currDistance = math.pow(255, 3)
    currIndex = 0
    for key in colorsRGB.keys():
        red = colorsRGB[key][0]
        blue = colorsRGB[key][1]
        green = colorsRGB[key][2]
        temp = math.sqrt(math.pow(red - r, 2) + math.pow(blue - b, 2) + math.pow(green - g, 2))
        if (temp < currDistance):
            currDistance = temp
            currIndex = key
    return currIndex

def hueColorMatch(h):
    """
    Returns a string that best describes the color of the hue, h
    Reference Hue ranges from 0 to 359 inclusive.
    """
    # TODO: Find the actual hue values for the colors
    hue          =  [    0,       30,       60,     123,    240,      275,      0,       0,     38,        0,       0]
    indexToColor =  ["Red", "Orange", "Yellow", "Green", "Blue", "Violet", "Grey", "White", "Gold", "Silver", "Black"]

    # Normalize the HUE to vary from 0 to 179 instead.
    hue = [temp/2 for temp in hue]
    
    currDistance = math.pow(255, 3)
    currIndex = 0
    for i in range(len(hue)):
        temp = np.abs(hue[i] - h)
        if (temp < currDistance):
            currDistance = temp
            currIndex = i
    return indexToColor[currIndex]

def hsvColorMatch(h, s, v):
    """
    Returns a string that best describes the color of the HSV.
    """
    # Here, the HUE is assumed to vary from 0 to 359.
    hueLevels    =  [    0,       30,       60,     123,    240,      275,      0,       0,     38,        0,       0]
    satLevels    =  [  255,      255,      255,     193,    155,      104,      0,       0,    113,        0,       0]
    valLevels    =  [  255,      255,      255,     202,    255,      248,    140,     255,    208,      200,       0]
    indexToColor =  ["Red", "Orange", "Yellow", "Green", "Blue", "Violet", "Grey", "White", "Gold", "Silver", "Black"]
    
    # Normalize the HUE to vary from 0 to 179 instead.
    hueLevels = [h//2 for h in hueLevels]

    currDistance = math.pow(255, 3)
    currIndex = 0
    for i in range(len(hueLevels)):
        hue = hueLevels[i]
        sat = satLevels[i]
        val = valLevels[i]
        temp = math.sqrt(math.pow(hue - h, 2) + math.pow(sat - s, 2) + math.pow(val - v, 2))
        if (temp < currDistance):
            currDistance = temp
            currIndex = i
    return indexToColor[currIndex]

def normalizeHue(arr, currMax, futureMax):
    """
    Given a 2D array (arr), and the current maximum (currMax) value the
    array can have, this function normalizes all the values so that the
    maximum value that the entries can have is now futureMax.
    """
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            arr[i][j] = (arr[i][j] / currMax) * futureMax
    return arr

"""
TODO: make an estimate for all expected colors in LAB scale
"""




def labColorMatch():
    """
    Given LAB coords, match to nearest color

    Apparently LAB is linear and good for 3rd dist 
    """
    