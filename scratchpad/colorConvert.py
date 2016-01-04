import colorsys
import math

rSamples    = [67,     22,     142,      117,      72,       20,       128,   66,    125,      71,       29,      5]
gSamples    = [88,     40,     111,      97,       59,       7,        30,    4,     38,       16,       55,      21]
bSamples    = [107,    62,     31,       26,       89,       24,       31,    7,     19,       9,        44,      11]
sampleNames = ["Blue", "Blue", "Yellow", "Yellow", "Violet", "Violet", "Red", "Red", "Orange", "Orange", "Green", "Green"]

redLevels    =  [255, 255, 255, 0,   0,   205, 140, 255, 208, 200, 0]
greenLevels  =  [0,   128, 255, 255, 0,   146, 140, 255, 149, 200, 0]
blueLevels   =  [0,   0,   0,   0,   255, 248, 140, 255, 50,  200, 0]
indexToColor =  ["Red", "Orange", "Yellow", "Green", "Blue", "Violet", "Grey", "White", "Gold", "Silver", "Black"]



hue          =  [360,    0,       10,       60,     123,    240,      275,      0,       0,     38,        0,       0]
indexToColor =  ["Red",  "Red", "Orange", "Yellow", "Green", "Blue", "Violet", "Grey", "White", "Gold", "Silver", "Black"]


def convertRGBtoHSV(r, g, b):
	"""Expects r, g, b to be between 0 and 255. """
	h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
	hue = int(360 * h)
	saturation = int(255 * s)
	value = int(255 * v)
	return hue, saturation, value

def normalizeSaturationAndValue(r, g, b):
	""" Expects r, g, b to be between 0 and 255. """
	h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
	s = 1
	v = 1
	red, green, blue = colorsys.hsv_to_rgb(h, s, v)
	red   = 255 * red
	blue  = 255 * blue
	green = 255 * green
	return red, blue, green

def hueColorMatch(h):
    """
    Returns a string that best describes the color of the hue, h
    Reference Hue ranges from 0 to 359 inclusive.
    """
    # TODO: Find the actual hue values for the colors
    
    # Normalize the HUE to vary from 0 to 179 instead.
    # hue = [temp/2 for temp in hue]
    
    currDistance = math.pow(360, 3)
    currIndex = 0
    for i in range(len(hue)):
        temp = abs(hue[i] - h)
        if (temp < currDistance):
            currDistance = temp
            currIndex = i
    return indexToColor[currIndex]

def RGBColorMatch(r, g, b):
    """
    Returns a string that best describes the color of the RGB.
    """  
    currDistance = math.pow(255, 3)
    currIndex = 0
    for i in range(len(redLevels)):
        red = redLevels[i]
        blue = blueLevels[i]
        green = greenLevels[i]
        temp = math.sqrt(math.pow(red - r, 2) + math.pow(blue - b, 2) + math.pow(green - g, 2))
        if (temp < currDistance):
            currDistance = temp
            currIndex = i
    return indexToColor[currIndex]

def name(rSamples, gSamples, bSamples, sampleNames):
	for i in range(len(sampleNames)):
		r = rSamples[i]
		g = gSamples[i]
		b = bSamples[i]
		h, s, v = convertRGBtoHSV(r, g, b)
		nr, ng, nb = normalizeSaturationAndValue(r, g, b)
		print("Sample Color: " + sampleNames[i])
		print("Sample Normalized RGB: " + str(nr) + ", " + str(ng) + ", " + str(nb))
		print("Sample Hue: " + str(h))
		print("Classified Color Based on Hue: " + hueColorMatch(h))
		print("Classified Color Based on Regular RGB: " + RGBColorMatch(r, g, b))
		print("Classified Color Based on Normalized RGB: " + RGBColorMatch(nr, ng, nb) + "\n")

name(rSamples, gSamples, bSamples, sampleNames)





