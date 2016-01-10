import colorsys
import math

rSamples        = [    67,     22,      142,      117,       72,       20,   128,    66,      125,       71,      29,       5]
gSamples        = [    88,     40,      111,       97,       59,        7,    30,     4,       38,       16,      55,      21]
bSamples        = [   107,     62,       31,       26,       89,       24,    31,     7,       19,        9,      44,      11]
sampleNames     = ["Blue", "Blue", "Yellow", "Yellow", "Violet", "Violet", "Red", "Red", "Orange", "Orange", "Green", "Green"]

redLevels       =  [  255,      255,      255,       0,      0,      149,    140,     255,    208,      200,       0]
greenLevels     =  [    0,       42,      191,     255,      0,        0,    140,     255,    149,      200,       0]
blueLevels      =  [    0,        0,        0,       0,    255,      255,    140,     255,     50,      200,       0]
indexToColor    =  ["Red", "Orange", "Yellow", "Green", "Blue", "Violet", "Grey", "White", "Gold", "Silver", "Black"]

hue             =  [  360,      0,       10,       45,     123,    240,         275,         0,          0,        30,              0,          0]
hueIndexToColor =  ["Red",  "Red", "Orange", "Yellow", "Green", "Blue", "Violet", "Grey", "White", "Gold", "Silver", "Black"]


def colorRatio(r, g, b):
    total = r + g + b + 0.0
    if (total < 1.0):
        total = 1.0 
    r = r/total
    g = g/total
    b = b/total
    return r, g, b

rRedLevels    = []
rGreenLevels  = []
rBlueLevels   = []
rIndexToColor = []

# Create the ratio reference values
for i in range(len(indexToColor)):
    r = redLevels[i]
    g = greenLevels[i]
    b = blueLevels[i]
    r, g, b = colorRatio(redLevels[i], greenLevels[i], blueLevels[i])
    rRedLevels.append(r)
    rGreenLevels.append(g)
    rBlueLevels.append(b)
    rIndexToColor.append(indexToColor[i])

# Hand calculated values. Left here just in case.
# rRedLevels    =  [1.000,   0.8586,   0.5717,   0.000,  0.000,   0.3688, 0.3333, 0.5111]
# rGreenLevels  =  [0.000,   0.1414,   0.4283,   1.000,  0.000,   0.0000, 0.3333, 0.3661]
# rBlueLevels   =  [0.000,   0.0000,   0.0000,   0.000,  1.000,   0.6312, 0.3333, 0.1228]
# rIndexToColor =  ["Red", "Orange", "Yellow", "Green", "Blue", "Violet", "Mono", "Gold"]


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

def normalizedColorMatch(r, g, b):
    nr, ng, nb = normalizeSaturationAndValue(r, g, b)
    return RGBColorMatch(nr, ng, nb)

def ratioColorMatch(r, g, b):
    r, g, b = colorRatio(r, g, b)

    currDistance = math.pow(255, 3)
    currIndex = 0
    for i in range(len(rRedLevels)):
        red = rRedLevels[i]
        blue = rBlueLevels[i]
        green = rGreenLevels[i]
        temp = abs(red - r) + abs(blue - b) + abs(green - g)
        if (temp < currDistance):
            currDistance = temp
            currIndex = i
    return rIndexToColor[currIndex]

def name(rSamples, gSamples, bSamples, sampleNames):
    for i in range(len(sampleNames)):
        r = rSamples[i]
        g = gSamples[i]
        b = bSamples[i]
        h, s, v = convertRGBtoHSV(r, g, b)
        nr, ng, nb = normalizeSaturationAndValue(r, g, b)
        rr, rg, rb = colorRatio(r, g, b)
        print("Sample Color: " + sampleNames[i])
        #print("Sample Normalized RGB: " + str(nr) + ", " + str(ng) + ", " + str(nb))
        print("Sample Ratio RGB: " + str(rr) + ", " + str(rg) + ", " + str(rb))
        print("Sample Hue: " + str(h))
        print("Classified Color Based on Hue: " + hueColorMatch(h))
        print("Classified Color Based on Regular RGB: " + RGBColorMatch(r, g, b))
        print("Classified Color Based on Normalized RGB: " + RGBColorMatch(nr, ng, nb))
        print("Classified Color Based on Ratio RGB: " + ratioColorMatch(r, g, b))
        print("Classified Color Based on Normalized Ratio RGB: " + ratioColorMatch(nr, ng, nb) + "\n")

name(rSamples, gSamples, bSamples, sampleNames)





