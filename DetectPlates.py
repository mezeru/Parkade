#DetectPlates.py
import cv2
import numpy as np
import math
import Main
import random

import Preprocess
import DetectChars
import PossiblePlate
import PossibleChar

PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5

def detectPlatesInScene(imgOriginalScene):
    listOfPossiblePlates = []                   
    height, width, numChannels = imgOriginalScene.shape
    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)
    cv2.destroyAllWindows()
    #Preprocess to get grayscale and threshold images
    imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(imgOriginalScene)         
    #Find all possible characters in the image
    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene)
    #Given a list of all possible characters, find groups of matching characters
    listOfListsOfMatchingCharsInScene = DetectChars.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)
    #For each group of matching chars
    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:  
        #Attempt to extract plate
        possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)        
        #If plate was found
        if possiblePlate.imgPlate is not None:  
            #Add to list of possible plates
            listOfPossiblePlates.append(possiblePlate)                  
    return listOfPossiblePlates

def findPossibleCharsInScene(imgThresh):
    listOfPossibleChars = []               
    intCountOfPossibleChars = 0
    imgThreshCopy = imgThresh.copy()
    #To find all contours
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   
    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)
    for i in range(0, len(contours)):                       
        possibleChar = PossibleChar.PossibleChar(contours[i])
        #If contour is a possible character
        if DetectChars.checkIfPossibleChar(possibleChar):      
            #Increment count of possible characters
            intCountOfPossibleChars = intCountOfPossibleChars + 1
            #Add to list of possible characters
            listOfPossibleChars.append(possibleChar)                       
    return listOfPossibleChars

def extractPlate(imgOriginal, listOfMatchingChars):
    possiblePlate = PossiblePlate.PossiblePlate()           
    #Sort characters from left to right based on x position
    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)       
    #Calculate the center point of the plate
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0
    ptPlateCenter = fltPlateCenterX, fltPlateCenterY
    #Calculate plate width and height
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)
    intTotalOfCharHeights = 0
    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)
    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)
    #Calculate correction angle of plate region
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = DetectChars.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)
    #Pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )
    #Get the rotation matrix for our calculated correction angle
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)
    height, width, numChannels = imgOriginal.shape      
    #Rotate the entire image
    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))       
    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))
    #Copy the cropped plate image into the applicable member variable of the possible plate
    possiblePlate.imgPlate = imgCropped         
    return possiblePlate