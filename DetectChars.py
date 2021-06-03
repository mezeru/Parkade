#DetectChars.py
import os
import cv2
import numpy as np
import math
import random

import Main
import Preprocess
import PossibleChar
import Fare

#Creating a model
kNearest = cv2.ml.KNearest_create()

#Constants for checkIfPossibleChar function
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0
MIN_PIXEL_AREA = 80

#Constants for comparing two chars
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0
MAX_CHANGE_IN_AREA = 0.5
MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2
MAX_ANGLE_BETWEEN_CHARS = 12.0

#Other constants
MIN_NUMBER_OF_MATCHING_CHARS = 3
RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30
MIN_CONTOUR_AREA = 100

def loadKNNDataAndTrainKNN():
    allContoursWithData = []               
    validContoursWithData = []              
    try:
        #Read in training classifications
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                  
    except:                                                                                
        print("Error, unable to open classifications.txt, exiting program\n")  
        return False                                                                       
    try:
        #Read in training images
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 
    except:                                                                                
        print("Error, unable to open flattened_images.txt, exiting program\n")  
        return False  
    #Reshape numpy array to 1d
    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       
    #Set default K to 1 (Nearest Neighbour)
    kNearest.setDefaultK(1)  
    #Train KNN object
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)           
    #If training was successful 
    return True                             

def detectCharsInPlates(listOfPossiblePlates):
    intPlateCounter = 0
    imgContours = None
    contours = []
    if len(listOfPossiblePlates) == 0:          
        return listOfPossiblePlates             
    for possiblePlate in listOfPossiblePlates:          
        #Preprocess to get grayscale and threshold images
        possiblePlate.imgGrayscale, possiblePlate.imgThresh = Preprocess.preprocess(possiblePlate.imgPlate)     
        #Increase size of plate image for easier viewing and character detection
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6)
        #Threshold again to eliminate any gray areas
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #Find all possible chars in the plate, this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)
        #Given a list of all possible chars, find groups of matching chars within the plate
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)
        #If no groups of matching chars were found in the plate
        if (len(listOfListsOfMatchingCharsInPlate) == 0):
            possiblePlate.strChars = ""
            continue
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):                          
            #Sort chars from left to right
            listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX) 
            #Remove inner overlapping chars
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])              

        #Within each possible plate, suppose the longest list of potential matching chars is the actual list of chars
        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0
        #Loop through all the vectors of matching chars, get the index of the one with the most chars
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i
        #Suppose that the longest list of matching chars within the plate is the actual list of chars
        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]
        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate)
    return listOfPossiblePlates

def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []                       
    contours = []
    imgThreshCopy = imgThresh.copy()
    #Finding all contours in plate
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:                        
        possibleChar = PossibleChar.PossibleChar(contour)
        #If contour is a possible character
        if checkIfPossibleChar(possibleChar):  
            #Add to list of possible characters
            listOfPossibleChars.append(possibleChar)      
    return listOfPossibleChars

def checkIfPossibleChar(possibleChar):
    #This function does a rough check on a contour to see if it could be a character
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and 
        possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and 
        possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False
    
def findListOfListsOfMatchingChars(listOfPossibleChars):
    #This function is used to re-arrange the one big list of characters into a list of lists of matching characters
    listOfListsOfMatchingChars = []                  
    for possibleChar in listOfPossibleChars:  
        #To find all characters in the big list that match the current char
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)        
        #Add the current char to current possible list of matching chars
        listOfMatchingChars.append(possibleChar)                
        #If current possible list of matching characters is not long enough to constitute a possible plate
        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:     
            continue                            
        #Add to our list of lists of matching characters
        listOfListsOfMatchingChars.append(listOfMatchingChars)      
        listOfPossibleCharsWithCurrentMatchesRemoved = []
        #To remove the current list of matching characters from the big list so we don't use those same characters twice 
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))
        #Recursive call
        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)      
        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars: 
            #Add to our original list of lists of matching chars
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)             
        break      
    return listOfListsOfMatchingChars

def findListOfMatchingChars(possibleChar, listOfChars):
    #To find all chars in the big list that are a match for the single possible character, and return those matching characters as a list
    listOfMatchingChars = []              
    for possibleMatchingChar in listOfChars: 
        #If the character we attempting to find matches for is the exact same character as the character in the big list we are currently checking
        if possibleMatchingChar == possibleChar:    
            #Then we do not include it in the list of matches b/c that would end up double including the current char  
            continue                               
        
        #To compute parameters to check if characters are a match
        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)
        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)
        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)
        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)
        #Checking if characaters match
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):
            #If the characters are a match, add the current character to list of matching characters
            listOfMatchingChars.append(possibleMatchingChar)        
    return listOfMatchingChars                  

#A function that uses Pythagorean theorem to calculate distance between two characters
def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)
    return math.sqrt((intX ** 2) + (intY ** 2))

#A function that use basic trigonometry to calculate angle between characters
def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))
    #To check to make sure we do not divide by zero if the center X positions are equal
    if fltAdj != 0.0:      
        #If adjacent is not zero, calculate angle
        fltAngleInRad = math.atan(fltOpp / fltAdj)      
    else:
        #If adjacent is zero
        fltAngleInRad = 1.5708                          
    #Calculating angle in degrees
    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)       
    return fltAngleInDeg

#If we have two characters overlapping or to close to each other to possibly be separate characters, we remove the inner (smaller) char,to prevent including the same character twice if two contours are found for the same character
def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)                
    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            #If current character and other character are not the same character
            if currentChar != otherChar:       
                #If current character and other character have center points at almost the same location
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    #Overlapping charcaters are obtained. Finding smaller character
                    #If current character is smaller than other character
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:
                        #If current character was not already removed
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved: 
                            #Remove current character
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)         
                    #Else, if other char is smaller than current char    
                    else:  
                        #If other char was not already removed 
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:    
                            #Remove other character
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)          
    return listOfMatchingCharsWithInnerCharRemoved

#This function is used for character recognition
def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ""               
    height, width = imgThresh.shape
    imgThreshColor = np.zeros((height, width, 3), np.uint8)
    #Sort chars from left to right
    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        
    #To make color version of threshold image so we can draw contours in color on it
    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)                     
    for currentChar in listOfMatchingChars:                                        
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))
        #To draw green box around the character
        cv2.rectangle(imgThreshColor, pt1, pt2, Main.SCALAR_GREEN, 2)          
        #Crop character out of threshold image
        imgROI = imgThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]
        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))           
        #Flatten image into 1d numpy array
        npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))        
        #Convert from 1d numpy array of ints to 1d numpy array of floats
        npaROIResized = np.float32(npaROIResized)               
            #Using Nearest Neighbour algorithm
        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)              
        #Getting character from results
        strCurrentChar = str(chr(int(npaResults[0][0])))            
        #Appending current character to full string
        strChars = strChars + strCurrentChar                        
    #Calling the function to calculate cost of parking for each car
    lot1 = Fare.cost(strChars)
    return strChars
    return lot1