from datetime import datetime
import math
import time

import DetectChars
global ParkingLot

#Rate of car parking per minute
Rate = 50

def cost(strChars):
    ParkingLot = {}
    #Time of entry
    ParkingLot[strChars] = datetime.now()
    print(f"Car {strChars} has entered the parking lot at {ParkingLot[strChars]}")
    #For testing purposes ONLY
    time.sleep(60)
    #Check if car with license number 'strChars' is already in the parkking lot or not
    #If car was already in the parking lot
    if strChars in ParkingLot:
        TimeWaited =  datetime.now() - ParkingLot[strChars]
        #Converting time to minutes
        TimeWaited = (((TimeWaited.total_seconds())/60)) 
        #Finding the final amount to be paid (per minute)
        Amount = Rate * TimeWaited
        print(f"The Car {strChars} has exited the parking lot after {TimeWaited} minutes")
        print("Fare is %.2f Rs"%Amount)
        #Remove the car's details from dictionary
        ParkingLot.pop(strChars)
        