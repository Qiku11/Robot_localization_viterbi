# Shuhao Duan a1807323 Artificial Intellegence Assignment 3

import numpy as np
import sys

# implementation of the viterbi algorithm on the localization of a robot

# First we parse imput

fileName = sys.argv[1]
inputfile = open(fileName, "r")
mapDimensions = [0,0]
mapDimensions[0],mapDimensions[1] = map(int, inputfile.readline().split())

#print("Map dimensions: ", mapDimensions)

# Then we parse the map
map = []

for i in range(mapDimensions[0]):
    map.append(inputfile.readline().split())

# print("Map: ")
# for i in range(mapDimensions[0]):
#     print(map[i])

# then the observations
numObservations = int(inputfile.readline())

#print("Number of observations: ", numObservations)

# I want to store the observations as binary ints so checking them would be easy with bitwise ands
observations = []
for i in range(numObservations):
    observations.append(int(inputfile.readline(), 2))

# print("Observations: ")
# # I need to print out the numbers in binary form
# for i in range(numObservations):
#     print(bin(observations[i]))

# lastly, the error rate
errorRate = float(inputfile.readline())

# print("Error rate: ", errorRate)

# now parsing of all input is done. time to prepare data for viterbi to operate on

# we need the following things to start viterbi:
# 1. initial probabilities
# 2. transition matrix
# 3. emission matrix

# initial probabilities, we count the number of possible locations in the map and divide 1 by the total number of possible locations
possibleLocationsNum = 0
possibleLocations = []
for i in range(mapDimensions[0]):
    for j in range(mapDimensions[1]):
        if map[i][j] == "0":
            possibleLocationsNum += 1
            possibleLocations.append((i,j))

initialProbabilities = np.full((1, possibleLocationsNum), 1/possibleLocationsNum)

# print("Initial probabilities: ")
# print(initialProbabilities)

# transition matrix, this is a matrix of size possibleLocationsNum x possibleLocationsNum, where each cell represents
# the probability of transitioning from row'th location to column'th location

# we need to find the possible transitions from each location, and then divide 1 by the number of possible transitions
# to get the probability of each transition

# first we set up an arry to store the possible transitions from each location
possibleTransitionsForEachPossibleLocation = []

# we can create a helper array to store the correct observation for each possible location in binary form
correctObservations = []

# then we loop through the map
for i in range(mapDimensions[0]):
    for j in range(mapDimensions[1]):
        if map[i][j] == "0":
            # for every location that's traversible, we count the number of neighbours that are also traversible
            # and append the number to possibleTransitionsForEachPossibleLocation
            binaryObservation = 15
            possibleTransitions = 0
            if i-1 >= 0 and map[i-1][j] == "0":
                # N Neighbour
                possibleTransitions += 1
                binaryObservation -= 8
            if i+1 < mapDimensions[0] and map[i+1][j] == "0":
                # S Neighbour
                possibleTransitions += 1
                binaryObservation -= 4
            if j-1 >= 0 and map[i][j-1] == "0":
                # W Neighbour
                possibleTransitions += 1
                binaryObservation -= 2
            if j+1 < mapDimensions[1] and map[i][j+1] == "0":
                # E Neighbour
                possibleTransitions += 1
                binaryObservation -= 1
            possibleTransitionsForEachPossibleLocation.append(possibleTransitions)
            correctObservations.append(binaryObservation)

# next we set up the transition matrix
transitionMatrix = np.zeros((possibleLocationsNum, possibleLocationsNum))

# we loop through each possible location
# for every possible location, we loop through all other possible locations
# if the other possible location is a neighbour of the current possible location, we set its corresponding cell in the transition matrix to 1/possibleTransitions

for i in range(possibleLocationsNum):
    for j in range(possibleLocationsNum):
        if i != j:
            # check if the other location is a neighbour
            if abs(possibleLocations[i][0] - possibleLocations[j][0]) + abs(possibleLocations[i][1] - possibleLocations[j][1]) == 1:
                transitionMatrix[i][j] = 1/possibleTransitionsForEachPossibleLocation[i]

# # print out the transition matrix
# for i in range(possibleLocationsNum):
#     print(transitionMatrix[i])

# emission matrix, this is a matrix of size possibleLocationsNum x numObservations, where each cell represents
# the probability of observing the observation at column'th time step given that the robot is at row'th location

# set up the emission matrix
emissionMatrix = np.zeros((possibleLocationsNum, numObservations))

# to fill up the emission matrix, we loop through each possible location
# for each possible location, we loop through each observation
# for each observation, we check if the observation matches the possible location, and use the error rate to calculate the probability of observing the observation given that the robot is at the possible location



for i in range(possibleLocationsNum):
    for j in range(numObservations):
        # use bitwise and to check every bit of the observation whether it matches the possible location
        # we count the number of bits that don't match
        # then we use the error rate to calculate the probability of observing the observation given that the robot is at the possible location
        # the probability is 1 - errorRate ^ number of bits that match * errorRate ^ number of bits that don't match
        # we use bitwise xor to count the number of bits that don't match
        numberOfBitsThatDontMatch = bin(observations[j] ^ correctObservations[i]).count("1")
        #print("Observation: ", bin(observations[j])[2:].zfill(4), "Correct observation: ", bin(correctObservations[i])[2:].zfill(4), "Number of bits that don't match: ", numberOfBitsThatDontMatch)
        emissionMatrix[i][j] = ((1 - errorRate) ** (4 - numberOfBitsThatDontMatch)) * (errorRate ** numberOfBitsThatDontMatch)

# # # print out the emission matrix
# for i in range(possibleLocationsNum):
#     print(emissionMatrix[i])

# # print out the correct observations, fill up to four bits
# for i in range(possibleLocationsNum):
#     print(bin(correctObservations[i])[2:].zfill(4))

# now we have all the data we need to run viterbi
# we need to run viterbi for each observation
# we save the entire trellis matrix of size possibleLocationsNum x numObservations

trellisMatrix = np.zeros((possibleLocationsNum, numObservations))

# initialize the trellis matrix to time step 0 using initial probabilities and emission matrix
for i in range(possibleLocationsNum):
    trellisMatrix[i][0] = initialProbabilities[0][i] * emissionMatrix[i][0]

# next we loop through each time step
# for each time step, we loop through each possible location
# for each possible location, we loop through each other possible location
# we calculate the probability of the robot being at the other possible location at the previous time step
# we multiply that probability by the probability of transitioning from the other possible location to the current possible location
# we multiply that by the probability of observing the observation at the current time step given that the robot is at the current possible location
# we save the maximum probability in the trellis matrix

for i in range(1, numObservations):
    for j in range(possibleLocationsNum):
        for k in range(possibleLocationsNum):
            trellisMatrix[j][i] = max(trellisMatrix[j][i], trellisMatrix[k][i-1] * transitionMatrix[k][j] * emissionMatrix[j][i])

# now we have the trellis matrix, time to reflect all the probability values back to the map
# this time our map will be consisted of probabilities and 0s

# first we set up the map
output4MapsOfDifferentTimeStates = []

# we loop through each time step
# in every time step, we loop through each possible location
# for each possible location, write its corresponding probability from the trellis maxtrix to the map

for i in range(numObservations):
    mapInCurrentTimeStep = np.zeros((mapDimensions[0], mapDimensions[1]))
    for j in range(possibleLocationsNum):
        mapInCurrentTimeStep[possibleLocations[j][0]][possibleLocations[j][1]] = trellisMatrix[j][i]
    output4MapsOfDifferentTimeStates.append(mapInCurrentTimeStep)

output4MapsOfDifferentTimeStates = np.array(output4MapsOfDifferentTimeStates)

np.savez("output.npz", *output4MapsOfDifferentTimeStates)

