import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_json
from numpy import load
from collections import defaultdict
from collections import Counter

# load json and create model
json_file = open('pm_VGG19_ImageNet.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

trainedModel = model_from_json(loaded_model_json)
# load weights into new model
trainedModel.load_weights("pm_VGG19_ImageNet.h5")
print("Loaded model from disk")
trainedWeights = np.array(trainedModel.get_weights())
print (len(trainedWeights))

weightList = [x for x in range(0, 37, 2)]
print(weightList)

maxWeightArray = []
for i in weightList:
  layer = trainedWeights[i]
  layer = layer.flatten()
  maxWeightArray.append(abs(max(layer)))

print(maxWeightArray)

maxWeightGlobal = max(maxWeightArray)
print(maxWeightGlobal)

quantizedWeightArray = []
for i in weightList:
  layer = trainedWeights[i]
  layer = layer.flatten()
  layer = layer / maxWeightGlobal
  layer = layer * 255
  layer = layer.astype(int)
  quantizedWeightArray.append(layer)

print(len(quantizedWeightArray))

absWeightValues = quantizedWeightArray
print(np.shape(absWeightValues))
print(len(absWeightValues))
# absWeightValues = []
for i in range(len(absWeightValues)):
	absWeightValues[i] = abs(quantizedWeightArray[i])

print(np.shape(absWeightValues))

weightCount = 0
for i in range(len(absWeightValues)):
  # print(np.shape(absWeightValues[i]))
  weightCount += len(absWeightValues[i])

print(weightCount)

absWeightValuesConcat = np.concatenate(absWeightValues, axis=0)
print(len(absWeightValuesConcat))

uniqueVals = np.unique(absWeightValuesConcat)
print("The unique weight values are: ")
print(uniqueVals)
print("The number of unique weight values is:", len(uniqueVals))

dictionaryEntries = Counter(absWeightValuesConcat).most_common(len(uniqueVals))
print(dictionaryEntries)

for i in range(len(dictionaryEntries)):
  if dictionaryEntries[i][0] == 0:
    print("Count of 0s: ", dictionaryEntries[i][1], "Percentage of 0s: ", dictionaryEntries[i][1]/len(absWeightValuesConcat))
  if dictionaryEntries[i][0] == 1:
    print("Count of 1s: ", dictionaryEntries[i][1], "Percentage of 1s: ", dictionaryEntries[i][1]/len(absWeightValuesConcat))
