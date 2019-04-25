import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import csv
from torch.autograd import Variable 
import matplotlib.pyplot as plt
import random 
import statistics


def formRow(row):
  rv = []
  for i in range(len(row)):
    if i == 2: # R / G
      rv.append(float(row[2]) / 10)
    elif i == 6: # Runs
      rv.append(float(row[6]) / 1000)
    elif i == 7: # Hits
      rv.append(float(row[7]) / 2000)
    elif i == 16: # Batting Average
      rv.append(row[16])
    elif i == 17: # OBP
      rv.append(row[17])
    elif i == 18: # SLG
      rv.append(row[18])
    elif i == 19: # OPS
      rv.append(row[19])
    elif i == 20: # OPS+
      rv.append(float(row[20]) / 200)
    elif i == 30: # RA / G
      rv.append(float(row[30]) / 10)
    elif i == 34: # ERA
      rv.append(float(row[33]) / 10)
    elif i == 43: # Hits Allowed
      rv.append(float(row[42]) / 2000) # was 1000
    elif i == 53: # ERA+
      rv.append(float(row[53]) / 200)
    elif i == 55: # WHIP
      rv.append(float(row[55]) / 3)
  return rv


def data_setup(inputData, winData, rawData, w):
  with open('../data/data.csv', newline='') as csvFile:
    reader = csv.reader(csvFile)
    i = 0
    for row in reader:
      if i >= 1:
        temp = row.copy()
        rawData.append(temp)
        year = row.pop(0)
        team = row.pop(0)
        win = row.pop(31)
        if float(win) >= float(w):
          winData.append([float(1)])
        else:
          winData.append([float(0)])
        
        newRow = formRow(row)

        for x in range(len(newRow)):
          newRow[x] = float(newRow[x])
        inputData.append(newRow)
        # if float(year) == float(sys.argv[1]) and team.lower() == sys.argv[2].lower():
        #   desiredRow = newRow.copy()
        #   desiredRowWins = float(0)
        #   if (float(w) >= float(sys.argv[3])):
        #     desiredRowWins = float(1)
      i = i + 1  
  
  return 0, 0
  # return desiredRowWins, desiredRow


def check_accuracy(inputData, our_model, winData):
  correct = 0
  zeroCounter = 0
  # averageMaker
  indices = random.sample(range(0, 1400), 200)
  # print(indices)
  for xy in indices:
    nv = Variable(torch.Tensor(inputData[xy]))
    py = our_model(nv)
    
    if (py.data[0].item() >= 0.50):
      predictedValue = 1
    else:
      zeroCounter = zeroCounter + 1
      predictedValue = 0
    
    if (winData[xy][0] == predictedValue):
      correct = correct + 1
  return float(correct) / 200
  # print("Zero Percentage: ", float(zeroCounter) / 200)


class Net(torch.nn.Module): 
  
  def __init__(self): 
    super(Net, self).__init__() 
    self.fc1 = nn.Linear(13, 1)

  def forward(self, x): 
    y_pred = torch.sigmoid(self.fc1(x))
    return y_pred


def main():
  inputData = []
  winData = []
  rawData = []

  errorArray = []

  w = 0
  while w <= 162:
    try:
      data_setup(inputData, winData, rawData, w)
      our_model = torch.load("./model" + str(w) +".pt")
      # print(w)
      acc = 0
      for i in range(100):
        acc += check_accuracy(inputData, our_model, winData)
      acc = acc / 100
      error = 1 - acc
      print([w, error])
      errorArray.append([w, error])
      
    

    except FileNotFoundError:
      pass

    inputData.clear()
    winData.clear()
    rawData.clear()
    w += 1


  xWins = []
  yError = []
  for pair in errorArray:
    xWins.append(pair[0])
    yError.append(pair[1])

  plt.figure(1)
  plt.plot(xWins, yError)
  plt.show()

if __name__ == '__main__':
  main()
