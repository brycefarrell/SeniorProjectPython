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

def revert(sds, avgChanges, desiredRow, newDesiredRow):
  for i in range(len(newDesiredRow)):
    if i == 0: # R / G
      sds[i] = sds[i] * 10
      avgChanges[i] = avgChanges[i] * 10
      desiredRow[i] = desiredRow[i] * 10
      newDesiredRow[i] = newDesiredRow[i] * 10
    elif i == 1: # Runs
      sds[i] = sds[i] * 1000
      avgChanges[i] = avgChanges[i] * 1000
      desiredRow[i] = desiredRow[i] * 1000
      newDesiredRow[i] = newDesiredRow[i] * 1000
    elif i == 2: # Hits
      sds[i] = sds[i] * 2000
      avgChanges[i] = avgChanges[i] * 2000
      desiredRow[i] = desiredRow[i] * 2000
      newDesiredRow[i] = newDesiredRow[i] * 2000
    elif i == 7: # OPS+
      sds[i] = sds[i] * 200
      avgChanges[i] = avgChanges[i] * 200
      desiredRow[i] = desiredRow[i] * 200
      newDesiredRow[i] = newDesiredRow[i] * 200
    elif i == 8: # RA / G
      sds[i] = sds[i] * 10
      avgChanges[i] = avgChanges[i] * 10
      desiredRow[i] = desiredRow[i] * 10
      newDesiredRow[i] = newDesiredRow[i] * 10
    elif i == 9: # ERA
      sds[i] = sds[i] * 10
      avgChanges[i] = avgChanges[i] * 10
      desiredRow[i] = desiredRow[i] * 10
      newDesiredRow[i] = newDesiredRow[i] * 10
    elif i == 10: # Hits Allowed
      sds[i] = sds[i] * 2000
      avgChanges[i] = avgChanges[i] * 2000
      desiredRow[i] = desiredRow[i] * 2000
      newDesiredRow[i] = newDesiredRow[i] * 2000
    elif i == 11: # ERA+
      sds[i] = sds[i] * 200
      avgChanges[i] = avgChanges[i] * 200
      desiredRow[i] = desiredRow[i] * 200
      newDesiredRow[i] = newDesiredRow[i] * 200
    elif i == 12: # WHIP
      sds[i] = sds[i] * 3
      avgChanges[i] = avgChanges[i] * 3
      desiredRow[i] = desiredRow[i] * 3
      newDesiredRow[i] = newDesiredRow[i] * 3


def print_output(sds, avgChanges, newDesiredRow, desiredRow):
  statNames = ["R / G:", "Runs:", "Hits:", "Batting Average:", "OBP:", "SLG:", "OPS:", "OPS+:", "RA / G:", "ERA:", "Hits Allowed:", "ERA+:", "WHIP:"]
  print()
  for i in range(len(sds)):
    print("Actual %-18s %8.3f     Predicted %-18s %8.3f     Average Change: %8.3f     Standard Deviation: %8.3f" % (statNames[i], desiredRow[i], statNames[i], newDesiredRow[i], avgChanges[i], sds[i]))


def adjust_parameters(desiredRow, desiredRowWins, our_model):
  # Change desired row and rerun with model

  newWins = desiredRowWins
  
  totalChanges = []
  for iteration in range(10):
    predictedValue = 0
    new = desiredRow.copy()
    changes = [0 for x in range(13)]
    while True:
      
      new_var = Variable(torch.Tensor(new)) 
      pred_y = our_model(new_var) 
     
      if (pred_y.data[0].item() >= 0.50):
        predictedValue = 1
      else:
        predictedValue = 0
     

      if predictedValue == 1:
        break

      numberToChange = random.randrange(1, 14)
      
      for stat in random.sample(range(0, 13), numberToChange):
        if stat <= 7 or stat == 11:
          new[stat] = new[stat] + 0.001
          changes[stat] = changes[stat] + 0.001
        else:
          new[stat] = new[stat] - 0.001
          changes[stat] = changes[stat] - 0.001
    
    totalChanges.append(changes)
  
  avgChanges = [0 for y in range(13)]
  for ind in range(len(avgChanges)):
    for statInd in range(10):
      avgChanges[ind] = avgChanges[ind] + totalChanges[statInd][ind]
  
  for i in range(len(avgChanges)):
    avgChanges[i] = avgChanges[i] / 10
    
  
  transposedTotalChanges = [*zip(*totalChanges)]
  sds = []
  for j in range(13):
    sd = statistics.stdev(list(transposedTotalChanges[j]))
    sds.append(sd)

  # get std dev of the totalChanges
  # also convert this percentages back into hits, R / G, etc.

  newDesiredRow = desiredRow.copy()
  for k in range(13):
    newDesiredRow[k] = newDesiredRow[k] + avgChanges[k]

  revert(sds, avgChanges, desiredRow, newDesiredRow)
  print_output(sds, avgChanges, newDesiredRow, desiredRow)


def data_setup(inputData, winData, rawData):
  drowRawWins = 0
  with open('../data/data.csv', newline='') as csvFile:
    reader = csv.reader(csvFile)
    i = 0
    for row in reader:
      if i >= 1:
        temp = row.copy()
        rawData.append(temp)
        year = row.pop(0)
        team = row.pop(0)
        w = row.pop(31)
        if float(w) >= float(sys.argv[3]):
          winData.append([float(1)])
        else:
          winData.append([float(0)])
        
        newRow = formRow(row)

        for x in range(len(newRow)):
          newRow[x] = float(newRow[x])
        inputData.append(newRow)
        if float(year) == float(sys.argv[1]) and team.lower() == sys.argv[2].lower():
          desiredRow = newRow.copy()
          drowRawWins = float(w)
          desiredRowWins = float(0)
          if (float(w) >= float(sys.argv[3])):
            desiredRowWins = float(1)
      i = i + 1  
  
  return desiredRowWins, desiredRow, drowRawWins


def train_model(inputData, winData):
  
  # our model 
  our_model = Net() 

  optimizer = torch.optim.Adam(our_model.parameters(), lr=0.0006)#6
  criterion = torch.nn.BCELoss()

  our_model.train()
  for epoch in range(30):  # loop over the dataset multiple times

    running_loss = 0.0
    optimizer.zero_grad()
    for i, data in enumerate(inputData, 0):
      # get the inputs
      # inputs, labels = data
      inputs = Variable(torch.tensor(data))#, requires_grad=True)
      
      labels = Variable(torch.tensor(winData[i]))#, requires_grad=True)
     

      # zero the parameter gradients

      # forward + backward + optimize
      outputs = our_model(inputs)
     
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()


      #print statistics
      running_loss += loss.item()
      
      if i % 100 == 99:    # print every 100 mini-batches
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
        running_loss = 0.0
  return our_model


def check_accuracy(inputData, our_model, winData):
  correct = 0
  zeroCounter = 0
  # averageMaker
  indices = random.sample(range(0, 1400), 200)
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
  print("Accuracy: ", float(correct) / 200)
  print("Zero Percentage: ", float(zeroCounter) / 200)


def check_specific_team(desiredRow, desiredRowWins, drowRawWins, our_model):
  new_var = Variable(torch.Tensor(desiredRow)) 
  pred_y = our_model(new_var) 
  print()
  if (desiredRowWins == 0):
    print("Did the team actually meet this win total:   NO  Actual Wins: " + str(drowRawWins))
  else:
    print("Did the team actually meet this win total:   YES Actual Wins: " + str(drowRawWins))

  if (pred_y.data[0].item() >= 0.50):
    predictedValue = 1
    print("Does the algorithm predict this team to meet this win total:   YES")
  else:
    predictedValue = 0
    print("Does the algorithm predict this team to meet this win total:   NO")

  if predictedValue != 1:
    adjust_parameters(desiredRow, desiredRowWins, our_model)


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
  # rawWinData = []
  rawData = []
  desiredRow = []

  teamCodes = ["ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", 
      "CLE", "COL", "DET", "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", 
      "OAK", "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR", "WSN"]

  if len(sys.argv) != 4:
    print("Usage: python3 win_nn.py year team wins")
    print("Years: 1960 - 2018")
    print("Team Codes:")
    for code in teamCodes:
      print(code)
    exit()

  desiredRowWins, desiredRow, drowRawWins = data_setup(inputData, winData, rawData)

  try:
    # print(sys.argv[3])
    our_model = torch.load("./model" + sys.argv[3] +".pt")
    check_specific_team(desiredRow, desiredRowWins, drowRawWins, our_model)
    exit()

  except FileNotFoundError:
    pass
  
  our_model = train_model(inputData, winData)

  check_accuracy(inputData, our_model, winData)  

  check_specific_team(desiredRow, desiredRowWins, drowRawWins, our_model)

  torch.save(our_model, "./model" + sys.argv[3] + ".pt")


if __name__ == '__main__':
  main()