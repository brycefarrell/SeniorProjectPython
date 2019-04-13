import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import csv
from torch.autograd import Variable 
import matplotlib.pyplot as plt
# import statistics
  

inputData = []
winData = []
rawData = []
desiredRow = 0
desiredRowWins = 0
modelDNE = 1


def formRow(row):
  rv = []
  for i in range(len(row)):
    if i == 2: # R / G
      rv.append(float(row[2]) / 10)
    elif i == 6: # Runs
      rv.append(float(row[6]) / 1000)
    elif i == 7: # Hits
      rv.append(float(row[7]) / 2000)
    if i == 16: # Batting Average
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
    elif i == 43: # Hits
      rv.append(float(row[42]) / 2000) # was 1000
    elif i == 53: # ERA+
      rv.append(float(row[53]) / 200)
    elif i == 55: # WHIP
      rv.append(float(row[55]) / 3)
  return rv


if len(sys.argv) != 4:
  print("Usage: python3 win_nn.py year team wins")
  exit()

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
        # print(year)
        # print(team.lower())
        desiredRow = newRow.copy()
        desiredRowWins = float(0)
        if (float(w) >= float(sys.argv[3])):
          desiredRowWins = float(1)
    i = i + 1  

# print(rawData[0])
# print(desiredRow)
# print(desiredRowWins)
print(inputData[0])
# print(winData)
# print(len(inputData[0]))
# print(wp)

  
  
class Net(torch.nn.Module): 
  
    def __init__(self): 
        super(Net, self).__init__() 
        self.fc1 = nn.Linear(13, 1)
  
    def forward(self, x): 
        y_pred = torch.sigmoid(self.fc1(x))
        return y_pred

try:
  our_model = torch.load("./model" + sys.argv[3] +".pt")
  new_var = Variable(torch.Tensor(desiredRow)) 
  pred_y = our_model(new_var) 
  print("actual", desiredRowWins)
  if (pred_y.data[0].item() >= 0.50):
    predictedValue = 1
  else:
    predictedValue = 0
  print("prediction (after training)", predictedValue)
  exit()

except FileNotFoundError:
  pass


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
    # print(inputs)
    labels = Variable(torch.tensor(winData[i]))#, requires_grad=True)
    # print(labels)

    # zero the parameter gradients



    # forward + backward + optimize
    outputs = our_model(inputs)
    # print(outputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if i == 1:
      print(inputs)
      print(outputs)

    #print statistics
    running_loss += loss.item()
    # print(loss.item())
    if i % 100 == 99:    # print every 100 mini-batches
      print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
      running_loss = 0.0


for val in our_model.parameters():
  print("Weights: ", val.data)

correct = 0
zeroCounter = 0
# averageMaker
for xy in range(200):
  nv = Variable(torch.Tensor(inputData[xy]))
  py = our_model(nv)
  print("actual", winData[xy])
  if (py.data[0].item() >= 0.50):
    predictedValue = 1
  else:
    zeroCounter = zeroCounter + 1
    predictedValue = 0
  print("prediction (after training)", py.data[0].item())
  if (winData[xy][0] == predictedValue):
    correct = correct + 1
print("Accuracy: ", float(correct) / 200)
print("Zero Percentage: ", float(zeroCounter) / 200)
  



new_var = Variable(torch.Tensor(desiredRow)) 
pred_y = our_model(new_var) 
print("CLI actual", desiredRowWins)
if (pred_y.data[0].item() >= 0.50):
  predictedValue = 1
else:
  predictedValue = 0
print("CLI prediction (after training)", predictedValue)


# Change desired row and rerun with model
print(desiredRow)


torch.save(our_model, "./model" + sys.argv[3] + ".pt")