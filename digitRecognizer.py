import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torch
import csv


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = func.relu(self.fc1(x))
        return func.softmax(self.fc2(x))


types = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

inputs = []
targets = []
_target = []
title = False
with open('E:/program/kaggle/Digit Recognizer/data/digit-recognizer/train.csv') as myFile:
    lines = csv.reader(myFile)
    for line in lines:
        if title:

            target = []
            for j in range(len(types)):
                if line[0] == types[j]:
                    target.append(1)
                else:
                    target.append(0)
            targets.append(target)
            _target.append(int(line[0]))

            for i in range(len(line)):
                line[i] = int(line[i])
            inputs.append(line[1:])

        else:
            title = True

total = len(target)

inputs = torch.Tensor(inputs)
targets = torch.Tensor(targets)
inputs = inputs.cuda()
targets = targets.cuda()

model = Classifier()
model.cuda()
mse_loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.005)

for j in range(10):

    for i in range(1000):
        out = model(inputs)
        loss = mse_loss(out, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predicted = torch.max(out.data, 1)[1]
    predicted = predicted.tolist()
    correct = 0
    for i in range(total):
        if predicted[i] == _target[i]:
            correct += 1

    accuracy = correct / total

    print(loss, accuracy)


inputs = []
title = False
with open('E:/program/kaggle/Digit Recognizer/data/digit-recognizer/test.csv') as myFile:
    lines = csv.reader(myFile)
    for line in lines:
        if title:

            for i in range(len(line)):
                line[i] = int(line[i])
            inputs.append(line)

        else:
            title = True

inputs = torch.Tensor(inputs)
inputs = inputs.cuda()

out = model(inputs)
predicted = torch.max(out.data, 1)[1]
predicted = predicted.tolist()

csvfile = open('E:/program/kaggle/Digit Recognizer/data/digit-recognizer/result1.csv', "w", newline="")
writer = csv.writer(csvfile)
writer.writerow(['ImageId', 'Label'])
for i in range(len(predicted)):
    writer.writerow([str(i+1), str(predicted[i])])
