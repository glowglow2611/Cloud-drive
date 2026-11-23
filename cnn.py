import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torch
import csv


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5, padding=2)
        self.conv2 = nn.Conv2d(10, 10, 5, padding=2)
        self.fc3 = nn.Linear(28*28*10, 20)
        self.fc4 = nn.Linear(20, 10)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc3(x))
        return func.softmax(self.fc4(x), dim=1)


types = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

inputs = []
targets = []
_target = []
title = False
with open('../data/digit-recognizer/train.csv') as myFile:
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

train_inputs = inputs[:int(len(inputs)*0.8)]
test_inputs = inputs[int(len(inputs)*0.8):]

train_targets = targets[:int(len(targets)*0.8)]
test_targets = targets[int(len(targets)*0.8):]

train_target = _target[:int(len(targets)*0.8)]
test_target = _target[int(len(targets)*0.8):]

total = len(train_targets)

train_inputs = torch.Tensor(train_inputs)
train_inputs = train_inputs.reshape(int(len(inputs)*0.8), 1, 28, 28)
train_targets = torch.Tensor(train_targets)
train_target = torch.Tensor(train_target)

train_inputs = train_inputs.cuda()
train_targets = train_targets.cuda()
train_target = train_target.cuda()

model = Classifier()
model.cuda()
mse_loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.005)

for j in range(20):

    for i in range(100):
        out = model(train_inputs)
        loss = mse_loss(out, train_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predicted = torch.max(out.data, 1)[1]
    # predicted = predicted.tolist()
    # correct = 0
    # for i in range(total):
    #     if predicted[i] == _target[i]:
    #         correct += 1

    correct = predicted == train_target
    correct = correct.sum().item()
    accuracy = correct / total

    print(j)
    print(correct)
    print(loss, accuracy)

print('测试集结果')

test_inputs = torch.Tensor(test_inputs)
test_inputs = test_inputs.reshape(int(len(inputs)*0.2), 1, 28, 28)
test_targets = torch.Tensor(test_targets)
test_target = torch.Tensor(test_target)

total = len(test_targets)

test_inputs = test_inputs.cuda()
test_targets = test_targets.cuda()
test_target = test_target.cuda()

out = model(test_inputs)
loss = mse_loss(out, test_targets)

predicted = torch.max(out.data, 1)[1]

correct = predicted == test_target
correct = correct.sum().item()
accuracy = correct / total

print(correct)
print(loss, accuracy)


# passed = input('是否达标')
# if passed == 'True':
#     inputs = []
#     title = False
#     with open('../data/digit-recognizer/test.csv') as myFile:
#         lines = csv.reader(myFile)
#         for line in lines:
#             if title:
#
#                 for i in range(len(line)):
#                     line[i] = int(line[i])
#                 inputs.append(line)
#
#             else:
#                 title = True
#
#     inputs = torch.Tensor(inputs)
#     inputs = inputs.reshape(28000, 1, 28, 28)
#     inputs = inputs.cuda()
#
#     out = model(inputs)
#     predicted = torch.max(out.data, 1)[1]
#     predicted = predicted.tolist()
#
#     csvfile = open('../data/digit-recognizer/result6.csv', "w", newline="")
#     writer = csv.writer(csvfile)
#     writer.writerow(['ImageId', 'Label'])
#     for i in range(len(predicted)):
#         writer.writerow([str(i+1), str(predicted[i])])

torch.save(model.state_dict(), 'cnn_model_weights2pth')
