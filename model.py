
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class FacialKeypointsDetection(nn.Module):

    def __init__(self):
        super(FacialKeypointsDetection, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.batch_normal1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 5)
        self.batch_normal2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 5)
        self.batch_normal3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(p=0.4)
        self.activation = nn.ReLU()

        self.linear1 = nn.Linear(36864, 1024)
        self.linear2 = nn.Linear(1024, 136)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_normal1(x)
        x = self.activation(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch_normal2(x)
        x = self.activation(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.batch_normal3(x)
        x = self.activation(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x



if __name__ == "__main__":
    net = Net()
    dummy_input = torch.randn(1, 1, 224, 224)
    output = net(dummy_input)
    print(output.shape)
