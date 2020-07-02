## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=3, padding=0)

        # output size = (W-F)/S +1 = (224-5)/3 + 1 = 74
        self.pool1 = nn.MaxPool2d(2, 2)
        # 74/2 = 37  the output Tensor for one image, will have the 
        #dimensions: (16, 37, 37) 

        self.conv2 = nn.Conv2d(16,32,3) 
        #dimensions: (32, 35, 35) 
        
        self.conv3 = nn.Conv2d(32,64,3)
        #dimensions: (64, 33, 33) 
        self.pool3 = nn.MaxPool2d(2, 2)
        #33/2=17    the output Tensor for one image, will have the 
        #dimensions: (64, 17, 17) 
        self.conv4 = nn.Conv2d(64,128,1)
        #dimensions: (128, 17, 17)
        self.pool4 = nn.MaxPool2d(2, 2)
        #dimensions: (256, 9, 9) 
        self.fc1 = nn.Linear(128*8*8, 256)
        self.fc2 = nn.Linear(256, 136)

        self.drop2 = nn.Dropout(p = 0.25)
        self.drop3 = nn.Dropout(p = 0.25)
        self.drop4 = nn.Dropout(p = 0.25)
        

    # define the feedforward behavior
    def forward(self, x):
      
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.drop2(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop3(x)
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.drop4(x)
        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
