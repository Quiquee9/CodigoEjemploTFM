import torch
import torch.nn as nn
import torch.nn.functional as F

class Deep_Emotion_Saliency(nn.Module):
    # modifico esta linea
    #def __init__(self, training=True)
    def __init__(self, training=True, n_classes = 8):
        '''
        Deep_Emotion class contains the network architecture.
        '''
        super(Deep_Emotion_Saliency,self).__init__()
        self.trainingState = training
        self.conv1 = nn.Conv2d(1,10,3)
        self.conv2 = nn.Conv2d(10,10,3)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(10,10,3)
        self.conv4 = nn.Conv2d(10,10,3)
        self.pool4 = nn.MaxPool2d(2,2)

        self.norm = nn.BatchNorm2d(10)

        self.fc1 = nn.Linear(810,50)
        self.fc2 = nn.Linear(50, n_classes)  # capa salida
        # modifico esta linea
        # self.fc2 = nn.Linear(50,3)
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x_img, x_saliency):
        xs = self.localization(x_saliency)
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x_saliency.size())
        x = F.grid_sample(x_img, grid)
        return x

    def forward(self, input, input_saliency):
        out = self.stn(input, input_saliency)

        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        out = F.relu(self.pool2(out))

        out = F.relu(self.conv3(out))
        out = self.norm(self.conv4(out))
        out = F.relu(self.pool4(out))
        #Nuevo p antes 0.5
        out = F.dropout(out, training=self.trainingState, p=0.5)
        out = out.view(-1, 810)
        out = F.relu(self.fc1(out))
        # a??ado esta
        out = F.dropout(out, training=self.trainingState, p=0.5)
        out = self.fc2(out)

        return out