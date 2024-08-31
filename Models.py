import torch
import torch.nn as nn


class CustomDinoV1(nn.Module):
    def __init__(self, num_classes):
        super(CustomDinoV1, self).__init__()
        self.resnet = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=True)
        self.activation = nn.LeakyReLU()
        self.fc1 = nn.Linear(768 * 2, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, img1, img2):
        x1 = self.resnet(img1)
        x2 = self.resnet(img2)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class CustomDinoV2(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(CustomDinoV2, self).__init__()
        self.resnet = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=True)
        self.activation = nn.LeakyReLU()
        self.fc1 = nn.Linear(768 * 2, 1024)
        self.dropout = nn.Dropout(p=dropout_rate)  # Add dropout layer
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, img1, img2):
        x1 = self.resnet(img1)
        x2 = self.resnet(img2)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)  # Apply dropout after activation
        x = self.fc2(x)
        return x