import torch.nn as nn


class FaceKeypointModel(nn.Module):
    def __init__(self, freeze_resnet=False):
        super(FaceKeypointModel, self).__init__()

        # Convert 1 filter 3 filter because resnet accepts 3 filter only
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3, 3), stride=1, padding=1,
                                     padding_mode='zeros')

        # Resnet Architecture
        self.resnet18 = models.resnet18(pretrained=True)
        if freeze_resnet:
            for param in self.resnet18.parameters():
                param.requires_grad = False
        # replacing last layer of resnet
        # by default requires_grad in a layer is True
        self.resnet18.fc = torch.nn.Linear(self.resnet18.fc.in_features, 384)

        self.relu = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(384, 30)

    def forward(self, x):
        y0 = self.conv1(x)
        y1 = self.resnet18(y0)
        y_relu = self.relu(y1)
        out = self.linear1(y_relu)
        return out