import torch
from torch import nn

class CNNEMnist(nn.Module):
    def __init__(self, args):
        #TODO: number of channels is fixed in this model now
        super(CNNEMnist, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      ),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      ),
            nn.ReLU(),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(7 * 7 * 32, 256)
        self.fc2 = nn.Linear(256, 62)

    def forward(self, x):
        conv1_ = self.conv1(x)
        conv2_ = self.conv2(conv1_)
        fc_ = conv2_.view(-1, 32*7*7)
        fc1_ = self.fc1(fc_).clamp(min=0)  # Achieve relu using clamp
        output = self.fc2(fc1_)
        return output


class MelCNN(nn.Module):
    def __init__(self, args):
        super(MelCNN, self).__init__()

        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.features = nn.Sequential(
            block(1, 32),
            block(32, 64),
            block(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, args.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
