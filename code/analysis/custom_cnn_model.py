"""The custom CNN baseline of Section 2.5.1 of the paper (class NeuralNetwork), with no-op freeze/unfreeze methods so
the original training code (code/run_all_experiments.py) can train it like the ResNets.
install() makes base.TransferResNet return this network for variant "custom_cnn" and a ResNet otherwise.
"""
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((4, 4))

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64 * 4 * 4, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        return self.fc2(x).squeeze(-1)

    def freeze_backbone(self):  # trained from scratch: nothing is frozen
        pass

    def unfreeze_backbone(self):
        pass


def install(base):
    resnet = base.TransferResNet

    def factory(variant="resnet50"):
        return NeuralNetwork() if variant == "custom_cnn" else resnet(variant=variant)
    base.TransferResNet = factory
