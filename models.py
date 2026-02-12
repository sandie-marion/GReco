import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

# Model Selection
################################################################################################
################################################################################################
def get_model(dataset_name: str, device: torch.device) -> Module:
    """
    Returns a model corresponding to the given dataset name.
    """
    model_mapping = {
        'CIFAR10': (CIFAR10_Model, 10),
        'MNIST': (NIST_Model, 10),
        'Fashion_MNIST': (NIST_Model, 10),
        'Purchase100': (Purchase100_Model, 100),
        'EMNIST' : (NIST_Model, 62),
    }

    if dataset_name not in model_mapping:
        raise ValueError("Unknown dataset name for model selection")

    model_class, n_classes = model_mapping[dataset_name]

    return model_class(n_classes).to(device)

# Models
################################################################################################
################################################################################################

# Model from paper: Machine Learning with Membership Privacy using Adversarial Regularization
################################################################################################
class Purchase100_Model(Module):
    def __init__(self, n_classes):
        super(Purchase100_Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(600, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# Model from paper: Fixing by Mixing: A Recipe for Optimal Byzantine ML Under Heterogeneity
################################################################################################
class NIST_Model(Module):
    def __init__(self, n_classes):
        super(NIST_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, padding=0, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=5, padding=0, stride=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(20 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(-1, 20 * 4 * 4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    

# Model from paper: LEAF : A Benchmark for Federated Settings 
################################################################################################
class EMNIST_Model(Module):
    def __init__(self, n_classes):
        super(EMNIST_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=0, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=0, stride=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64*4*4, 2048)
        self.fc2 = nn.Linear(2048, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x) 

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(-1, 64*4*4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


# Model from paper: Fixing by Mixing: A Recipe for Optimal Byzantine ML Under Heterogeneity
################################################################################################
class CIFAR10_Model(Module):
    def __init__(self, n_classes):
        super(CIFAR10_Model, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(128)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.25)

        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn4(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = x.view(-1, 128 * 2 * 2)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x
