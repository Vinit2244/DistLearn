"""
This is a script for defining the MLP and CNN models 
for the MNIST, Diabetes and Fashion MNIST datasets.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd

###################################################################################################

class MNISTDataLoader:
    def __init__(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.train_dataset = torchvision.datasets.MNIST(root='../data', train=True, transform=transform, download=False)
        self.test_dataset = torchvision.datasets.MNIST(root='../data', train=False, transform=transform, download=False)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

# Define the MLP model
class MNISTMLP(nn.Module):
    def __init__(self):
        super(MNISTMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
###################################################################################################

# Custom dataset class
class DiabetesDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        
        # Extract features and target
        self.X = df.iloc[:, :-1].values  
        self.y = df.iloc[:, -1].values   
        
        # Standardize features
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        
        # Convert to PyTorch tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)  # Make it (N, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# Define the MLP model
class DiabetesMLP(nn.Module):
    def __init__(self, input_size):
        super(DiabetesMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()  # Binary classification activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)  # Output between 0 and 1
        return x

###################################################################################################


class FashionMNISTDataLoader:
    def __init__(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.train_dataset = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=transform, download=False)
        self.test_dataset = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=transform, download=False)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)


class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)
    
###################################################################################################

