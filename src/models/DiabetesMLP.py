import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch.optim as optim
import argparse
import logging
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)

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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


def calculate_accuracy(outputs, labels):
    """Calculate accuracy for binary classification"""
    predictions = (outputs >= 0.5).float()
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return 100 * correct / total


def get_optimizer(model, optimizer_name, learning_rate):
    """Return optimizer based on name"""
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train_model(model, train_loader, criterion, optimizer, device, epochs):
    """Training loop using full dataset"""
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_accuracy = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_accuracy += calculate_accuracy(outputs, labels)
        
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_accuracy / len(train_loader)
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {round(avg_loss, 4)}, Accuracy: {round(avg_accuracy, 2)}%")


def evaluate_model(path_to_weights, dataset_path, batch_size=64):
    dataset = DiabetesDataset(dataset_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiabetesMLP(input_size=dataset.X.shape[1]).to(device)

    model.load_state_dict(torch.load(path_to_weights))
    model.eval()

    criterion = nn.BCELoss()

    total_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_accuracy += calculate_accuracy(outputs, labels)

    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_accuracy / len(data_loader)

    return avg_loss, avg_accuracy


def main(args_list):
    # Parse arguments
    parser = argparse.ArgumentParser(description='Diabetes MLP Training')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--optimizer', type=str, default='adam', 
                        choices=['adam', 'sgd', 'rmsprop'], 
                        help='Optimizer to use (adam, sgd, rmsprop)')
    parser.add_argument('--dataset_path', type=str, default="../data/diabetes_dataset.csv")
    parser.add_argument('--model_save_path', type=str, default="diabetes_mlp.pth")
    
    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(42)

    # Initialize dataset and loader
    dataset = DiabetesDataset(args.dataset_path)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiabetesMLP(input_size=dataset.X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = get_optimizer(model, args.optimizer, args.learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, device, args.epochs)

    # Save only the model weights
    torch.save(model.state_dict(), args.model_save_path)
    logging.info(f"Model weights saved to {args.model_save_path}")
    # print(f"Model weights saved to {args.model_save_path}")


if __name__ == "__main__":
    main()
