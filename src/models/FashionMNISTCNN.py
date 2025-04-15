import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import argparse
import pandas as pd
# from tqdm import tqdm
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

class FashionMNISTDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        
        # Extract features and labels
        self.labels = df['label'].values
        self.images = df.drop('label', axis=1).values
        
        # Normalize and reshape images
        self.images = self.images / 255.0  # Scale to [0, 1]
        self.images = self.images.reshape(-1, 1, 28, 28)  # Reshape to (N, C, H, W)
        
        # Convert to tensors
        self.images = torch.tensor(self.images, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.model(x)


def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
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
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_accuracy = 0
        
        for images, labels in train_loader:
        # for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_accuracy += calculate_accuracy(outputs, labels)
        
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_accuracy / len(train_loader)
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%")
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {round(avg_loss, 4)}, Accuracy: {round(avg_accuracy, 2)}%")


def evaluate_model(path_to_weights, dataset_path, batch_size=64):
    dataset = FashionMNISTDataset(dataset_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FashionMNISTCNN().to(device)

    model.load_state_dict(torch.load(path_to_weights))
    model.eval()

    criterion = nn.CrossEntropyLoss()

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
    parser = argparse.ArgumentParser(description='FashionMNIST CNN Training')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--optimizer', type=str, default='adam', 
                        choices=['adam', 'sgd', 'rmsprop'])
    parser.add_argument('--dataset_path', type=str, default='fashion_mnist_dataset.csv')
    parser.add_argument('--model_save_path', type=str, default='fashion_mnist_cnn.pth')

    set_seed(42)
    
    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    # Initialize dataset and loader
    dataset = FashionMNISTDataset(args.dataset_path)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FashionMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args.optimizer, args.learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, device, args.epochs)

    # Save only the model weights
    torch.save(model.state_dict(), args.model_save_path)
    logging.info(f"Model weights saved to {args.model_save_path}")
    # print(f"Model weights saved to {args.model_save_path}")


if __name__ == "__main__":
    main()