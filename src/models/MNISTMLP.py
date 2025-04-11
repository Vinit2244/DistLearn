import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import argparse
import pandas as pd
# from tqdm import tqdm
import logging


class MNISTDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        
        # Extract features and labels
        self.labels = df['label'].values
        self.images = df.drop('label', axis=1).values
        
        # Normalize images to [-1, 1] range (similar to original transform)
        self.images = (self.images / 255.0) * 2.0 - 1.0
        
        # Convert to tensors
        self.images = torch.tensor(self.images, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


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
    dataset = MNISTDataset(dataset_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTMLP().to(device)

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
    parser = argparse.ArgumentParser(description='MNIST MLP Training')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--optimizer', type=str, default='adam', 
                        choices=['adam', 'sgd', 'rmsprop'])
    parser.add_argument('--dataset_path', type=str, default='mnist_dataset.csv')
    parser.add_argument('--model_save_path', type=str, default='mnist_mlp.pth')
    
    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    # Initialize dataset and loader
    dataset = MNISTDataset(args.dataset_path)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTMLP().to(device)
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