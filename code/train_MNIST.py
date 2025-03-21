import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from models.Models import MNISTMLP, MNISTDataLoader

# Define hyperparameters
batch_size = 64
epochs = 10
learning_rate = 0.001

# Initialize DataLoader
data_loader = MNISTDataLoader(batch_size)
train_loader = data_loader.train_loader
test_loader = data_loader.test_loader

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in tqdm.tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")

# Evaluate on test data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Save the model
torch.save({"model_state_dict": model.state_dict(), "architecture": model}, "mnist_mlp.pth")
