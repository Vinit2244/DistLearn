"""
This script trains all three models (MNIST MLP, Diabetes MLP, and Fashion MNIST CNN)
and provides detailed training progress for each model.
"""

import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import models.DiabetesMLP as diabetes
import models.MNISTMLP as mnist
import models.FashionMNISTCNN as fashion_mnist

def print_header(title):
    print("\n" + "="*50)
    print(f"Training {title}")
    print("="*50 + "\n")

def train_mnist():
    print_header("MNIST MLP Model")
    args_list = ['--batch_size', '64', '--learning_rate', '0.001', '--epochs', '10', '--optimizer', 'adam', '--dataset_path', '../data/mnist_dataset.csv', '--model_save_path', 'mnist_mlp.pth']
    mnist.main(args_list)


def train_diabetes():
    print_header("Diabetes MLP Model")
    args_list = ['--batch_size', '64', '--learning_rate', '0.001', '--epochs', '10', '--optimizer', 'adam', '--dataset_path', '../data/diabetes_dataset.csv', '--model_save_path', 'diabetes_mlp.pth']
    diabetes.main(args_list)

def train_fashion_mnist():
    print_header("Fashion MNIST CNN Model")
    args_list = ['--batch_size', '64', '--learning_rate', '0.001', '--epochs', '10', '--optimizer', 'adam', '--dataset_path', '../data/fashion_mnist_dataset.csv', '--model_save_path', 'fashion_mnist_cnn.pth']
    fashion_mnist.main(args_list)

if __name__ == "__main__":
    print("Starting training of all models...")
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\n")
    
    # Train all models
    train_mnist()
    train_diabetes()
    train_fashion_mnist()
    
    print("\nTraining completed for all models!")
