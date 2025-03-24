import pandas as pd
from torchvision.datasets import FashionMNIST, MNIST

def process_dataset(dataset, dataset_name):
    """Process a torchvision dataset into a flattened DataFrame"""
    # Get images and labels
    images = dataset.data.numpy()  # shape (n_samples, 28, 28)
    labels = dataset.targets.numpy()
    
    # Flatten images
    flattened_images = images.reshape(images.shape[0], -1)  # shape (n_samples, 784)
    
    # Create DataFrame
    columns = [f'pixel_{i}' for i in range(flattened_images.shape[1])]
    df = pd.DataFrame(flattened_images, columns=columns)
    df['label'] = labels
    
    return df

def save_dataset_to_csv(dataset_class, root_path, output_filename):
    """Load, process and save a dataset to CSV"""
    # Load both train and test sets
    train_data = dataset_class(root=root_path, train=True, download=False)
    test_data = dataset_class(root=root_path, train=False, download=False)
    
    # Process both datasets
    train_df = process_dataset(train_data, "train")
    test_df = process_dataset(test_data, "test")
    
    # Combine
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Save to CSV
    combined_df.to_csv(output_filename, index=False)
    
    print(f"\nDataset saved to {output_filename}")
    print(f"Shape: {combined_df.shape}")
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    print(f"Total samples: {len(combined_df)}")

# Process FashionMNIST
print("Processing FashionMNIST...")
save_dataset_to_csv(FashionMNIST, '.', 'fashion_mnist_dataset.csv')

# Process MNIST
print("\nProcessing MNIST...")
save_dataset_to_csv(MNIST, '.', 'mnist_dataset.csv')

print("\nBoth datasets processed successfully!")