import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def create_folders_and_distribute_data(n, data_folder='data', clients_folder='clients', server_data_folder='server_data'):
    # Create clients and server_data folders
    os.makedirs(clients_folder, exist_ok=True)
    os.makedirs(server_data_folder, exist_ok=True)
    
    # Create client folders
    for i in range(1, n + 1):
        os.makedirs(os.path.join(clients_folder, str(i)), exist_ok=True)
    
    # List of dataset files
    dataset_files = ['diabetes_dataset.csv', 'fashion_mnist_dataset.csv', 'mnist_dataset.csv']
    
    for dataset_file in dataset_files:
        # Load the dataset
        dataset_path = os.path.join(data_folder, dataset_file)
        data = pd.read_csv(dataset_path)
        
        # Split the data into server test data (10%) and remaining data (90%)
        train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
        
        # Save the test data to the server_data folder
        test_data.to_csv(os.path.join(server_data_folder, dataset_file), index=False)
        
        # Split the remaining data equally among the clients
        client_data_splits = np.array_split(train_data, n)
        
        for i, client_data in enumerate(client_data_splits, start=1):
            client_data.to_csv(os.path.join(clients_folder, str(i), dataset_file), index=False)

# Example usage
create_folders_and_distribute_data(n=5)