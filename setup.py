import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import shutil

def create_folders_and_distribute_data(n, data_folder='data', clients_folder='clients', server_data_folder='server_data', client_script='src/client/client.py'):
    # Create clients and server_data folders
    os.makedirs(clients_folder, exist_ok=True)
    os.makedirs(server_data_folder, exist_ok=True)

    # Verify client script exists
    if not os.path.exists(client_script):
        raise FileNotFoundError(f"Client script not found at {client_script}")
    
    # Create client folders
    for i in range(1, n + 1):
        client_dir = os.path.join(clients_folder, str(i))
        os.makedirs(client_dir, exist_ok=True)
        os.makedirs(os.path.join(client_dir,'logs'), exist_ok=True)

        # Copy client.py to each client folder
        shutil.copy(client_script, client_dir)
    
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
            client_dir = os.path.join(clients_folder, str(i))
            client_data.to_csv(os.path.join(client_dir, dataset_file), index=False)

# Example usage
create_folders_and_distribute_data(n=5)