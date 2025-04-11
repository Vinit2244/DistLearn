import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import shutil
import argparse
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore all warnings
warnings.filterwarnings("ignore")

# Configure visual style
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)
PLOTS_DIR = "../plots"
COLOR_PALETTE = "viridis"


def create_folders_and_distribute_data(n, IID, NonIID, x, data_folder='../data', clients_folder='../clients', 
                                      server_data_folder='../server_data', client_script="./client/client.py",
                                      server_certificate='./server/server.crt'):
    # Create clients and server_data folders
    os.makedirs(clients_folder, exist_ok=True)
    os.makedirs(server_data_folder, exist_ok=True)

    # Verify client script exists
    if not os.path.exists(client_script):
        raise FileNotFoundError(f"Client script not found at {client_script}")
    
    # Verify server certificate exists
    if not os.path.exists(server_certificate):
        raise FileNotFoundError(f"Client script not found at {server_certificate}")
    
    # Create client folders and copy client script into each one
    for i in range(1, n + 1):
        client_dir = os.path.join(clients_folder, str(i))
        os.makedirs(client_dir, exist_ok=True)
        os.makedirs(os.path.join(client_dir, "data"), exist_ok=True)
        shutil.copy(client_script, client_dir)
        shutil.copy(server_certificate, clients_folder)
        with open(f"{client_dir}/client_{i}.cnf", "w") as f:
            config_file_content = [
                f'[ req ]\n', 
                f'default_bits       = 2048\n', 
                f'prompt             = no\n', 
                f'default_md         = sha256\n', 
                f'distinguished_name = req_distinguished_name\n', 
                f'attributes         = req_attributes\n', 
                f'req_extensions     = req_ext\n', 
                f'\n', 
                f'[ req_distinguished_name ]\n', 
                f'countryName               = IN\n', 
                f'stateOrProvinceName       = Telangana\n', 
                f'localityName              = Hyderabad\n', 
                f'organizationName          = Client_{i} Pvt Ltd\n', 
                f'organizationalUnitName    = FL Client\n', 
                f'commonName                = Client_{i}\n', 
                f'emailAddress              = client{i}@gmail.com\n', 
                f'\n', 
                f'[ req_attributes ]\n', 
                f'challengePassword         = c{i}\n', 
                f'unstructuredName          = c{i}\n', 
                f'\n', 
                f'[ req_ext ]\n', 
                f'subjectAltName = @alt_names\n', 
                f'\n', 
                f'[ alt_names ]\n', 
                f'DNS.1 = localhost\n', 
                f'IP.1 = 127.0.0.1\n']
            f.writelines(config_file_content)
    
    # List of dataset files to process
    dataset_files = ['diabetes_dataset.csv', 'fashion_mnist_dataset.csv', 'mnist_dataset.csv']
    
    for dataset_file in dataset_files:
        # Load the dataset
        dataset_path = os.path.join(data_folder, dataset_file)
        data = pd.read_csv(dataset_path)

        train_data, test_data = train_test_split(data, test_size=0.01, random_state=42)
        
        # Save the test data to the server_data folder
        test_data.to_csv(os.path.join(server_data_folder, dataset_file), index=False)
        
        if dataset_file == 'diabetes_dataset.csv':
            # Split the remaining data equally among the clients
            client_data_splits = np.array_split(train_data, n)
            for i, client_data in enumerate(client_data_splits, start=1):
                client_data_folder = os.path.join(clients_folder, str(i), "data")
                client_data.to_csv(os.path.join(client_data_folder, dataset_file), index=False)
        else:
            # Identify label column and unique classes in the dataset
            label_column = data.columns[-1]
            unique_classes = np.unique(train_data[label_column])
            
            for i in range(1, n + 1):
                if i <= IID:
                    client_sample = train_data.sample(n=600, random_state=42 + i)
                else:
                    # Non-IID: randomly select a single class for this client.
                    selected_class = np.random.choice(unique_classes)
                    subset = train_data[train_data[label_column] == selected_class]

                    replace_flag = len(subset) < 600
                    client_sample = subset.sample(n=600, replace=replace_flag, random_state=42 + i)
                
                client_data_folder = os.path.join(clients_folder, str(i), "data")
                client_sample.to_csv(os.path.join(client_data_folder, dataset_file), index=False)
                    
                client_data_folder = os.path.join(clients_folder, str(i), "data")
                client_sample.to_csv(os.path.join(client_data_folder, dataset_file), index=False)


def visualize_initial_data(data_folder='../data'):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    datasets = ['diabetes_dataset.csv', 'fashion_mnist_dataset.csv', 'mnist_dataset.csv']
    
    # Use a more visually appealing color palette
    color_palette = "plasma"
    
    for dataset in datasets:
        dataset_path = os.path.join(data_folder, dataset)
        if not os.path.exists(dataset_path):
            continue
            
        df = pd.read_csv(dataset_path)
        label_col = df.columns[-1]
        dataset_name = dataset.replace("_dataset.csv", "")
        
        # Get class counts for better figsize estimation
        class_counts = df[label_col].value_counts()
        num_classes = len(class_counts)
        
        # Adjust figure size based on number of classes
        fig_width = max(12, num_classes * 1.2)
        plt.figure(figsize=(fig_width, 8))
        
        # Create plot with improved styling
        ax = sns.countplot(x=label_col, data=df, palette=color_palette, 
                           edgecolor="black", linewidth=1.2, saturation=0.8)
        
        # Improve plot style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add a title and improve axes labels
        plt.title(f'Class Distribution: {dataset_name.title()} Dataset', 
                  pad=20, fontsize=18, fontweight='bold')
        plt.xlabel('Class', labelpad=15, fontsize=14)
        plt.ylabel('Count', labelpad=15, fontsize=14)
        
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
    
        plt.tight_layout(pad=3.0)
        
        # Save
        plot_path = os.path.join(PLOTS_DIR, f'initial_{dataset_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


def visualize_distributed_data(clients_folder='../clients', server_data_folder='../server_data', num_clients=3):
    """Save professional visualizations of distributed class distributions with consistent y-axis and no annotations"""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    datasets = ['diabetes_dataset.csv', 'fashion_mnist_dataset.csv', 'mnist_dataset.csv']
    
    # Use a consistent but distinctive color palette
    color_palette = "viridis"
    
    for dataset in datasets:
        server_path = os.path.join(server_data_folder, dataset)
        if not os.path.exists(server_path):
            continue

        # Load and prepare data
        server_df = pd.read_csv(server_path)
        label_col = server_df.columns[-1]
        server_df['source'] = 'Server'
        
        client_dfs = []
        for client_id in range(1, num_clients + 1):
            client_path = os.path.join(clients_folder, str(client_id), 'data', dataset)
            if not os.path.exists(client_path):
                continue
                
            client_df = pd.read_csv(client_path)
            client_df['source'] = f'Client {client_id}'
            client_dfs.append(client_df)
        
        combined_df = pd.concat([server_df] + client_dfs, ignore_index=True)
        dataset_name = dataset.replace("_dataset.csv", "")
        
        # Find global maximum count to set consistent y-axis
        max_count = 0
        sources = ['Server'] + [f'Client {i}' for i in range(1, num_clients + 1)]
        for source in sources:
            source_data = combined_df[combined_df['source'] == source]
            if len(source_data) > 0:
                source_counts = source_data[label_col].value_counts()
                source_max = source_counts.max()
                max_count = max(max_count, source_max)
        
        # Add some headroom for the y-axis maximum
        y_max = int(max_count * 1.1)
        
        # Count unique classes to better size the plot
        num_classes = combined_df[label_col].nunique()
        
        # Create a custom figure to have more control
        n_panels = num_clients + 1  # Server + clients
        n_cols = min(3, n_panels)  # Max 3 columns
        n_rows = (n_panels + n_cols - 1) // n_cols  # Ceiling division
        
        # Adjust figure size based on number of classes and panels
        fig_width = min(18, max(15, 5 * n_cols))
        fig_height = min(15, max(10, 5 * n_rows))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), 
                                 sharex=True, sharey=True)  # Set sharey=True for consistent y-axis
        if n_panels == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Hide unused axes
        for i in range(n_panels, len(axes)):
            axes[i].set_visible(False)
        
        # Process data sources one by one
        for i, source in enumerate(sources):
            if i >= n_panels:
                break
                
            # Get data for this source
            source_data = combined_df[combined_df['source'] == source]
            if len(source_data) == 0:
                continue
                
            # Plot on the corresponding axis
            ax = axes[i]
            source_counts = source_data[label_col].value_counts().reset_index()
            source_counts.columns = [label_col, 'count']
            source_counts = source_counts.sort_values(by=label_col)
            
            # Create bar plot
            sns.barplot(x=label_col, y='count', data=source_counts, 
                        ax=ax, palette=color_palette, edgecolor="black", 
                        linewidth=1, saturation=0.8)
            
            # Improve styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.grid(True, linestyle='--', alpha=0.6)
            
            # Set consistent y-axis limit for all plots
            ax.set_ylim(0, y_max)
            
            # Set titles and labels
            ax.set_title(f'{source}', fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel('Class', fontsize=12, labelpad=10)
            ax.set_ylabel('Count', fontsize=12, labelpad=10)
            
            # Ensure no rotation on x-ticks
            ax.tick_params(axis='x', rotation=0, labelsize=11)
            ax.tick_params(axis='y', labelsize=11)
        
        # Add overall title
        plt.suptitle(f'Distributed Class Distribution: {dataset_name.title()} Dataset', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96], pad=3.0)
        
        # Save high-quality figure
        plot_path = os.path.join(PLOTS_DIR, f'distributed_{dataset_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Data distribution and visualization system")
    argparser.add_argument("--num_clients", type=int, default=3)
    argparser.add_argument("--visualize_initial", action="store_true")
    argparser.add_argument("--visualize_distributed", action="store_true")
    argparser.add_argument('--IID', type=int, required=True, help="Number of IID clients")
    argparser.add_argument('--NonIID', type=int, required=True, help="Number of Non-IID clients")
    argparser.add_argument('--x', type=int, required=True, help="Number of classes at the Non-IID clients")
    
    args = argparser.parse_args()
    create_folders_and_distribute_data(n=args.num_clients, IID=args.IID, NonIID=args.NonIID, x=args.x)
    
    if args.visualize_initial:
        visualize_initial_data(data_folder='../data')
    
    if args.visualize_distributed:
        visualize_distributed_data(num_clients=args.num_clients)
