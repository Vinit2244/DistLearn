import shutil
import argparse
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Ignore all warnings
warnings.filterwarnings("ignore")
setup_folder_abs_path = Path(__file__).parent.resolve()

# Configure visual style
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)
PLOTS_DIR = setup_folder_abs_path / "data_distribution_plots"
COLOR_PALETTE = "viridis"


def create_folders_and_distribute_data(n, IID, NonIID, x, data_folder=setup_folder_abs_path / "../../data", 
                                       clients_folder=setup_folder_abs_path / "../clients", 
                                       server_data_folder=setup_folder_abs_path / "../server/data", 
                                       client_script=setup_folder_abs_path / "../client/client.py",
                                       server_certificate=setup_folder_abs_path / "../server/certs/server.crt",
                                       ca_certificate=setup_folder_abs_path / "../CA/ca.crt",
                                       test_data_fraction=0.1,
                                       num_data_points=100,
                                       num_data_points_diabetes=400,
                                       n_classes=10):
    # Create clients and server_data folders
    clients_folder.mkdir(parents=True, exist_ok=True)
    server_data_folder.mkdir(parents=True, exist_ok=True)

    # Verify files exist
    if not client_script.exists():
        raise FileNotFoundError(f"Client script not found at {client_script}")
    
    if not server_certificate.exists():
        raise FileNotFoundError(f"Client script not found at {server_certificate}")
    
    # Create client folders and copy client script into each one
    for i in range(1, n + 1):
        client_dir = clients_folder / str(i)
        client_dir.mkdir(parents=True, exist_ok=True)
        client_data_dir = client_dir / "data"
        client_data_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(client_script, client_dir)
        shutil.copy(server_certificate, clients_folder)
        shutil.copy(ca_certificate, clients_folder)
        shutil.copy(ca_certificate, setup_folder_abs_path / "../server/certs")

        client_certs_folder = client_dir / "certs"
        client_certs_folder.mkdir(parents=True, exist_ok=True)
        client_config_file = client_certs_folder / f"client_{i}.cnf"
        with open(client_config_file, "w") as f:
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
        # Reading and splitting data
        dataset_path = data_folder / dataset_file
        data = pd.read_csv(dataset_path)
        train_data, test_data = train_test_split(data, test_size=test_data_fraction, random_state=42)
        
        # Save the test data to the server_data folder
        test_data.to_csv(server_data_folder / dataset_file, index=False)
        
        if dataset_file == 'diabetes_dataset.csv':
            for i in range(1, n + 1):
                client_sample = train_data.sample(n=num_data_points_diabetes, random_state=42 + i)
                client_data_folder = clients_folder / str(i) / "data"
                client_sample.to_csv(client_data_folder / dataset_file, index=False)
            
        else:
            shard_size = num_data_points // x

            prev_label = -1

            for i in range(1, n + 1):
                if i <= IID:
                    # IID client: randomly sample from the entire dataset
                    client_sample = train_data.sample(n=num_data_points, random_state=42 + i)
                else:
                    client_shards = []
                    for j in range(x):
                        next_label = (prev_label + 1) % n_classes
                        shard = train_data[train_data['label'] == next_label]
                        shard = shard.sample(n=shard_size, random_state=42 + i + j)
                        client_shards.append(shard)
                        prev_label = next_label
                    client_sample = pd.concat(client_shards)
                
                client_data_folder = clients_folder / str(i) / "data"
                client_sample.to_csv(client_data_folder / dataset_file, index=False)


def visualize_initial_data(data_folder=setup_folder_abs_path / "../../data"):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    datasets = ['diabetes_dataset.csv', 'fashion_mnist_dataset.csv', 'mnist_dataset.csv']
    
    # Use a more visually appealing color palette
    color_palette = "plasma"
    
    for dataset in datasets:
        dataset_path = data_folder / dataset
        if not dataset_path.exists():
            continue
            
        df = pd.read_csv(dataset_path)
        label_col = df.columns[-1]
        dataset_name = dataset.replace("_dataset.csv", "")
        
        class_counts = df[label_col].value_counts()
        num_classes = len(class_counts)

        fig_width = max(12, num_classes * 1.2)
        plt.figure(figsize=(fig_width, 8))
        
        # Styling
        ax = sns.countplot(x=label_col, data=df, palette=color_palette, edgecolor="black", linewidth=1.2, saturation=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.title(f'Class Distribution: {dataset_name.title()} Dataset', pad=20, fontsize=18, fontweight='bold')
        plt.xlabel('Class', labelpad=15, fontsize=14)
        plt.ylabel('Count', labelpad=15, fontsize=14)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
    
        plt.tight_layout(pad=3.0)
        plt.savefig(PLOTS_DIR / f'initial_{dataset_name}.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


def visualize_distributed_data(clients_folder=setup_folder_abs_path / "../clients", server_data_folder=setup_folder_abs_path / "../server/data", num_clients=3):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    datasets = ['diabetes_dataset.csv', 'fashion_mnist_dataset.csv', 'mnist_dataset.csv']
    
    # Use a consistent but distinctive color palette
    color_palette = "viridis"
    
    for dataset in datasets:
        server_dataset_path = server_data_folder / dataset
        if not server_dataset_path.exists():
            continue

        # Load and prepare data
        server_df = pd.read_csv(server_dataset_path)
        label_col = server_df.columns[-1]
        server_df['source'] = 'Server'
        
        client_dfs = []
        for client_id in range(1, num_clients + 1):
            client_dataset_path = clients_folder / str(client_id) / 'data' / dataset
            if not client_dataset_path.exists():
                continue
                
            client_df = pd.read_csv(client_dataset_path)
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
        
        n_panels = num_clients + 1  # +1 for server
        n_cols = min(3, n_panels)
        n_rows = (n_panels + n_cols - 1) // n_cols
        
        # Adjust figure size based on number of classes and panels
        fig_width = min(18, max(15, 5 * n_cols))
        fig_height = min(15, max(10, 5 * n_rows))
        
        _, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharex=True, sharey=True)
        if n_panels == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Hide unused axes
        for i in range(n_panels, len(axes)):
            axes[i].set_visible(False)
        
        for i, source in enumerate(sources):
            if i >= n_panels:
                break
            
            source_data = combined_df[combined_df['source'] == source]
            if len(source_data) == 0:
                continue
            
            ax = axes[i]
            source_counts = source_data[label_col].value_counts().reset_index()
            source_counts.columns = [label_col, 'count']
            source_counts = source_counts.sort_values(by=label_col)
            
            sns.barplot(x=label_col, y='count', data=source_counts, 
                        ax=ax, palette=color_palette, edgecolor="black", 
                        linewidth=1, saturation=0.8)
            
            # Styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.grid(True, linestyle='--', alpha=0.6)
            ax.set_ylim(0, y_max)
            ax.set_title(f'{source}', fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel('Class', fontsize=12, labelpad=10)
            ax.set_ylabel('Count', fontsize=12, labelpad=10)
            ax.tick_params(axis='x', rotation=0, labelsize=11)
            ax.tick_params(axis='y', labelsize=11)
        
        plt.suptitle(f'Distributed Class Distribution: {dataset_name.title()} Dataset', fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96], pad=3.0)
        plt.savefig(PLOTS_DIR / f'distributed_{dataset_name}.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Data distribution and visualization system")
    argparser.add_argument("--num_clients", type=int, default=3)
    argparser.add_argument("--visualize_initial", action="store_true")
    argparser.add_argument("--visualize_distributed", action="store_true")
    argparser.add_argument('--IID', type=int, required=True, help="Number of IID clients")
    argparser.add_argument('--NonIID', type=int, required=True, help="Number of Non-IID clients")
    argparser.add_argument('--x', type=int, required=True, help="Number of classes at the Non-IID clients")
    argparser.add_argument('--num_data_points', type=int, default=1500, help="Number of data points per client")
    argparser.add_argument('--num_data_points_diabetes', type=int, default=400, help="Number of data points for diabetes dataset")
    
    args = argparser.parse_args()
    create_folders_and_distribute_data(n=args.num_clients, IID=args.IID, NonIID=args.NonIID, x=args.x, 
                                       num_data_points=args.num_data_points, num_data_points_diabetes=args.num_data_points_diabetes)
    
    if args.visualize_initial:
        visualize_initial_data()
    
    if args.visualize_distributed:
        visualize_distributed_data(num_clients=args.num_clients)
