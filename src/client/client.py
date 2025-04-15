# ============================= IMPORTS ==============================
import os
import sys
import time
import json
import grpc
import torch
import psutil
import logging
import argparse
import warnings
import speedtest
import numpy as np
from pathlib import Path
from concurrent import futures

# Ignore all warnings
warnings.filterwarnings("ignore")

client_folder_abs_path = Path(__file__).parent.resolve()

sys.path.append(str(client_folder_abs_path / "../../generated"))
import file_transfer_pb2
import file_transfer_pb2_grpc as file_transfer_grpc

sys.path.append(str(client_folder_abs_path / "../../utils"))
from utils import clear_screen, wait_for_enter, get_server_address, get_ip, STYLES, CHUNK_SIZE, DIABETES_MLP_INPUT_SIZE

server_stub = None
fedmodcs_flag = 0  # Declared at module level (global)

# ============================= CLASSES ==============================
class ServerSessionManager:
    def __init__(self):
        server_ip, server_port = get_server_address()
        self.server_ip = server_ip
        self.server_port = server_port
        self.channel = None

    def __enter__(self):
        if encrypt:
            trusted_certs_path = client_folder_abs_path / "../ca.crt"
            with open(trusted_certs_path, "rb") as f:
                trusted_certs = f.read()

            client_key_path = client_folder_abs_path / f"certs/client_{client.id}.key"
            with open(client_key_path, "rb") as f:
                client_key = f.read()

            client_certificate_path = client_folder_abs_path / f"certs/client_{client.id}.crt"
            with open(client_certificate_path, "rb") as f:
                client_certificate = f.read()
            
            credentials = grpc.ssl_channel_credentials(
                root_certificates=trusted_certs,
                private_key=client_key,
                certificate_chain=client_certificate
            )

            credentials = grpc.ssl_channel_credentials(root_certificates=trusted_certs)
            self.channel = grpc.secure_channel(f"{self.server_ip}:{self.server_port}", credentials)
        else:
            self.channel = grpc.insecure_channel(f"{self.server_ip}:{self.server_port}")
        global server_stub
        server_stub = file_transfer_grpc.FLServerStub(self.channel)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.channel:
            self.channel.close()
        global server_stub
        server_stub = None


class Client:
    def __init__(self, id, ip, port):
        self.id = id
        self.ip = ip
        self.port = port
        self.server = None

    def start_my_server(self):
        client_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        file_transfer_grpc.add_ClientServicer_to_server(ClientServicer(), client_server)

        if encrypt:
            certificate_chain_path = client_folder_abs_path / f"certs/client_{self.id}.crt"
            with open(certificate_chain_path, "rb") as f:
                certificate_chain = f.read()

            private_key_path = client_folder_abs_path / f"certs/client_{self.id}.key"
            with open(private_key_path, "rb") as f:
                private_key = f.read()

            root_certificate_path = client_folder_abs_path / "../ca.crt"
            with open(root_certificate_path, "rb") as f:
                root_certificate = f.read()
            
            client_credentials = grpc.ssl_server_credentials(
                [(private_key, certificate_chain)],
                root_certificates=root_certificate
            )

            client_server.add_secure_port(f"{self.ip}:{self.port}", client_credentials)
        else:
            client_server.add_insecure_port(f"{self.ip}:{self.port}")
        client_server.start()
        logging.info(f"Client {self.id} server started at address {self.ip}:{self.port}")

        # Stop the previous server if it exists
        if self.server is not None:
            self.server.stop(0)
        self.server = client_server
        self.send_file_to_server(client_folder_abs_path / f"certs/client_{self.id}.crt")

    def register_with_server(self):
        with ServerSessionManager():
            request_obj = file_transfer_pb2.ClientInfo()
            
            request_obj.id = self.id
            request_obj.ip = self.ip
            request_obj.port = self.port

            response = server_stub.RegisterClient(request_obj)
            if response.err_code == 0:
                logging.info(f"Client {self.id} successfully registered with FL server")
            else:
                logging.error(response.msg)

    def deregister_from_server(self):
        if self.server is None:
            logging.error("Server not running")
            return
        
        with ServerSessionManager():
            request_obj = file_transfer_pb2.ClientInfo()
            
            request_obj.id = self.id

            response = server_stub.DeregisterClient(request_obj)
            if response.err_code == 0:
                logging.info(f"Client {self.id} successfully deregistered from FL Server")
                self.server.stop(0)
                self.server = None
            else:
                logging.error(response.msg)

    def send_file_to_server(self, file_path):
        with ServerSessionManager():
            filename = file_path.name
            file_size = file_path.stat().st_size
            bandwidth = self.get_network_bandwidth()  # In Mbps
            logging.info(f"Starting to send file: {filename} (size: {file_size} bytes)")
            
            def request_iterator():
                with open(file_path, "rb") as f:
                    chunk_number = 0
                    while True:
                        chunk = f.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        chunk_number += 1
                        logging.info(f"Sending chunk {chunk_number} of {len(chunk)} bytes")

                        if fedmodcs_flag:
                            logging.info(f"Simulating network delay for chunk {chunk_number}")
                            # Simulate network delay based on bandwidth
                            # bandwidth is in Mbps, convert to bytes/sec
                            bytes_per_sec = (bandwidth * 1_000_000) / 8
                            time_to_send = len(chunk) / bytes_per_sec
                            time.sleep(time_to_send)

                        yield file_transfer_pb2.FileChunk(
                            filename=filename,
                            id=my_id,
                            chunk=chunk,
                            is_last_chunk=len(chunk) < CHUNK_SIZE
                        )
            
            response = server_stub.TransferFile(request_iterator())
            if response.err_code == 0:
                logging.info(f"File transfer successful: {response.msg}")
            else:
                logging.error(f"File transfer failed: {response.msg}")

    def initialise_fl(self, config_path, model_path):
        sys.path.append(str(client_folder_abs_path / "received_files"))
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            
            logging.info(f"Loaded FL config: {config}")
            
            # Extract configuration parameters
            model_type      = config.get("model_type")
            optimizer_type  = config.get("optimizer")
            learning_rate   = config.get("learning_rate")
            num_epochs      = config.get("num_epochs")
            batch_size      = config.get("batch_size")
            lr_decay        = config.get("lr_decay")
            
            # Initialize the model based on model_type
            if model_type == "DiabetesMLP":
                global DiabetesMLP, DiabetesDataset, train_diabetes_model, get_diabetes_optimizer
                from DiabetesMLP import DiabetesMLP, DiabetesDataset, train_model as train_diabetes_model, get_optimizer as get_diabetes_optimizer
                model = DiabetesMLP(input_size=DIABETES_MLP_INPUT_SIZE)

            elif model_type == "FashionMNISTCNN":
                global FashionMNISTCNN, FashionMNISTDataset, train_fashion_mnist_model, get_fashion_mnist_optimizer
                from FashionMNISTCNN import FashionMNISTCNN, FashionMNISTDataset, train_model as train_fashion_mnist_model, get_optimizer as get_fashion_mnist_optimizer
                model = FashionMNISTCNN()

            elif model_type == "MNISTMLP":
                global MNISTMLP, MNISTDataset, train_mnist_model, get_mnist_optimizer
                from MNISTMLP import MNISTMLP, MNISTDataset, train_model as train_mnist_model, get_optimizer as get_mnist_optimizer
                model = MNISTMLP()

            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Load the model weights
            model.load_state_dict(torch.load(model_path))
            logging.info(f"Model weights loaded from {model_path}")
            
            logging.info("Federated learning initialized successfully")
            logging.info(f"Model type: {model_type}, Epochs: {num_epochs}, Optimizer: {optimizer_type}, Learning rate: {learning_rate}, Batch size: {batch_size}")

            # Here you could start training immediately or wait for a separate start command
            # For now, we'll just save the initialized model for verification
            save_path = client_folder_abs_path / "models"
            save_path.mkdir(exist_ok=True)
            torch.save(model.state_dict(), save_path / f"round_0.pt")
            logging.info(f"Initialized model saved to {save_path}/round_0.pt")
            
        except Exception as e:
            logging.error(f"Error initializing federated learning: {str(e)}", exc_info=True)

    def get_dataset_size(self):
        try:
            config_path = client_folder_abs_path / "received_files/fl_config_client.json"

            with open(config_path, "r") as f:
                config = json.load(f)
            
            model_type = config.get("model_type")
            
            # Determine dataset path based on model type
            if model_type == "DiabetesMLP":
                dataset_path = client_folder_abs_path / "data/diabetes_dataset.csv"
            elif model_type == "FashionMNISTCNN":
                dataset_path = client_folder_abs_path / "data/fashion_mnist_dataset.csv"
            elif model_type == "MNISTMLP":
                dataset_path = client_folder_abs_path / "data/mnist_dataset.csv"
            else:
                logging.error(f"Unsupported model type: {model_type}")
                return 0
            
            with open(dataset_path, "r") as f:
                # Subtract 1 to exclude header row
                dataset_size = len(f.readlines()) - 1
                
            logging.info(f"Dataset size for {model_type}: {dataset_size} samples")
            return dataset_size
            
        except Exception as e:
            logging.error(f"Error determining dataset size: {str(e)}")
            return 0

    def get_cpu_speed_factor(self):
        """Get CPU speed factor from client capabilities with variation"""
        try:
            # Get path to client_capabilities.json (assuming it's in the same directory as the script)
            capabilities_file = Path(__file__).parent / "client_capabilities.json"
            
            with open(capabilities_file, 'r') as f:
                capabilities = json.load(f)
                
            mean_capability = capabilities["mean_capability"]
            variation_percentage = capabilities["variation_percentage"]
            
            # Apply Gaussian variation
            std_dev = mean_capability * (variation_percentage / 100)
            current_capability = max(1, np.random.normal(mean_capability, std_dev))
            
            return current_capability
            
        except Exception as e:
            print(f"Error reading capabilities: {e}, using fallback")
            return 1.0  # Default fallback

    def get_network_bandwidth(self):
        """Get network bandwidth from client capabilities with variation"""
        try:
            # Get path to client_capabilities.json
            capabilities_file = Path(__file__).parent / "client_capabilities.json"
            
            with open(capabilities_file, 'r') as f:
                capabilities = json.load(f)
                
            mean_bandwidth = capabilities["mean_bandwidth"]
            variation_percentage = capabilities["variation_percentage"]
            
            # Apply Gaussian variation (convert to Mbps first if needed)
            std_dev = mean_bandwidth * (variation_percentage / 100)
            current_bandwidth = max(0.1, np.random.normal(mean_bandwidth, std_dev))
            
            return current_bandwidth
            
        except Exception as e:
            print(f"Error reading capabilities: {e}, using fallback")
            return 10.0  # Fallback if speedtest fails
        
    def get_gpu_info(self):
        """Get GPU information from client capabilities"""
        try:
            # Get path to client_capabilities.json
            capabilities_file = Path(__file__).parent / "client_capabilities.json"
            
            with open(capabilities_file, 'r') as f:
                capabilities = json.load(f)
                
            gpu_available = capabilities.get("has_gpu", False)
            
            # If GPU is available, get additional GPU specs if they exist
            if gpu_available:
                return True
            else:
                return False
            
        except Exception as e:
            print(f"Error reading GPU capabilities: {e}, using fallback")
            return {"available": False}  # Default fallback        

class ClientServicer(file_transfer_grpc.ClientServicer):
    def TransferFile(self, request_iterator, context):
        try:
            # Try to get the first chunk
            try:
                first_chunk = next(request_iterator)
                filename = first_chunk.filename
                logging.info(f"Receiving file: {filename}")
            except StopIteration:
                logging.error("No data received from client")
                return file_transfer_pb2.FileResponse(
                    err_code=1,
                    msg="No data received from client"
                )
            
            received_dir = client_folder_abs_path / "received_files"
            received_dir.mkdir(exist_ok=True)
            
            file_path = received_dir / filename
            total_chunks = 0
            total_bytes = 0
            
            with open(file_path, "wb") as f:
                f.write(first_chunk.chunk)
                total_chunks += 1
                total_bytes += len(first_chunk.chunk)
                logging.info(f"Received first chunk of {len(first_chunk.chunk)} bytes")
                
                for chunk in request_iterator:
                    f.write(chunk.chunk)
                    total_chunks += 1
                    total_bytes += len(chunk.chunk)
                    if chunk.is_last_chunk:
                        logging.info(f"Received chunk {total_chunks} of {len(chunk.chunk)} bytes")
                        break
            
            logging.info(f"File transfer completed. Total chunks: {total_chunks}, Total bytes: {total_bytes}")
            
            return file_transfer_pb2.FileResponse(
                err_code=0,
                msg=f"File {filename} transferred successfully"
            )
                
        except Exception as e:
            logging.error(f"Unexpected error during file transfer: {str(e)}", exc_info=True)
            return file_transfer_pb2.FileResponse(
                err_code=1,
                msg=f"Unexpected error: {str(e)}"
            )

    def StartTraining(self, request, context):
        sys.path.append(str(client_folder_abs_path / "received_files"))
        model_type = request.model_type
        if model_type not in ["DiabetesMLP", "FashionMNISTCNN", "MNISTMLP"]:
            raise ValueError(f"Unsupported model type: {model_type}")
        logging.info(f"Received training request for model type: {model_type}")

        if model_type == "DiabetesMLP":
            global DiabetesMLP, DiabetesDataset, train_diabetes_model, get_diabetes_optimizer
            from DiabetesMLP import DiabetesMLP, DiabetesDataset, train_model as train_diabetes_model, get_optimizer as get_diabetes_optimizer
            model = DiabetesMLP(input_size=DIABETES_MLP_INPUT_SIZE)
        elif model_type == "FashionMNISTCNN":
            global FashionMNISTCNN, FashionMNISTDataset, train_fashion_mnist_model, get_fashion_mnist_optimizer
            from FashionMNISTCNN import FashionMNISTCNN, FashionMNISTDataset, train_model as train_fashion_mnist_model, get_optimizer as get_fashion_mnist_optimizer
            model = FashionMNISTCNN()
        elif model_type == "MNISTMLP":
            global MNISTMLP, MNISTDataset, train_mnist_model, get_mnist_optimizer
            from MNISTMLP import MNISTMLP, MNISTDataset, train_model as train_mnist_model, get_optimizer as get_mnist_optimizer
            model = MNISTMLP()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        save_path = client_folder_abs_path / "models"
        save_path.mkdir(exist_ok=True)
        
        try:
            # Extract training parameters from request
            round_id = request.round_id
            local_epochs = request.local_epochs
            
            logging.info(f"Starting training for round {round_id}")

            if round_id < 1:
                client.initialise_fl(client_folder_abs_path / "received_files/fl_config_client.json", client_folder_abs_path / "received_files/global_model_round_0.pt")
            
            # Save received model weights to file
            model_path = client_folder_abs_path / f"received_files/global_model_round_{round_id}.pt"

            client_config_path = client_folder_abs_path / "received_files/fl_config_client.json"
            with open(client_config_path, "r") as f:
                config = json.load(f)

            # Extract configuration parameters
            model_type = config.get("model_type")
            optimizer_type = config.get("optimizer")
            learning_rate = config.get("learning_rate")
            batch_size = config.get("batch_size")
            lr_decay = config.get("lr_decay")

            # Initialize the model based on model_type
            if model_type == "DiabetesMLP":
                model = DiabetesMLP(input_size=16)
            elif model_type == "FashionMNISTCNN":
                model = FashionMNISTCNN()
            elif model_type == "MNISTMLP":
                model = MNISTMLP()
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Load the model
            model.load_state_dict(torch.load(model_path))
            logging.info(f"Model weights loaded from {model_path}")

            # Calculate the decayed learning rate
            if lr_decay > 0:
                learning_rate = learning_rate * (lr_decay ** round_id)
                logging.info(f"Learning rate decayed to {learning_rate}")

            # Initialize the model based on model_type
            if model_type == "DiabetesMLP":
                criterion = torch.nn.BCELoss()
                optimizer = get_diabetes_optimizer(model, optimizer_type, learning_rate)
                dataset = DiabetesDataset(client_folder_abs_path / "data/diabetes_dataset.csv")
            elif model_type == "FashionMNISTCNN":
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = get_fashion_mnist_optimizer(model, optimizer_type, learning_rate)
                dataset = FashionMNISTDataset(client_folder_abs_path / "data/fashion_mnist_dataset.csv")
            elif model_type == "MNISTMLP":
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = get_mnist_optimizer(model, optimizer_type, learning_rate)
                dataset = MNISTDataset(client_folder_abs_path / "data/mnist_dataset.csv")
            else:
                raise ValueError(f"Unsupported model type: {model_type}")       
           
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)   
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Train the model
            if model_type == "DiabetesMLP":
                train_diabetes_model(model, train_loader, criterion, optimizer, device, local_epochs)
            elif model_type == "FashionMNISTCNN":
                train_fashion_mnist_model(model, train_loader, criterion, optimizer, device, local_epochs)
            elif model_type == "MNISTMLP":
                train_mnist_model(model, train_loader, criterion, optimizer, device, local_epochs)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Save the trained model weights
            trained_model_path = client_folder_abs_path / f"models/round_{round_id}_trained.pt"
            torch.save(model.state_dict(), trained_model_path)

            if fedmodcs_flag:
                cpu_speed = client.get_cpu_speed_factor() # samples per second
                simulated_training_time = len(dataset) / cpu_speed
                logging.info(f"Simulating training delay: {simulated_training_time:.2f} seconds based on CPU speed factor")
                time.sleep(simulated_training_time)

            client.send_file_to_server(trained_model_path)

            return file_transfer_pb2.TrainingResponse(
                err_code=0,
                msg=f"Training completed for round {round_id}",
                client_id = my_id,
                round_id = round_id,
                samples_processed = len(dataset)
            )    

        except Exception as e:
            logging.error(f"Error during training: {str(e)}", exc_info=True)
            return file_transfer_pb2.TrainingResponse(
                err_code=1,
                msg=f"Error during training: {str(e)}"
            )   

    def GetResourceInfo(self, request, context):
        try:
            dataset_size = client.get_dataset_size()
            cpu_speed_factor = client.get_cpu_speed_factor()
            network_bandwidth = client.get_network_bandwidth()
            has_gpu = client.get_gpu_info()

            logging.info(f"Sending resource info -> Dataset Size: {dataset_size}, CPU Factor: {cpu_speed_factor}, Bandwidth: {network_bandwidth} Mbps, GPU: {has_gpu}")
            global fedmodcs_flag
            fedmodcs_flag = 1
            
            return file_transfer_pb2.ResourceInfo(
                dataset_size=dataset_size,
                cpu_speed_factor=cpu_speed_factor,
                network_bandwidth=network_bandwidth,
                has_gpu=has_gpu
            )
        except Exception as e:
            logging.error(f"Error in GetResourceInfo: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Failed to retrieve resource info")
            raise        


# ============================= FUNCTIONS ============================
def menu():
    while True:
        clear_screen()
        print(f"{STYLES.FG_MAGENTA + STYLES.BOLD}Client {my_id}{STYLES.RESET}")
        print()
        print(f"{STYLES.FG_YELLOW}Choose an option:{STYLES.RESET}")
        print("  1. Register with Server")
        print("  2. Deregister from Server")
        print("  3. Transfer File")
        print("  4. Exit")
        print()
        choice = input(f"{STYLES.FG_YELLOW}Enter your choice: {STYLES.RESET}")
        if choice == "":
            continue

        elif choice == "1":
            client.start_my_server()
            client.register_with_server()
            wait_for_enter()

        elif choice == "2":
            client.deregister_from_server()
            wait_for_enter()

        elif choice == "3":
            file_path = Path(input(f"{STYLES.FG_YELLOW}Enter the file path: {STYLES.RESET}"))
            if not file_path:
                print(f"{STYLES.BG_RED}File not found. Please try again.{STYLES.RESET}")
                wait_for_enter()
                continue
            client.send_file_to_server(file_path)
            wait_for_enter()

        elif choice == "4":
            print(f"{STYLES.FG_YELLOW}Exiting...{STYLES.RESET}")
            client.deregister_from_server()
            break

        else:
            print(f"{STYLES.BG_RED}Invalid choice. Please try again.{STYLES.RESET}")
            wait_for_enter()


# =============================== MAIN ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True, help="Client port")
    parser.add_argument("--id", type=int, required=True, help="Client ID")
    parser.add_argument("--mode", type=str, default="i", help="Mode: i for interactive, a for automatic")
    parser.add_argument("--encrypt", type=int, default=0, help="Enable encryption (1) or not (0)")

    args = parser.parse_args()
    global encrypt
    encrypt = args.encrypt
    mode = args.mode

    global my_port, my_id, my_ip
    my_port = args.port
    my_id = args.id
    my_ip = get_ip()

    # Create a client object
    global client
    client = Client(my_id, my_ip, my_port)

    logs_dir = client_folder_abs_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Logging configuration
    logging.basicConfig(
        filename=client_folder_abs_path / "logs" / f"client_{my_id}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info(f"Client {my_id} started with IP {my_ip} and port {my_port}")
    logging.info(f"Pid: {os.getpid()}")

    if mode.lower() == "i":     # Interactive mode
        menu()
    elif mode.lower() == "a":   # Automatic mode
        client.start_my_server()
        client.register_with_server()
        while True:
            ...
