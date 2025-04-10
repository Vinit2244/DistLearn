# ============================= IMPORTS ==============================
import argparse
import logging
from concurrent import futures
import grpc
import sys
import os
from pathlib import Path
import torch
import json
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','src','generated'))
import file_transfer_pb2
import file_transfer_pb2_grpc as file_transfer_grpc

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','src','utils'))
from utils import clear_screen, wait_for_enter, get_server_address, get_ip, STYLES, CHUNK_SIZE

client_abs_path = Path(__file__).parent.resolve()


# ============================= CLASSES ==============================
class Client:
    def __init__(self, id, ip, port):
        self.id = id
        self.ip = ip
        self.port = port
        self.server = None

    def start_my_server(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        file_transfer_grpc.add_ClientServicer_to_server(ClientServicer(), server)
        server.add_insecure_port(f"{self.ip}:{self.port}")
        server.start()
        logging.info(f"Client {self.id} server started at address {self.ip}:{self.port}")

        # Stop the previous server if it exists
        if self.server is not None:
            self.server.stop(0)
        self.server = server

    def register_with_server(self):
        server_ip, server_port = get_server_address()
        server_channel = grpc.insecure_channel(f"{server_ip}:{server_port}")
        server_stub = file_transfer_grpc.FLServerStub(server_channel)
        request_obj = file_transfer_pb2.ClientInfo()
        
        request_obj.id = self.id
        request_obj.ip = self.ip
        request_obj.port = self.port

        response = server_stub.RegisterClient(request_obj)
        if response.err_code == 0:
            logging.info(f"Client {self.id} successfully registered with FL server")
        else:
            logging.error(response.msg)
        server_channel.close()

    def deregister_from_server(self):
        if self.server is None:
            logging.error("Server not running")
            return
        
        server_ip, server_port = get_server_address()
        server_channel = grpc.insecure_channel(f"{server_ip}:{server_port}")
        server_stub = file_transfer_grpc.FLServerStub(server_channel)
        request_obj = file_transfer_pb2.ClientInfo()
        
        request_obj.id = self.id

        response = server_stub.DeregisterClient(request_obj)
        if response.err_code == 0:
            logging.info(f"Client {self.id} successfully deregistered from FL Server")
            self.server.stop(0)
            self.server = None
        else:
            logging.error(response.msg)
        server_channel.close()

    def send_file_to_server(self, file_path):
        server_ip, server_port = get_server_address()
        server_channel = grpc.insecure_channel(f"{server_ip}:{server_port}")
        server_stub = file_transfer_grpc.FLServerStub(server_channel)
        
        filename = file_path.split('/')[-1]
        file_size = os.path.getsize(file_path)
        logging.info(f"Starting to send file: {filename} (size: {file_size} bytes)")
        
        def request_iterator():
            with open(file_path, 'rb') as f:
                chunk_number = 0
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    chunk_number += 1
                    logging.info(f"Sending chunk {chunk_number} of {len(chunk)} bytes")
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
        sys.path.append(os.path.join(os.path.dirname(__file__), 'received_files'))
        try:
            # Load the configuration
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logging.info(f"Loaded FL config: {config}")
            
            # Extract configuration parameters
            model_type = config.get("model_type")
            optimizer_type = config.get("optimizer")
            learning_rate = config.get("learning_rate")
            num_epochs = config.get("num_epochs")
            batch_size = config.get("batch_size")
            
            # Initialize the model based on model_type
            if model_type == "DiabetesMLP":
                global DiabetesMLP, DiabetesDataset, train_diabetes_model, get_diabetes_optimizer
                from DiabetesMLP import DiabetesMLP, DiabetesDataset, train_model as train_diabetes_model, get_optimizer as get_diabetes_optimizer
                model = DiabetesMLP(input_size=16)
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
            save_path = client_abs_path / 'models'
            save_path.mkdir(exist_ok=True)
            torch.save(model.state_dict(), save_path / f"round_0.pt")
            logging.info(f"Initialized model saved to {save_path}/round_0.pt")
            
        except Exception as e:
            logging.error(f"Error initializing federated learning: {str(e)}", exc_info=True)


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
            
            received_dir = client_abs_path / 'received_files'
            received_dir.mkdir(exist_ok=True)
            # logging.info(f"Created/verified received_files directory at {received_dir}")
            
            file_path = received_dir / filename
            total_chunks = 0
            total_bytes = 0
            
            with open(file_path, 'wb') as f:
                f.write(first_chunk.chunk)
                total_chunks += 1
                total_bytes += len(first_chunk.chunk)
                logging.info(f"Received first chunk of {len(first_chunk.chunk)} bytes")
                
                for chunk in request_iterator:
                    f.write(chunk.chunk)
                    total_chunks += 1
                    total_bytes += len(chunk.chunk)
                    logging.info(f"Received chunk {total_chunks} of {len(chunk.chunk)} bytes")
                    if chunk.is_last_chunk:
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
        try:
            # Extract training parameters from request
            round_id = request.round_id
            # model_weights = request.model_weights
            local_epochs = request.local_epochs
            
            logging.info(f"Starting training for round {round_id}")

            if round_id < 1:
                client.initialise_fl(client_abs_path / 'received_files' / 'fl_config_client.json', client_abs_path / 'received_files' / 'global_model_round_0.pt')
            
            # Save received model weights to file
            # model_path = client_abs_path / "models" / f"round_{round_id}.pt"
            model_path = client_abs_path / 'received_files' / f"global_model_round_{round_id}.pt"
            # with open(model_path, "wb") as f:
            #     f.write(model_weights)

            with open(client_abs_path / 'received_files' / 'fl_config_client.json', 'r') as f:
                config = json.load(f)

            # Extract configuration parameters
            model_type = config.get("model_type")
            optimizer_type = config.get("optimizer")
            learning_rate = config.get("learning_rate")
            # num_epochs = config.get("num_epochs")
            batch_size = config.get("batch_size")

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

            # Initialize the model based on model_type
            if model_type == "DiabetesMLP":
                criterion = torch.nn.BCELoss()
                optimizer = get_diabetes_optimizer(model, optimizer_type, learning_rate)
                dataset = DiabetesDataset(client_abs_path / "data/diabetes_dataset.csv")
            elif model_type == "FashionMNISTCNN":
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = get_fashion_mnist_optimizer(model, optimizer_type, learning_rate)
                dataset = FashionMNISTDataset(client_abs_path / "data/fashion_mnist_dataset.csv")
            elif model_type == "MNISTMLP":
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = get_mnist_optimizer(model, optimizer_type, learning_rate)
                dataset = MNISTDataset(client_abs_path / "data/mnist_dataset.csv")
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
            trained_model_path = client_abs_path / f"models/round_{round_id}_trained.pt"
            torch.save(model.state_dict(), trained_model_path)

            client.send_file_to_server(str(trained_model_path))

            # Send the trained model weights back to the server
            # with open(trained_model_path, "rb") as f:
            #     model_weights = f.read()

            return file_transfer_pb2.TrainingResponse(
                err_code=0,
                msg=f"Training completed for round {round_id}",
                client_id = my_id,
                round_id = round_id,
                # updated_weights=model_weights,
                samples_processed = len(dataset)
            )    

        except Exception as e:
            logging.error(f"Error during training: {str(e)}", exc_info=True)
            return file_transfer_pb2.TrainingResponse(
                err_code=1,
                msg=f"Error during training: {str(e)}"
            )            


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
            file_path = input(f"{STYLES.FG_YELLOW}Enter the file path: {STYLES.RESET}")
            if not Path(file_path).is_file():
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
    parser.add_argument('--port', type=int, required=True, help='Client port')
    parser.add_argument('--id', type=int, required=True, help='Client ID')
    parser.add_argument('--mode', type=str, default='i', help='Mode: i for interactive, a for automatic') 

    args = parser.parse_args()
    mode = args.mode

    global my_port, my_id, my_ip
    my_port = args.port
    my_id = args.id
    my_ip = get_ip()

    # Create a client object
    global client
    client = Client(my_id, my_ip, my_port)

    os.makedirs(client_abs_path / "logs", exist_ok=True)

    # Logging configuration
    logging.basicConfig(
        filename=client_abs_path / "logs" / f"client_{my_id}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info(f"Client {my_id} started with IP {my_ip} and port {my_port}")
    logging.info(f"Pid: {os.getpid()}")

    # 'a' mode for testing using multiple clients
    if mode.lower() == 'i':     # Interactive mode
        menu()
    elif mode.lower() == 'a':   # Automatic mode
        client.start_my_server()
        client.register_with_server()
        while True:
            ...
