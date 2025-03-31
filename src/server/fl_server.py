# ============================= IMPORTS =============================
import argparse
import logging
from concurrent import futures
import grpc
import sys
import os
from pathlib import Path
import json
import subprocess
import torch
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'generated'))
import file_transfer_pb2
import file_transfer_pb2_grpc as file_transfer_grpc

sys.path.append("utils")
from utils import clear_screen, wait_for_enter, get_ip, STYLES, CHUNK_SIZE

sys.path.append("../models")
from DiabetesMLP import DiabetesMLP, evaluate_model as evaluate_diabetes_model
from FashionMNISTCNN import FashionMNISTCNN, evaluate_model as evaluate_fashion_mnist_model
from MNISTMLP import MNISTMLP, evaluate_model as evaluate_mnist_model


# ============================= CLASSES =============================
class FLServer:
    def __init__(self, port, ip):
        self.my_port = port
        self.my_ip = ip
        self.clients = {}
        self.server_instance = None

        self.register_with_consul()
        self.start_server()
    
    def register_with_consul(self):
        data = {
            "service": {
                "name": "fl-server",
                "id": "fl",
                "port": self.my_port,
                "check": {
                    "http": f"http://{self.my_ip}:{self.my_port}/health",
                    "interval": "2s"
                }
            }
        }

        # Creating a consul service definition file
        with open("./server/fl-server.json", "w") as json_file:
            json.dump(data, json_file)
        logging.info("FL service definition file created")

        # Registering the service with consul
        command = ["consul", "services", "register", "./server/fl-server.json"]
        try:
            result = subprocess.run(command, check=True)
            logging.info("FL service registered with consul")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error registering service: {e}")
            logging.error("Error registering FL service")
            sys.exit(1)

    def start_server(self):
    # Starting FL server
        fl_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        file_transfer_grpc.add_FLServerServicer_to_server(FLServerServicer(), fl_server)
        fl_server.add_insecure_port(f"{self.my_ip}:{self.my_port}")
        fl_server.start()
        logging.info(f"FL server started at address {self.my_ip}:{self.my_port}")
        clear_screen()
        print(f"FL server started at address {self.my_ip}:{self.my_port}")
        self.server_instance = fl_server

    def terminate_server(self):
        if self.server_instance is None:
            logging.error("Server not running")
            return
        self.server_instance.stop(0)
        self.server_instance = None
        logging.info("FL server terminated")
        clear_screen()
        print("FL server terminated")

    def send_file_to_client(self, file_path, client_id):
        print(self.clients)
        if client_id not in self.clients.keys():
            print(f"{STYLES.BG_RED}Client not registered. Please try again.{STYLES.RESET}")
            return
        client_ip, client_port = self.clients[client_id]
        channel = grpc.insecure_channel(f"{client_ip}:{client_port}")
        stub = file_transfer_grpc.ClientStub(channel)
        
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
                        id=-1,
                        chunk=chunk,
                        is_last_chunk=len(chunk) < CHUNK_SIZE
                    )
        
        response = stub.TransferFile(request_iterator())
        if response.err_code == 0:
            logging.info(f"File transfer successful: {response.msg}")
            print(f"{STYLES.FG_GREEN}Success: {response.msg}{STYLES.RESET}")
        else:
            logging.error(f"File transfer failed: {response.msg}")
            print(f"{STYLES.BG_RED + STYLES.FG_WHITE}Error: {response.msg}{STYLES.RESET}")

    def initialize_fl(self, num_epochs, learning_rate, optimizer, batch_size, model_type, client_fraction):
        """ Initializes FL by saving the config file, initializing model, and sending them to clients """
        
        # Step 1: Create FL configuration file
        fl_config = {
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "optimizer": optimizer,
            "batch_size": batch_size,
            "model_type": model_type,
            "client_fraction": client_fraction
        }
        config_path = "./server/fl_config.json"
        with open(config_path, "w") as f:
            json.dump(fl_config, f)
        
        logging.info(f"FL Configuration saved at {config_path}")

        # Step 2: Initialize model based on model_type
        model = None
        if model_type == "DiabetesMLP":
            model = DiabetesMLP(input_size=16)
        elif model_type == "FashionMNISTCNN":
            model = FashionMNISTCNN()
        elif model_type == "MNISTMLP":
            model = MNISTMLP()
        else:
            logging.error(f"Invalid model type: {model_type}")
            return

        os.makedirs("./server/models", exist_ok=True)
        
        model_path = "./server/models/initialized_model.pt"
        torch.save(model.state_dict(), model_path)
        logging.info(f"Initialized model saved at {model_path}")

        # Step 3: Select clients randomly based on client_fraction
        num_clients = len(self.clients)
        num_selected = max(1, int(client_fraction * num_clients))
        selected_clients = random.sample(list(self.clients.keys()), num_selected)
        
        logging.info(f"Selected {num_selected}/{num_clients} clients for training: {selected_clients}")

        # Step 4: Send config and model to each selected client
        for client_id in selected_clients:
            logging.info(f"Sending configuration and model to client {client_id}")
            self.send_file_to_client(config_path, client_id)
            self.send_file_to_client(model_path, client_id)
        
        return selected_clients

    def start_federated_training(self, num_rounds):
        """Start the federated training process with proper weight aggregation"""
        try:
            # Load FL configuration
            with open("./server/fl_config.json", "r") as f:
                fl_config = json.load(f)
            
            model_type = fl_config["model_type"]
            client_fraction = fl_config["client_fraction"]
            
            # Initialize the appropriate model
            if model_type == "DiabetesMLP":
                model = DiabetesMLP(input_size=16)
            elif model_type == "FashionMNISTCNN":
                model = FashionMNISTCNN()
            elif model_type == "MNISTMLP":
                model = MNISTMLP()
            else:
                print(f"{STYLES.BG_RED}Invalid model type in config{STYLES.RESET}")
                return
            
            # Load initial weights
            initial_weights_path = "./server/models/initialized_model.pt"
            model.load_state_dict(torch.load(initial_weights_path))
            
            # Start training rounds
            for round_id in range(num_rounds):
                print(f"\n{STYLES.FG_CYAN}=== Starting Round {round_id + 1}/{num_rounds} ==={STYLES.RESET}")
                
                # Select clients for this round
                num_clients = len(self.clients)
                num_selected = max(1, int(client_fraction * num_clients))
                selected_clients = random.sample(list(self.clients.keys()), num_selected)
                print(f"Selected clients for this round: {selected_clients}")
                
                # Collect responses from all selected clients
                client_responses = []
                total_samples = 0
                
                for client_id in selected_clients:
                    try:
                        client_ip, client_port = self.clients[client_id]
                        channel = grpc.insecure_channel(f"{client_ip}:{client_port}")
                        stub = file_transfer_grpc.ClientStub(channel)
                        
                        # Send current global model to client
                        with open(f"./server/models/global_model_round_{round_id}.pt" if round_id > 0 else initial_weights_path, "rb") as f:
                            model_weights = f.read()
                        
                        response = stub.StartTraining(file_transfer_pb2.TrainingRequest(
                            round_id=round_id,
                            model_version="1.0",
                            model_weights=model_weights,
                            local_epochs=fl_config["num_epochs"]
                        ))
                        
                        if response.err_code == 0:
                            client_responses.append(response)
                            total_samples += response.samples_processed
                            print(f"Client {client_id} completed training successfully")
                        else:
                            print(f"{STYLES.BG_RED}Client {client_id} failed: {response.msg}{STYLES.RESET}")
                        
                        channel.close()
                    
                    except Exception as e:
                        print(f"{STYLES.BG_RED}Error with client {client_id}: {str(e)}{STYLES.RESET}")
                
                if not client_responses:
                    print(f"{STYLES.BG_RED}No successful client responses in round {round_id + 1}{STYLES.RESET}")
                    break
                
                # Aggregate client updates (Federated Averaging)
                avg_weights = {}
                for response in client_responses:
                    # Save client's model weights
                    client_model_path = f"./server/models/client_{response.client_id}_round_{round_id}.pt"
                    with open(client_model_path, "wb") as f:
                        f.write(response.updated_weights)
                    
                    # Load weights into model
                    client_model = model
                    client_model.load_state_dict(torch.load(client_model_path))
                    
                    # Calculate weight for this client (proportional to samples)
                    client_weight = response.samples_processed / total_samples
                    
                    # Add weighted contribution to average
                    for name, param in client_model.state_dict().items():
                        if name not in avg_weights:
                            avg_weights[name] = param * client_weight
                        else:
                            avg_weights[name] += param * client_weight
                
                # Update global model with averaged weights
                model.load_state_dict(avg_weights)
                
                # Save new global model
                global_model_path = f"./server/models/global_model_round_{round_id + 1}.pt"
                torch.save(model.state_dict(), global_model_path)

                if model_type == "DiabetesMLP":
                    path_to_server_test_data = "../server_data/diabetes_dataset.csv"
                    
                    # Calculate metrics
                    loss, acc = evaluate_diabetes_model(global_model_path, path_to_server_test_data)
                    print(f"Loss: {round(loss, 4)}")
                    print(f"Accuracy: {round(acc, 4)}%")
                
                elif model_type == "FashionMNISTCNN":
                    path_to_server_test_data = "../server_data/fashion_mnist_dataset.csv"
                    
                    # Calculate metrics
                    loss, acc = evaluate_fashion_mnist_model(global_model_path, path_to_server_test_data)
                    print(f"Loss: {round(loss, 4)}")
                    print(f"Accuracy: {round(acc, 4)}%")

                elif model_type == "MNISTMLP":
                    path_to_server_test_data = "../server_data/mnist_dataset.csv"
                    
                    # Calculate metrics
                    loss, acc = evaluate_mnist_model(global_model_path, path_to_server_test_data)
                    print(f"Loss: {round(loss, 4)}")
                    print(f"Accuracy: {round(acc, 4)}%")

                print(f"{STYLES.FG_GREEN}Round {round_id + 1} completed{STYLES.RESET}")
                print(f"  Total Samples: {total_samples}")
            
            print(f"\n{STYLES.FG_GREEN}Federated training completed after {num_rounds} rounds{STYLES.RESET}")
            
        except Exception as e:
            print(f"{STYLES.BG_RED}Error during federated training: {str(e)}{STYLES.RESET}")
            logging.error(f"Error during federated training: {str(e)}", exc_info=True)


class FLServerServicer(file_transfer_grpc.FLServerServicer):
    def __init__(self):
        self.clients = {}
        self.model = None

    def RegisterClient(self, request, context):
        client_id = request.id
        client_ip = request.ip
        client_port = request.port

        if client_id in fl_server.clients.keys():
            return file_transfer_pb2.ClientResponse(err_code=1, msg="Client already registered")

        fl_server.clients[client_id] = (client_ip, client_port)
        logging.info(f"Client {client_id} registered with address {client_ip}:{client_port}")
        return file_transfer_pb2.ClientResponse(err_code=0, msg="Client registered successfully")

    def DeregisterClient(self, request, context):
        client_id = request.id
        if client_id not in fl_server.clients:
            return file_transfer_pb2.ClientResponse(err_code=1, msg="Client not registered")

        del fl_server.clients[client_id]
        logging.info(f"Client {client_id} deregistered")
        return file_transfer_pb2.ClientResponse(err_code=0, msg="Client deregistered successfully")

    def TransferFile(self, request_iterator, context):
        try:
            # Try to get the first chunk
            try:
                first_chunk = next(request_iterator)
                client_id = first_chunk.id
                filename = first_chunk.filename
                logging.info(f"Receiving file: {filename}")
            except StopIteration:
                logging.error("No data received from client")
                return file_transfer_pb2.FileResponse(
                    err_code=1,
                    msg="No data received from client"
                )
            
            # Creating a separate directory for each client to store received files
            received_dir = Path(__file__).parent / 'received_files'
            received_dir.mkdir(exist_ok=True)
            received_dir = Path(__file__).parent / 'received_files' / str(client_id)
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
                msg=f"File {filename} received successfully"
            )
                
        except Exception as e:
            logging.error(f"Unexpected error during file transfer: {str(e)}", exc_info=True)
            return file_transfer_pb2.FileResponse(
                err_code=1,
                msg=f"Unexpected error: {str(e)}"
            )

# ============================= FUNCTIONS =============================
def menu():
    while True:
        clear_screen()
        print(f"{STYLES.FG_MAGENTA + STYLES.BOLD}Federated Learning Server{STYLES.RESET}")
        print()
        print(f"{STYLES.FG_YELLOW}Choose an option: {STYLES.RESET}")
        print("  1. Transfer File")
        print("  2. Exit")
        print("  3. Initialize Federated Learning")
        print("  4. Start Federated Training")
        print()
        choice = input(f"{STYLES.FG_YELLOW}Enter your choice: {STYLES.RESET}")
        if choice == "1":
            file_path = input(f"{STYLES.FG_YELLOW}Enter the file path: {STYLES.RESET}")
            if not Path(file_path).is_file():
                print(f"{STYLES.BG_RED}File not found. Please try again.{STYLES.RESET}")
            client_id = input(f"{STYLES.FG_YELLOW}Enter the client ID: {STYLES.RESET}").strip()
            try:
                if client_id.lower() == "all":
                    all_client_ids = fl_server.clients.keys()
                    for client_id in all_client_ids:
                        fl_server.send_file_to_client(file_path, client_id)
                else:
                    client_id = int(client_id)
                    fl_server.send_file_to_client(file_path, client_id)
            except ValueError:
                print(f"{STYLES.BG_RED}Invalid client ID. Please try again.{STYLES.RESET}")
                wait_for_enter()
                continue
            wait_for_enter()
        elif choice == "2":
            break
        elif choice == "3":
            num_epochs = int(input(f"{STYLES.FG_YELLOW}Enter number of epochs: {STYLES.RESET}"))
            learning_rate = float(input(f"{STYLES.FG_YELLOW}Enter learning rate: {STYLES.RESET}"))
            optimizer = input(f"{STYLES.FG_YELLOW}Enter optimizer (SGD/Adam): {STYLES.RESET}")
            batch_size = int(input(f"{STYLES.FG_YELLOW}Enter batch size: {STYLES.RESET}"))
            model_type = input(f"{STYLES.FG_YELLOW}Enter model type (DiabetesMLP/FashionMNISTCNN/MNISTMLP): {STYLES.RESET}")
            client_fraction = float(input(f"{STYLES.FG_YELLOW}Enter client fraction: {STYLES.RESET}"))

            # Validate inputs
            if optimizer not in ["SGD", "Adam"]:
                print(f"{STYLES.BG_RED}Invalid optimizer. Please enter SGD or Adam.{STYLES.RESET}")
                wait_for_enter()
                continue
                
            if model_type not in ["DiabetesMLP", "FashionMNISTCNN", "MNISTMLP"]:
                print(f"{STYLES.BG_RED}Invalid model type. Please enter DiabetesMLP, FashionMNISTCNN, or MNISTMLP.{STYLES.RESET}")
                wait_for_enter()
                continue
                
            if client_fraction <= 0 or client_fraction > 1:
                print(f"{STYLES.BG_RED}Invalid client fraction. Value must be between 0.0 and 1.0.{STYLES.RESET}")
                wait_for_enter()
                continue
            
            if len(fl_server.clients) == 0:
                print(f"{STYLES.BG_RED}No clients registered. Please wait for clients to connect.{STYLES.RESET}")
                wait_for_enter()
                continue
            
            try:
                selected_clients = fl_server.initialize_fl(num_epochs, learning_rate, optimizer, batch_size, model_type, client_fraction)
                print(f"{STYLES.FG_GREEN}Federated learning initialized successfully!{STYLES.RESET}")
                print(f"{STYLES.FG_CYAN}Selected {len(selected_clients)}/{len(fl_server.clients)} clients for training.{STYLES.RESET}")
                print(f"{STYLES.FG_CYAN}Client IDs: {', '.join(map(str, selected_clients))}{STYLES.RESET}")
            except Exception as e:
                print(f"{STYLES.BG_RED}Error initializing federated learning: {str(e)}{STYLES.RESET}")
            
            wait_for_enter()

        elif choice == "4":
            if not Path("./server/fl_config.json").exists():
                print(f"{STYLES.BG_RED}FL not initialized. Please initialize first.{STYLES.RESET}")
                wait_for_enter()
                continue
                
            if len(fl_server.clients) == 0:
                print(f"{STYLES.BG_RED}No clients registered. Please wait for clients to connect.{STYLES.RESET}")
                wait_for_enter()
                continue
                
            num_rounds = int(input(f"{STYLES.FG_YELLOW}Enter number of training rounds: {STYLES.RESET}"))
            if num_rounds <= 0:
                print(f"{STYLES.BG_RED}Number of rounds must be positive{STYLES.RESET}")
                wait_for_enter()
                continue
                
            fl_server.start_federated_training(num_rounds)
            wait_for_enter()          

        else:
            print(f"{STYLES.BG_RED}Invalid choice. Please try again.{STYLES.RESET}")
            wait_for_enter()


# ============================= MAIN =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning Server')
    parser.add_argument("--port", type=int, default=50051, help="Port Number for FL Server")
    args = parser.parse_args()

    my_port = args.port
    my_ip = get_ip()

    global fl_server
    fl_server = FLServer(my_port, my_ip)

    logging.basicConfig(
        filename=f"./server/logs/fl_server.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    menu()