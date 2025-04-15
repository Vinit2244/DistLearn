# ============================= IMPORTS =============================
import os
import sys
import grpc
import time
import json
import torch
import random
import logging
import argparse
import subprocess
import numpy as np
import seaborn as sns
from pathlib import Path
import concurrent.futures
from concurrent import futures
import matplotlib.pyplot as plt
from google.protobuf import empty_pb2

server_folder_abs_path = Path(__file__).parent.resolve()

sys.path.append(str(server_folder_abs_path / "../generated"))
import file_transfer_pb2
import file_transfer_pb2_grpc as file_transfer_grpc

sys.path.append(str(server_folder_abs_path / "../utils"))
from utils import clear_screen, wait_for_enter, get_ip, STYLES, CHUNK_SIZE

# Importing models to be trained
sys.path.append(str(server_folder_abs_path / "../models"))
from DiabetesMLP import DiabetesMLP, evaluate_model as evaluate_diabetes_model
from FashionMNISTCNN import FashionMNISTCNN, evaluate_model as evaluate_fashion_mnist_model
from MNISTMLP import MNISTMLP, evaluate_model as evaluate_mnist_model

sns.set_theme(style="whitegrid", context="talk")

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""


# ============================= GLOBALS =============================
training_algos = ["FedSGD", "FedAvg", "FedAdp", "FedModCS"]
optimizers = ["SGD", "Adam"]
model_types = ["DiabetesMLP", "FashionMNISTCNN", "MNISTMLP"]
losses = {}
accuracies = {}
timestamps = {}
fedmodcs_clients_selected_each_round = {}
ALPHA = 2

# ============================= CLASSES =============================
class ClientSessionManager:
    def __init__(self, client_id, client_ip, client_port):
        self.client_id = client_id
        self.client_ip = client_ip
        self.client_port = client_port
        self.channel = None

    def __enter__(self):
        if encrypt:
            trusted_certs_path = server_folder_abs_path / "certs/ca.crt"
            with open(trusted_certs_path, "rb") as f:
                trusted_certs = f.read()

            server_key_path = server_folder_abs_path / "certs/server.key"
            with open(server_key_path, "rb") as f:
                server_key = f.read()

            server_certificate_path = server_folder_abs_path / "certs/server.crt"
            with open(server_certificate_path, "rb") as f:
                server_certificate = f.read()
            
            credentials = grpc.ssl_channel_credentials(
                root_certificates=trusted_certs,
                private_key=server_key,
                certificate_chain=server_certificate
            )
            self.channel = grpc.secure_channel(f"{self.client_ip}:{self.client_port}", credentials)
        else:
            self.channel = grpc.insecure_channel(f"{self.client_ip}:{self.client_port}")
        client_stub = file_transfer_grpc.ClientStub(self.channel)
        return client_stub

    def __exit__(self, exc_type, exc_value, traceback):
        if self.channel:
            self.channel.close()


class FLServer:
    def __init__(self, port, ip):
        self.my_port = port
        self.my_ip = ip
        self.clients = {}
        self.server_instance = None

        self.register_with_consul()
        self.start_server()
        self.initially_selected_clients = []
    
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
        server_json_path = server_folder_abs_path / "fl_server.json"
        with open(server_json_path, "w") as json_file:
            json.dump(data, json_file)
        logging.info("FL service definition file created")

        # Registering the service with consul
        command = ["consul", "services", "register", str(server_folder_abs_path / "fl_server.json")]
        try:
            subprocess.run(command, check=True)
            logging.info("FL service registered with consul")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error registering service: {e}")
            logging.error("Error registering FL service")
            sys.exit(1)

    def start_server(self):
        # Starting FL server
        fl_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        file_transfer_grpc.add_FLServerServicer_to_server(FLServerServicer(), fl_server)

        if encrypt:
            root_certificate_path = server_folder_abs_path / "certs/ca.crt"
            with open(root_certificate_path, "rb") as f:
                root_certificate = f.read()

            certificate_chain_path = server_folder_abs_path / "certs/server.crt"
            with open(certificate_chain_path, "rb") as f:
                certificate_chain = f.read()
            
            private_key_path = server_folder_abs_path / "certs/server.key"
            with open(private_key_path, "rb") as f:
                private_key = f.read()

            server_credentials = grpc.ssl_server_credentials(
                [(private_key, certificate_chain)],
                root_certificates=root_certificate
            )

            fl_server.add_secure_port(f"{self.my_ip}:{self.my_port}", server_credentials)
        else:
            fl_server.add_insecure_port(f"{self.my_ip}:{self.my_port}")
        fl_server.start()
        logging.info(f"FL server started at address {self.my_ip}:{self.my_port}")
        self.server_instance = fl_server

    def terminate_server(self):
        if self.server_instance is None:
            logging.error("Server not running")
            return
        self.server_instance.stop(0)
        self.server_instance = None
        logging.info("FL server terminated")

    def send_file_to_client(self, file_path, client_id):
        if client_id not in self.clients.keys():
            print(f"{STYLES.BG_RED}Client not registered. Please try again.{STYLES.RESET}")
            return 1
        client_ip, client_port = self.clients[client_id]
        with ClientSessionManager(client_id, client_ip, client_port) as stub:
            filename = file_path.name
            file_size = file_path.stat().st_size
            logging.info(f"Starting to send file: {filename} (size: {file_size} bytes)")
            
            def request_iterator():
                with open(file_path, "rb") as f:
                    chunk_number = 0
                    while True:
                        chunk = f.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        chunk_number += 1
                        # logging.info(f"Sending chunk {chunk_number} of {len(chunk)} bytes")
                        yield file_transfer_pb2.FileChunk(
                            filename=filename,
                            id=-1,
                            chunk=chunk,
                            is_last_chunk=len(chunk) < CHUNK_SIZE
                        )
            
            response = stub.TransferFile(request_iterator())
            if response.err_code == 0:
                logging.info(f"File transfer successful: {response.msg}")
                # print(f"{STYLES.FG_GREEN}Success: File sent successfully!{STYLES.RESET}")
            else:
                logging.error(f"File transfer failed: {response.msg}")
                print(f"{STYLES.BG_RED + STYLES.FG_WHITE}Error: {response.msg}{STYLES.RESET}")
                return 1

    def initialize_fl(self, training_algo, num_epochs, learning_rate, optimizer, batch_size, model_type, client_fraction, t_round, t_final, lr_decay=0.995):
        fl_config_server = {
            "training_algo":    training_algo,
            "num_epochs":       num_epochs,
            "learning_rate":    learning_rate,
            "optimizer":        optimizer,
            "batch_size":       batch_size,
            "model_type":       model_type,
            "lr_decay":         lr_decay,
            "t_round":        t_round,
            "t_final":        t_final
        }

        fl_config_client = {
            "num_epochs":       num_epochs,
            "learning_rate":    learning_rate,
            "optimizer":        optimizer,
            "batch_size":       batch_size,
            "model_type":       model_type,
            "lr_decay":         lr_decay
        }

        config_path = server_folder_abs_path / "fl_config_server.json"
        with open(config_path, "w") as f:
            json.dump(fl_config_server, f)

        logging.info(f"FL Configuration (Server) saved at {config_path}")

        config_path = server_folder_abs_path / "fl_config_client.json"
        with open(config_path, "w") as f:
            json.dump(fl_config_client, f)
        
        logging.info(f"FL Configuration (Client) saved at {config_path}")

        model = None
        if model_type == model_types[0]:
            model = DiabetesMLP(input_size=16)
        elif model_type == model_types[1]:
            model = FashionMNISTCNN()
        elif model_type == model_types[2]:
            model = MNISTMLP()
        else:
            logging.error(f"Invalid model type: {model_type}")
            return

        server_models_dir = server_folder_abs_path / "models"
        server_models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = server_models_dir / "global_model_round_0.pt"
        torch.save(model.state_dict(), model_path)
        logging.info(f"Initialized model saved at {model_path}")

        # Selecting random clients based on client fraction
        num_clients = len(self.clients)
        num_selected = max(1, int(client_fraction * num_clients))
        selected_clients = random.sample(list(self.clients.keys()), num_selected)
        
        print(f"{STYLES.FG_CYAN}Selected {num_selected}/{num_clients} clients for training.{STYLES.RESET}")
        print(f"{STYLES.FG_CYAN}Client IDs: {selected_clients}{STYLES.RESET}")
        logging.info(f"Selected {num_selected}/{num_clients} clients for training: {selected_clients}")

        # Send config and model to each client
        for client_id in selected_clients:
            logging.info(f"Sending configuration and model to client {client_id}")
            self.send_file_to_client(config_path, client_id)
            self.send_file_to_client(model_path, client_id)
        self.initially_selected_clients = selected_clients

    def start_federated_training(self, num_rounds, client_fraction):
        try:
            server_config_path = server_folder_abs_path / "fl_config_server.json"
            with open(server_config_path, "r") as f:
                fl_config = json.load(f)
 
            training_algo   = fl_config["training_algo"]
            model_type      = fl_config["model_type"]
            lr              = fl_config["learning_rate"]
            optimizer       = fl_config["optimizer"]
            t_round         = fl_config["t_round"]
            t_final         = fl_config["t_final"]
            lr_decay       = fl_config["lr_decay"]

            # Initialize the appropriate model
            if model_type == model_types[0]:
                model = DiabetesMLP(input_size=16)
            elif model_type == model_types[1]:
                model = FashionMNISTCNN()
            elif model_type == model_types[2]:
                model = MNISTMLP()
            else:
                print(f"{STYLES.BG_RED}Invalid model type in config{STYLES.RESET}")
                return

            # Load initial weights
            initial_weights_path = server_folder_abs_path / "models/global_model_round_0.pt"
            torch.load(initial_weights_path)

            if model_type == model_types[2]:
                smoothed_theta_arr = np.zeros(len(self.clients))

            start_time_total = time.time()
            cumulative_time = 0
            for round_id in range(num_rounds):
                print(f"\n{STYLES.FG_CYAN}=== Starting Round {round_id + 1}/{num_rounds} ==={STYLES.RESET}")
                start_time_round = time.time()

                # Function to send training requests to clients
                def train_client(client_id):
                    try:
                        client_ip, client_port = self.clients[client_id]
                        
                        # Send current global model
                        with ClientSessionManager(client_id, client_ip, client_port) as stub:
                            model_weights_path = server_folder_abs_path / f"models/global_model_round_{round_id}.pt"
                            self.send_file_to_client(model_weights_path, client_id)
                            logging.info(f"Sent model weights to client {client_id}")                     

                            response = stub.StartTraining(file_transfer_pb2.TrainingRequest(
                                round_id=round_id,
                                model_version="1.0",
                                local_epochs=fl_config["num_epochs"],
                                model_path = str(model_weights_path.name),
                                model_type = model_type
                            ))

                            return client_id, response

                    except Exception as e:
                        print(f"{STYLES.BG_RED}Error with client {client_id}: {str(e)}{STYLES.RESET}")
                        return client_id, None
                    
                # Select clients
                num_clients = len(self.initially_selected_clients)
                num_selected = max(1, int(client_fraction * num_clients))
                selected_clients = random.sample(self.initially_selected_clients, num_selected)
                print(f"{STYLES.FG_CYAN}Selected {num_selected}/{num_clients} clients for training.{STYLES.RESET}")


                file_size_bytes = initial_weights_path.stat().st_size
                file_size_mb = file_size_bytes / (1024 * 1024)   

                client_responses = []
                total_samples = 0
                
                # FedModCS
                if training_algo == training_algos[3]:
                    # If FedModCS, request resource info from all clients
                    client_resource_info = {}
                    logging.info("Training algorithm is FedModCS: Requesting resource info from all clients.")
                    for client_id in selected_clients:
                        try:
                            client_ip, client_port = self.clients[client_id]
                            with ClientSessionManager(client_id, client_ip, client_port) as stub:
                                response = stub.GetResourceInfo(empty_pb2.Empty())
                                info = {
                                    "dataset_size": response.dataset_size,
                                    "cpu_speed_factor": response.cpu_speed_factor,
                                    "network_bandwidth": response.network_bandwidth,
                                    "has_gpu": response.has_gpu
                                }
                                client_resource_info[client_id] = info
                                logging.info(f"Client {client_id} resource info: {info}")
                        except Exception as e:
                            logging.error(f"Failed to get resource info from client {client_id}: {e}")
                            
                    selected_clients = self.select_clients_fedmodcs(client_resource_info, round_id, selected_clients, file_size_mb, t_round, t_final)
                    num_selected = len(selected_clients)
                    print(f"{STYLES.FG_CYAN}Selected {num_selected}/{num_clients} clients for training after FedModCS.{STYLES.RESET}")
                
                    if num_selected == 0:
                        print(f"{STYLES.BG_RED}No clients selected for this round according to FedMODCS. Going for random selection...{STYLES.RESET}")
                        num_clients = len(self.clients)
                        num_selected = max(1, int(client_fraction * num_clients))
                        selected_clients = random.sample(list(self.clients.keys()), num_selected)

                # Parallelize client training
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_selected) as executor:
                    future_to_client = {executor.submit(train_client, client_id): client_id for client_id in selected_clients}

                    for future in concurrent.futures.as_completed(future_to_client):
                        client_id, response = future.result()
                        if response and response.err_code == 0:
                            client_responses.append(response)
                            total_samples += response.samples_processed
                            print(f"Client {client_id} completed training successfully")
                        else:
                            print(f"{STYLES.BG_RED}Client {client_id} failed or did not respond{STYLES.RESET}")

                if not client_responses:
                    print(f"{STYLES.BG_RED}No successful client responses in round {round_id + 1}{STYLES.RESET}")
                    break

                weights_of_all_clients = []
                weight_updates_of_all_clients = []
                num_samples_arr = []

                model_weights_path = server_folder_abs_path / f"models/global_model_round_{round_id - 1}.pt" if round_id > 0 else initial_weights_path
                global_state_dict = torch.load(model_weights_path, map_location="cpu")
                global_w_prev = torch.cat([v.flatten() for v in global_state_dict.values()]).numpy().astype(np.float32)

                response_order = []
                for response in client_responses:
                    client_model_path = server_folder_abs_path / f"received_files/{response.client_id}/round_{round_id}_trained.pt"
                    client_state_dict = torch.load(client_model_path, map_location="cpu")
                    client_weights = torch.cat([v.flatten() for v in client_state_dict.values()]).numpy().astype(np.float32)
                    weights_of_all_clients.append(client_weights)
                    weight_updates_of_all_clients.append(client_weights - global_w_prev)
                    num_samples_arr.append(response.samples_processed)
                    response_order.append(response.client_id)
                
                weight_updates_of_all_clients = np.array(weight_updates_of_all_clients)
                num_samples_arr = np.array(num_samples_arr)

                # FedAdp
                if training_algo == training_algos[2]:
                    # Calculate decayed learning rate
                    lr = fl_config["learning_rate"] * (lr_decay ** round_id)

                    theta_arr = self.get_theta_arr(weight_updates_of_all_clients, num_samples_arr, round_id, initial_weights_path, lr)
                    if round_id == 0:
                        smoothed_theta_arr = theta_arr
                    else:
                        smoothed_theta_arr = (round_id * smoothed_theta_arr + theta_arr) / (round_id + 1)
                    psi_arr = self.get_psi_arr(smoothed_theta_arr, num_samples_arr)
                    psi_list = psi_arr.tolist()

                    # File path
                    json_file = 'psi_data.json'

                    # Load existing data or start fresh
                    if os.path.exists(json_file):
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                    else:
                        data = []

                    # Append new data
                    # Sort response order and correspondingly sort psi_list
                    sorted_indices = sorted(range(len(response_order)), key=lambda k: response_order[k])
                    response_order = [response_order[i] for i in sorted_indices]
                    psi_list = [psi_list[i] for i in sorted_indices]
                    
                    smoothed_theta_list = smoothed_theta_arr.tolist()
                    smoothed_theta_list = [smoothed_theta_list[i] * 360 / np.pi for i in sorted_indices]

                    data.append({
                        "response_order": response_order,
                        "psi_list": psi_list,
                        "smoothed_theta_list": smoothed_theta_list
                    })

                    # Write back to JSON file
                    with open(json_file, 'w') as f:
                        json.dump(data, f, indent=2)

                    global_weight_update = np.zeros_like(global_w_prev)

                    # Weighing weights
                    for idx, client_weight in enumerate(weights_of_all_clients):
                        global_weight_update += client_weight * psi_arr[idx]
                    updated_global_weights = global_weight_update

                # FedSGD and FedAvg
                else:
                    # Aggregate client updates (Federated Averaging)
                    global_weight_update = np.zeros_like(global_w_prev)

                    # Weighing weights
                    for idx, client_weight in enumerate(weights_of_all_clients):
                        global_weight_update += client_weight * (num_samples_arr[idx] / total_samples)
                    updated_global_weights = global_weight_update

                updated_state_dict = {}
                offset = 0
                for name, param in global_state_dict.items():
                    numel = param.numel()
                    updated_param = updated_global_weights[offset:offset + numel].reshape(param.shape)
                    updated_state_dict[name] = torch.tensor(updated_param, dtype=param.dtype)
                    offset += numel
                
                # Update global model with averaged weights
                model.load_state_dict(updated_state_dict)

                # Save new global model
                global_model_path = server_folder_abs_path / f"models/global_model_round_{round_id+1}.pt"
                torch.save(model.state_dict(), global_model_path)

                # Evaluate new model
                if model_type == model_types[0]:
                    path_to_server_test_data = server_folder_abs_path / "data/diabetes_dataset.csv"
                    loss, acc = evaluate_diabetes_model(global_model_path, path_to_server_test_data)

                elif model_type == model_types[1]:
                    path_to_server_test_data = server_folder_abs_path / "data/fashion_mnist_dataset.csv"
                    loss, acc = evaluate_fashion_mnist_model(global_model_path, path_to_server_test_data)

                elif model_type == model_types[2]:
                    path_to_server_test_data = server_folder_abs_path / "data/mnist_dataset.csv"
                    loss, acc = evaluate_mnist_model(global_model_path, path_to_server_test_data)

                else:
                    print(f"{STYLES.BG_RED}Invalid model type in config{STYLES.RESET}")
                    return

                print(f"{STYLES.FG_YELLOW}Loss: {round(loss, 4)}")
                print(f"Accuracy: {round(acc, 4)}%{STYLES.RESET}")

                if training_algo == training_algos[3]:
                    model_name = f"{training_algo}_{model_type}_{optimizer}_{t_round}"
                else:
                    model_name = f"{training_algo}_{model_type}_{optimizer}"

                # Store loss and accuracy for this round for plotting after training
                if losses.get(model_name) is None:
                    losses[model_name] = []
                if accuracies.get(model_name) is None:
                    accuracies[model_name] = []
                if timestamps.get(model_name) is None:
                    timestamps[model_name] = []
                if fedmodcs_clients_selected_each_round.get(model_name) is None:
                    fedmodcs_clients_selected_each_round[model_name] = []

                losses[model_name].append(loss)
                accuracies[model_name].append(acc)
                fedmodcs_clients_selected_each_round[model_name].append(len(selected_clients))

                if training_algo == training_algos[3]:
                    fedmodcs_time = time.time() - start_time_round
                    cumulative_time += fedmodcs_time
                    timestamps[model_name].append(cumulative_time)

                print(f"{STYLES.FG_GREEN}Round {round_id + 1} completed{STYLES.RESET}")
                print(f"  Total Samples: {total_samples}")
                print(f"  Time taken for round {round_id + 1}: {time.time() - start_time_round:.2f} seconds")
                # make_plots(len(self.clients), model_name, client_fraction, server_folder_abs_path / "metrics.json")

            print(f"\n{STYLES.FG_GREEN}Federated training completed after {num_rounds} rounds{STYLES.RESET}")
            print(f"{STYLES.FG_YELLOW}Total time taken: {time.time() - start_time_total:.2f} seconds{STYLES.RESET}")

            if training_algo == training_algos[3]:
                make_plots_fedmodcs(len(self.clients), model_name, client_fraction, server_folder_abs_path / "metrics.json")
            else:
                # Plotting loss and accuracy
                make_plots(len(self.clients), model_name, client_fraction, server_folder_abs_path / "metrics.json")

        except Exception as e:
            print(f"{STYLES.BG_RED}Error during federated training: {str(e)}{STYLES.RESET}")
            logging.error(f"Error during federated training: {str(e)}", exc_info=True)

    def gompertz_func(self, theta_arr, alpha=ALPHA):
        power_2 = -1 * alpha * (theta_arr - 1)
        power_1 = -1 * (np.exp(power_2))
        return alpha * (1 - np.exp(power_1))

    def get_psi_arr(self, theta_arr, num_samples_arr):
        # Psi array is weights for each client
        Nr = num_samples_arr * np.exp(self.gompertz_func(theta_arr))
        Dr = np.sum(Nr)
        return Nr / Dr

    def get_theta_arr(self, weight_updates_arr, num_samples_arr, curr_round_id, initial_weights_path, lr):
        gradients_arr = - 1 * weight_updates_arr / lr
        global_gradient = np.zeros_like(weight_updates_arr[0])
        total_samples = np.sum(num_samples_arr)
        for i, client_gradient in enumerate(gradients_arr):
            global_gradient += (num_samples_arr[i] / total_samples) * client_gradient

        # Theta = arccos((gradients_arr . global_gradient) / (||gradients_arr|| * ||global_gradient||))
        # where . is dot product and ||*|| is L2 norm
        Nr_arr = np.array([np.dot(gradients_arr[i].flatten(), global_gradient.flatten()) for i in range(gradients_arr.shape[0])])
        Dr_arr = np.array([np.linalg.norm(global_gradient) * np.linalg.norm(gradients_arr[i]) for i in range(gradients_arr.shape[0])])
        theta_arr = np.arccos(Nr_arr / Dr_arr)
        return theta_arr
    
    def select_clients_fedmodcs(self, client_resource_info, round_id, randomly_selected_clients, model_size, t_round, t_final):
        T_cs = 0.0  
        T_agg = 0.0  
        T_round = t_round 
        T_final = t_final
        T_s_empty = 0.0  
        
        K_prime = randomly_selected_clients.copy()  
        K_prime.sort()  
        
        logging.info(f"Round {round_id}: Initial random selection K': {K_prime}")
        
        # Setup client-specific parameters from resource info
        t_UL = {}  # Upload times
        t_UD = {}  # Update times
        
        for client_id in K_prime:
            if client_id in client_resource_info:
                info = client_resource_info[client_id]
                bandwidth = info["network_bandwidth"]
                dataset_size = info["dataset_size"]
                cpu_factor = info["cpu_speed_factor"]

                bytes_per_sec = (bandwidth * 1000000) / 8
                t_UL[client_id] = model_size / (bytes_per_sec + 1e-6)  # small epsilon to avoid div by 0

                # t_UL[client_id] = model_size / (bandwidth + 0.1)
                t_UD[client_id] = dataset_size / cpu_factor
                if info["has_gpu"]:
                    t_UD[client_id] *= 0.5  # GPU clients are faster   

            else:
                # Default values if resource info is missing
                t_UL[client_id] = 1.0
                t_UD[client_id] = 2.0                 
        
        # Initialization for the algorithm
        S = []
        T_s = T_s_empty
        Theta = 0
        
        # Clone K_prime to avoid modifying the original during iteration
        K_prime_remaining = K_prime.copy()
        
        # Client selection process
        while len(K_prime_remaining) > 0:            
            # Find client with maximum value according to the formula
            max_value = -float("inf")
            max_client = None
            
            for k in K_prime_remaining:
                T_S_union_k = T_s + model_size / (client_resource_info[k]["network_bandwidth"] + 0.1)
                denominator = T_S_union_k - T_s + t_UL[k] + max(0, t_UD[k] - Theta)
                if denominator > 0:  # Avoid division by zero
                    value = 1 / denominator
                    if value > max_value:
                        max_value = value
                        max_client = k
            
            if max_client is None:
                break  # No valid client found
                
            # Remove selected client from K'
            x = max_client
            K_prime_remaining.remove(x)
            
            # Update Theta'
            Theta_prime = Theta + t_UL[x] + max(0, t_UD[x] - Theta)

            T_S_union_x = T_s + model_size / (client_resource_info[x]["network_bandwidth"] + 0.1)
            
            # Calculate total time
            t = T_cs + T_S_union_x + Theta_prime + T_agg
            
            # Check if adding this client keeps us within round time
            if t < T_round:
                Theta = Theta_prime
                S.append(x)
                # T_s = T_S_union_x

                logging.info(f"Added client {x} to selection. New Theta: {Theta}, Time: {t}")
            else:
                logging.info(f"Client {x} would exceed round time ({t} > {T_round}). Skipping.")
        
        logging.info(f"Final client selection S: {S}")
        return S


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
            received_dir = server_folder_abs_path / "received_files" / str(client_id)
            received_dir.mkdir(parents=True, exist_ok=True)
            
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
                        logging.info(f"Received last chunk {total_chunks} of {len(chunk.chunk)} bytes")
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

# ============================= FUNCTIONS =============================
def menu():
    while True:
        clear_screen()
        print(f"{STYLES.FG_MAGENTA + STYLES.BOLD}Federated Learning Server{STYLES.RESET}")
        print()
        print(f"{STYLES.FG_YELLOW}Choose an option: {STYLES.RESET}")
        print("  1. Transfer File")
        print("  2. Initialize Federated Learning")
        print("  3. Start Federated Training")
        print("  4. Clear Loss and Accuracy Data")
        print("  5. Exit")
        print()
        choice = input(f"{STYLES.FG_YELLOW}Enter your choice: {STYLES.RESET}")
        
        if choice == "1":
            file_path = Path(input(f"{STYLES.FG_YELLOW}Enter the file path: {STYLES.RESET}"))
            if not file_path.is_file():
                print(f"{STYLES.BG_RED}File not found. Please try again.{STYLES.RESET}")
            client_id = input(f"{STYLES.FG_YELLOW}Enter the client ID: {STYLES.RESET}").strip()
            try:
                if client_id.lower() == "all":
                    all_client_ids = fl_server.clients.keys()
                    if len(all_client_ids) == 0:
                        print(f"{STYLES.BG_RED}No clients registered. Please wait for clients to connect.{STYLES.RESET}")
                        wait_for_enter()
                        continue
                    for client_id in all_client_ids:
                        if fl_server.send_file_to_client(file_path, client_id) == 1:
                            print(f"{STYLES.FG_GREEN}File sent to client ID: {client_id}{STYLES.RESET}")
                else:
                    client_id = int(client_id)
                    fl_server.send_file_to_client(file_path, client_id)
            except ValueError:
                print(f"{STYLES.BG_RED}Invalid client ID. Please try again.{STYLES.RESET}")
                wait_for_enter()
                continue
            wait_for_enter()
                
        elif choice == "2":
            # Select Algorithm
            print(f"{STYLES.FG_YELLOW}Enter the training algorithm: {STYLES.RESET}")
            for i, algo in enumerate(training_algos, 1):
                print(f"  {i}. {algo}")
            training_algo = int(input(f"{STYLES.FG_YELLOW}Enter your choice: {STYLES.RESET}"))
            if training_algo not in range(1, len(training_algos) + 1):
                print(f"{STYLES.BG_RED}Invalid choice. Please try again.{STYLES.RESET}")
                wait_for_enter()
                continue

            # Number of epochs
            if training_algo - 1 == 0:
                num_epochs = 1
            else:
                num_epochs = int(input(f"{STYLES.FG_YELLOW}Enter number of epochs: {STYLES.RESET}"))
            
            if training_algo - 1 == 3:
                # Ask for t_round
                t_round = int(input(f"{STYLES.FG_YELLOW}Enter t_round (in seconds): {STYLES.RESET}"))
                if t_round <= 0:
                    print(f"{STYLES.BG_RED}t_round must be positive{STYLES.RESET}")
                    wait_for_enter()
                    continue
                # Ask for t_final
                t_final = int(input(f"{STYLES.FG_YELLOW}Enter t_final (in seconds): {STYLES.RESET}"))
                if t_final <= 0:
                    print(f"{STYLES.BG_RED}t_final must be positive{STYLES.RESET}")
                    wait_for_enter()
                    continue
            
            else:
                t_round = None
                t_final = None

            # Learning Rate
            learning_rate = float(input(f"{STYLES.FG_YELLOW}Enter learning rate: {STYLES.RESET}"))
            
            # Optimizer
            print(f"{STYLES.FG_YELLOW}Enter the optimizer: {STYLES.RESET}")
            for i, opt in enumerate(optimizers, 1):
                print(f"  {i}. {opt}")
            optimizer = int(input(f"{STYLES.FG_YELLOW}Enter optimizer: {STYLES.RESET}"))
            if optimizer not in range(1, len(optimizers) + 1):
                print(f"{STYLES.BG_RED}Invalid choice. Please try again.{STYLES.RESET}")
                wait_for_enter()
                continue

            # Batch size
            batch_size = int(input(f"{STYLES.FG_YELLOW}Enter batch size: {STYLES.RESET}"))
            if batch_size <= 0:
                print(f"{STYLES.BG_RED}Batch size must be positive{STYLES.RESET}")
                wait_for_enter()
                continue
            
            # Model type
            print(f"{STYLES.FG_YELLOW}Enter the model type: {STYLES.RESET}")
            for i, model in enumerate(model_types, 1):
                print(f"  {i}. {model}")
            model_type = int(input(f"{STYLES.FG_YELLOW}Enter model type: {STYLES.RESET}"))
            if model_type not in range(1, len(model_types) + 1):
                print(f"{STYLES.BG_RED}Invalid choice. Please try again.{STYLES.RESET}")
                wait_for_enter()
                continue

            # Client fraction
            client_fraction = 1.0
            
            # Check if clients are registered or not
            if len(fl_server.clients) == 0:
                print(f"{STYLES.BG_RED}No clients registered. Please wait for clients to connect.{STYLES.RESET}")
                wait_for_enter()
                continue
            
            try:
                fl_server.initialize_fl(training_algos[training_algo - 1], num_epochs, learning_rate, optimizers[optimizer - 1], 
                                                           batch_size, model_types[model_type - 1], client_fraction, t_round, t_final)
                print(f"{STYLES.FG_GREEN}Federated learning initialized successfully!{STYLES.RESET}")
            except Exception as e:
                print(f"{STYLES.BG_RED}Error initializing federated learning: {str(e)}{STYLES.RESET}")
            
            wait_for_enter()

        elif choice == "3":
            config_server_file_path = server_folder_abs_path / "fl_config_server.json"
            if not config_server_file_path.exists():
                print(f"{STYLES.BG_RED}FL not initialized. Please initialize first.{STYLES.RESET}")
                wait_for_enter()
                continue
                
            if len(fl_server.clients) == 0:
                print(f"{STYLES.BG_RED}No clients registered. Please wait for clients to connect.{STYLES.RESET}")
                wait_for_enter()
                continue
            
            # Number of rounds
            num_rounds = int(input(f"{STYLES.FG_YELLOW}Enter number of training rounds (Epochs at server side): {STYLES.RESET}"))
            if num_rounds <= 0:
                print(f"{STYLES.BG_RED}Number of rounds must be positive{STYLES.RESET}")
                wait_for_enter()
                continue
            
            # Client fraction
            client_fraction = float(input(f"{STYLES.FG_YELLOW}Enter client fraction: {STYLES.RESET}"))
            if client_fraction <= 0 or client_fraction > 1:
                print(f"{STYLES.BG_RED}Invalid client fraction. Value must be between 0.0 and 1.0.{STYLES.RESET}")
                wait_for_enter()
                continue
                
            fl_server.start_federated_training(num_rounds, client_fraction)
            wait_for_enter()    
            
        elif choice == "4":
            # Clear loss and accuracy arrays
            losses.clear()
            accuracies.clear()
            timestamps.clear()
            fedmodcs_clients_selected_each_round.clear()
            print(f"{STYLES.FG_GREEN}Loss and accuracy arrays cleared!{STYLES.RESET}")
            wait_for_enter()      
        
        elif choice == "5":
            break

        else:
            print(f"{STYLES.BG_RED}Invalid choice. Please try again.{STYLES.RESET}")
            wait_for_enter()


def make_plots(num_clients, model_name, client_fraction, json_output_path):
    server_config_path = server_folder_abs_path / "fl_config_server.json"
    with open(server_config_path, "r") as f:
        fl_config = json.load(f)

    training_algo   = fl_config["training_algo"]
    num_epochs      = fl_config["num_epochs"]
    learning_rate   = fl_config["learning_rate"]
    optimizer       = fl_config["optimizer"]
    batch_size      = fl_config["batch_size"]
    model_type      = fl_config["model_type"]
    lr_decay        = fl_config["lr_decay"]

    metric_plots_dir = server_folder_abs_path / "metric_plots"
    metric_plots_dir.mkdir(parents=True, exist_ok=True)

    loss_list = losses[model_name]
    accuracy_list = accuracies[model_name]

    # Save loss and accuracy plots for current model
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.lineplot(ax=axes[0], x=range(len(loss_list)), y=loss_list, label="Loss", color="steelblue")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    sns.lineplot(ax=axes[1], x=range(len(accuracy_list)), y=accuracy_list, label="Accuracy", color="darkorange")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle(
        f"Federated Learning Metrics\nModel Type: {model_type} | Algorithm: {training_algo} | Rounds (Server): {len(loss_list)}\n"
        f"Epochs (Client): {num_epochs} | Learning Rate = {learning_rate} | Optimizer = {optimizer}\n"
        f"Batch Size = {batch_size} | Client Frac = {client_fraction} | # Clients: {num_clients} | LR Decay: {lr_decay}",
        fontsize=16
    )
    plt.tight_layout()
    plt.savefig(metric_plots_dir / f"{model_type}_{training_algo}_{num_clients}_metrics.png", dpi=300)
    plt.close(fig)

    # Plot all model losses and accuracies
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for model_name_iter, _ in losses.items():
        try:
            loss_list = losses[model_name_iter]
            accuracy_list = accuracies[model_name_iter]

        except:
            continue

        sns.lineplot(ax=axes[0], x=range(len(loss_list)), y=loss_list, label=model_name_iter)
        sns.lineplot(ax=axes[1], x=range(len(accuracy_list)), y=accuracy_list, label=model_name_iter)

    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve for Different Models")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title("Accuracy Curve for Different Models")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle("Loss & Accuracy Curves for Different Models")
    plt.tight_layout()
    plt.savefig(metric_plots_dir / f"loss_curve_all_models.png", dpi=300)
    plt.close(fig)

    # Append data to output JSON
    result_entry = {
        "model_name": model_name,
        "model_type": model_type,
        "training_algo": training_algo,
        "num_clients": num_clients,
        "client_fraction": client_fraction,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "batch_size": batch_size,
        "num_rounds": len(losses[model_name]),
        "losses": losses[model_name],
        "accuracies": accuracies[model_name]
    }

    # Read existing data (if any) and append
    if os.path.exists(json_output_path):
        with open(json_output_path, "r") as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    existing_data.append(result_entry)

    with open(json_output_path, "w") as f:
        json.dump(existing_data, f, indent=4)   

def make_plots_fedmodcs(num_clients, model_name, client_fraction, json_output_path):    
    server_config_path = server_folder_abs_path / "fl_config_server.json"
    with open(server_config_path, "r") as f:
        fl_config = json.load(f)

    training_algo   = fl_config["training_algo"]
    num_epochs      = fl_config["num_epochs"]
    learning_rate   = fl_config["learning_rate"]
    optimizer       = fl_config["optimizer"]
    batch_size      = fl_config["batch_size"]
    model_type      = fl_config["model_type"]
    t_round         = fl_config["t_round"]
    t_final         = fl_config["t_final"]

    metric_plots_dir = server_folder_abs_path / "metric_plots"
    metric_plots_dir.mkdir(parents=True, exist_ok=True)

    timestamp_list = timestamps[model_name]
    accuracy_list = accuracies[model_name]
    loss_list = losses[model_name]
    fedmodcs_clients_selected_list = fedmodcs_clients_selected_each_round[model_name]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    sns.lineplot(ax=axes[0], x=timestamp_list, y=loss_list, label="Loss", color="steelblue")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    sns.lineplot(ax=axes[1], x=timestamp_list, y=accuracy_list, label="Accuracy", color="darkorange")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)

    sns.lineplot(ax=axes[2], x=timestamp_list, y=fedmodcs_clients_selected_list, label="FedModCS Clients Selected", color="green")
    axes[2].set_title("FedModCS Clients Selected")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Clients Selected")
    axes[2].legend()
    axes[2].grid(True)

    plt.suptitle(
        f"Federated Learning Metrics\nModel Type: {model_type} | Algorithm: {training_algo} | Rounds (Server): {len(loss_list)}\n"
        f"Epochs (Client): {num_epochs} | Learning Rate = {learning_rate} | Optimizer = {optimizer}\n"
        f"Batch Size = {batch_size} | Client Frac = {client_fraction} | # Clients: {num_clients}\n"
        f"t_round = {t_round} | t_final = {t_final}",
        fontsize=16
    )
    plt.tight_layout()
    plt.savefig(metric_plots_dir / f"{model_type}_{training_algo}_{t_round}_metrics.png", dpi=300)
    plt.close(fig)

    # Plot all model losses and accuracies
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    for model_name_iter, _ in losses.items():
        try:
            loss_list = losses[model_name_iter]
            accuracy_list = accuracies[model_name_iter]
            timestamp_list = timestamps[model_name_iter]
            fedmodcs_clients_selected_list = fedmodcs_clients_selected_each_round[model_name_iter]

        except:
            continue
        sns.lineplot(ax=axes[0], x=timestamp_list, y=loss_list, label=model_name_iter)
        sns.lineplot(ax=axes[1], x=timestamp_list, y=accuracy_list, label=model_name_iter)
        sns.lineplot(ax=axes[2], x=timestamp_list, y=fedmodcs_clients_selected_list, label=model_name_iter)

    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve for Different Models")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title("Accuracy Curve for Different Models")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].set_title("FedModCS Clients Selected")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Clients Selected")
    axes[2].legend()
    axes[2].grid(True)

    plt.suptitle("Loss & Accuracy Curves for Different Models")
    plt.tight_layout()
    plt.savefig(metric_plots_dir / f"loss_curve_all_models_fedmodcs.png", dpi=300)
    plt.close(fig)

     # Append data to output JSON
    result_entry = {
        "model_name": model_name,
        "model_type": model_type,
        "training_algo": training_algo,
        "num_clients": num_clients,
        "client_fraction": client_fraction,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "batch_size": batch_size,
        "num_rounds": len(losses[model_name]),
        "T_round": t_round,
        "T_final": t_final,
        "losses" : losses[model_name],
        "timestamps": timestamps[model_name],
        "accuracies": accuracies[model_name],
        "fedmodcs_clients_selected_each_round": fedmodcs_clients_selected_each_round[model_name]
    }

    # Read existing data (if any) and append
    if os.path.exists(json_output_path):
        with open(json_output_path, "r") as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    existing_data.append(result_entry)

    with open(json_output_path, "w") as f:
        json.dump(existing_data, f, indent=4)   


# ============================= MAIN =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--port", type=int, default=50051, help="Port Number for FL Server")
    parser.add_argument("--encrypt", type=int, default=0, help="Enable encryption (1) or not (0)")
    args = parser.parse_args()

    global encrypt
    encrypt = args.encrypt
    my_port = args.port
    my_ip = get_ip()

    server_logs_dir = server_folder_abs_path / "logs"
    server_logs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=server_logs_dir / "fl_server.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    global fl_server
    fl_server = FLServer(my_port, my_ip)

    menu()
