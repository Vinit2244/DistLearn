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

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'generated'))
import file_transfer_pb2
import file_transfer_pb2_grpc as file_transfer_grpc

sys.path.append("utils")
from utils import clear_screen, wait_for_enter, get_ip, STYLES, CHUNK_SIZE


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


class FLServerServicer(file_transfer_grpc.FLServerServicer):
    def __init__(self):
        self.clients = {}

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
        print()
        choice = input(f"{STYLES.FG_YELLOW}Enter your choice: {STYLES.RESET}")
        if choice == "1":
            file_path = input(f"{STYLES.FG_YELLOW}Enter the file path: {STYLES.RESET}")
            if not Path(file_path).is_file():
                print(f"{STYLES.BG_RED}File not found. Please try again.{STYLES.RESET}")
            client_id = input(f"{STYLES.FG_YELLOW}Enter the client ID: {STYLES.RESET}")
            try:
                client_id = int(client_id)
            except ValueError:
                print(f"{STYLES.BG_RED}Invalid client ID. Please try again.{STYLES.RESET}")
                wait_for_enter()
                continue
            fl_server.send_file_to_client(file_path, client_id)
            wait_for_enter()
        elif choice == "2":
            break
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
