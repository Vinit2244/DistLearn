# ================================ IMPORTS ================================
import grpc
from concurrent import futures
import sys
import os
import argparse
import logging
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'generated'))
import file_transfer_pb2
import file_transfer_pb2_grpc as file_transfer_grpc

sys.path.append("utils")
from utils import clear_screen, wait_for_enter, FileTransferServicer


# ================================ GLOBALS ================================
CHUNK_SIZE = 1024  # 1KB chunks
server_ip = None
server_port = None
my_port = None
my_id = None


# ================================ FUNCTIONS ===============================
def send_file(stub, file_path):
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        print(f"Error: File not found: {file_path}")
        return

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
                    chunk=chunk,
                    is_last_chunk=len(chunk) < CHUNK_SIZE
                )
    
    response = stub.TransferFile(request_iterator())
    if response.success:
        logging.info(f"File transfer successful: {response.message}")
        print(f"Success: {response.message}")
    else:
        logging.error(f"File transfer failed: {response.message}")
        print(f"Error: {response.message}")


# ================================== MAIN ==================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_ip', type=str, default='localhost', help='Server IP address')
    parser.add_argument('--server_port', type=int, default=50051, help='Server port')
    parser.add_argument('--client_port', type=int, required=True, help='Client port')
    parser.add_argument('--client_id', type=int, required=True, help='Client ID')
    args = parser.parse_args()

    server_ip = args.server_ip
    server_port = args.server_port
    my_port = args.client_port
    my_id = args.client_id
    
    # Configure logging with client ID
    logging.basicConfig(
        filename=f"./client/logs/client_{args.client_id}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    file_transfer_grpc.add_FileTransferServiceServicer_to_server(
        FileTransferServicer(), server
    )
    server.add_insecure_port(f'localhost:{my_port}')
    server.start()
    logging.info(f"Client server started successfully on port {my_port}")
    
    while True:
        clear_screen()
        print("Options:")
        print("1. Send a file")
        print("2. Exit")
        choice = input("Enter your choice (1-2): ")
        
        if choice == "1":
            channel = grpc.insecure_channel(f'{server_ip}:{server_port}')
            stub = file_transfer_grpc.FileTransferServiceStub(channel)
            logging.info(f"Connected to server at {server_ip}:{server_port}")
            
            file_path = input("Enter the path to the file you want to send: ")
            logging.info(f"User selected file: {file_path}")
            send_file(stub, file_path)
            
            channel.close()
            wait_for_enter()
        elif choice == "2":
            logging.info("Exiting")
            sys.exit(0)
        else:
            logging.warning(f"Invalid choice entered: {choice}")
            print("Invalid choice. Please try again.")
            wait_for_enter()
    
    server.wait_for_termination()