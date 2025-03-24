# ============================= IMPORTS ==============================
import argparse
import logging
from concurrent import futures
import grpc
import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'generated'))
import file_transfer_pb2
import file_transfer_pb2_grpc as file_transfer_grpc

sys.path.append("utils")
from utils import clear_screen, wait_for_enter, get_server_address, get_ip, STYLES, CHUNK_SIZE

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
            print(f"{STYLES.FG_GREEN}Client {self.id} successfully registered with FL server{STYLES.RESET}")
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
            print(f"{STYLES.FG_GREEN}Client {self.id} successfully deregistered from FL Server{STYLES.RESET}")
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
            print(f"{STYLES.FG_GREEN}Success: {response.msg}{STYLES.RESET}")
        else:
            logging.error(f"File transfer failed: {response.msg}")
            print(f"{STYLES.BG_RED + STYLES.FG_WHITE}Error: {response.msg}{STYLES.RESET}")


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
            
            received_dir = Path(__file__).parent / 'received_files'
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
        if choice == "1":
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

    global my_port, my_id, my_ip, client
    my_port = args.port
    my_id = args.id
    my_ip = get_ip()

    # Create a client object
    client = Client(my_id, my_ip, my_port)

    # Logging configuration
    logging.basicConfig(
        filename=f"./client/logs/client_{my_id}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # 'a' mode for testing using multiple clients
    if mode.lower() == 'i':     # Interactive mode
        menu()
    elif mode.lower() == 'a':   # Automatic mode
        ...
