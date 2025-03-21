# ================================ IMPORTS ================================
import grpc
from concurrent import futures
import sys
import os
import argparse
import logging
from pathlib import Path

# Add the generated directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'generated'))
import file_transfer_pb2
import file_transfer_pb2_grpc

sys.path.append("utils")
from utils import FileTransferServicer


# ================================ GLOBALS ================================
CHUNK_SIZE = 1024  # 1KB chunks

logging.basicConfig(
    filename="./server/logs/server.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ================================ MAIN ==================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='File Transfer Server')
    parser.add_argument('--port', type=int, default=50051, help='Port to listen on')
    args = parser.parse_args()
    
    logging.info("Initializing server")
    logging.info(f"Starting server on port {args.port}")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    file_transfer_pb2_grpc.add_FileTransferServiceServicer_to_server(
        FileTransferServicer(), server
    )
    server.add_insecure_port(f'localhost:{args.port}')
    server.start()
    logging.info(f"Server started successfully on port {args.port}")

    server.wait_for_termination()