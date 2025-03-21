from enum import Enum
import os
import sys
import logging
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'generated'))
import file_transfer_pb2
import file_transfer_pb2_grpc as file_transfer_grpc

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def wait_for_enter():
    print()
    print("Press Enter to continue...", end="", flush=True)
    while True:
        char = sys.stdin.read(1)    # Reads one character at a time
        if char == "\n":            # Only proceed if Enter is pressed
            break

class FileTransferServicer(file_transfer_grpc.FileTransferServiceServicer):
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
                    success=False,
                    message="No data received from client"
                )
            
            received_dir = Path(__file__).parent / 'received_files'
            received_dir.mkdir(exist_ok=True)
            logging.info(f"Created/verified received_files directory at {received_dir}")
            
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
                success=True,
                message=f"File {filename} received successfully"
            )
                
        except Exception as e:
            logging.error(f"Unexpected error during file transfer: {str(e)}", exc_info=True)
            return file_transfer_pb2.FileResponse(
                success=False,
                message=f"Unexpected error: {str(e)}"
            )