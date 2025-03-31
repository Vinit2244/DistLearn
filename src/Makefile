PROTO_DIR=proto
OUT_DIR=generated

# Allow overriding with certain cmd line args
SERVER_PORT ?= 50051
N_CLIENTS ?= 1
CLIENT_ID ?= 1

# Find all .proto files in the proto directory
PROTO_FILES=$(wildcard $(PROTO_DIR)/*.proto)
GENERATED_FILES=$(PROTO_FILES:$(PROTO_DIR)/%.proto=$(OUT_DIR)/%_pb2.py)

all: compile

compile: $(GENERATED_FILES)

$(OUT_DIR)/%_pb2.py: $(PROTO_DIR)/%.proto
	python3 -m grpc_tools.protoc --proto_path=$(PROTO_DIR) --python_out=$(OUT_DIR) --grpc_python_out=$(OUT_DIR) $<

consul:
	consul agent -dev
	
start_server:
	python3 server/fl_server.py --port=$(SERVER_PORT)

setup:
	python3 ../setup.py --num_clients=3 --visualize_initial --visualize_distributed

start_clients:
	@for i in $(shell seq 1 $(N_CLIENTS)); do \
		port=$$(($(SERVER_PORT) + $$i)); \
		echo "Starting client $$i on port $$port"; \
		python3 client/client.py --port $$port --id $$i; \
	done

clean:
	rm -rf $(OUT_DIR)/*_pb2.py*
	rm -rf $(OUT_DIR)/*_pb2_grpc.py*
	find . -name "__pycache__" -type d -exec rm -rf {} +
	rm -rf ./server/logs/*.log 
	rm 