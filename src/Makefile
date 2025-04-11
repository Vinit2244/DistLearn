PROTO_DIR=proto
OUT_DIR=generated

# Allow overriding with certain cmd line args
SERVER_PORT ?= 50051
IID ?= 3
NonIID ?= 3
x ?= 1

# Find all .proto files in the proto directory
PROTO_FILES=$(wildcard $(PROTO_DIR)/*.proto)
GENERATED_FILES=$(PROTO_FILES:$(PROTO_DIR)/%.proto=$(OUT_DIR)/%_pb2.py)
PID_FILE=client_pids.txt

all: compile

compile: $(GENERATED_FILES)

$(OUT_DIR)/%_pb2.py: $(PROTO_DIR)/%.proto
	python3 -m grpc_tools.protoc --proto_path=$(PROTO_DIR) --python_out=$(OUT_DIR) --grpc_python_out=$(OUT_DIR) $<

consul:
	consul agent -dev
	
start_server:
	python3 server/fl_server.py --port=$(SERVER_PORT)

setup:
	python3 ../setup.py --num_clients=$$(($(IID) + $(NonIID))) --IID=$(IID) --NonIID=$(NonIID) --x=$(x)

setup_and_visualise:
	python3 ../setup.py --num_clients=$$(($(IID) + $(NonIID))) --visualize_initial --visualize_distributed --IID=$(IID) --NonIID=$(NonIID) --x=$(x)

start_clients:
	@for i in $(shell seq 1 $$(($(IID) + $(NonIID)))); do \
		port=$$(($(SERVER_PORT) + $$i)); \
		openssl genrsa -out ../clients/$$i/client_$$i.key 2048; \
		openssl req -new -key ../clients/$$i/client_$$i.key -out ../clients/$$i/client_$$i.csr -config ../clients/$$i/client_$$i.cnf; \
		openssl x509 -req -in ../clients/$$i/client_$$i.csr -CA ../CA/ca.crt -CAkey ../CA/ca.key -CAcreateserial -out ../clients/$$i/client_$$i.crt -days 365 -sha256 -extfile ../clients/$$i/client_$$i.cnf -extensions req_ext; \
		rm -f ../clients/$$i/client_$$i.csr; \
		python3 ../clients/$$i/client.py --port $$port --id $$i --mode a & \
		echo $$! >> $(PID_FILE); \
	done

kill_clients:
	@if [ -f $(PID_FILE) ]; then \
		while read pid; do \
			kill $$pid 2>/dev/null || true; \
		done < $(PID_FILE); \
		rm -f $(PID_FILE); \
	else \
		echo "No PID file found."; \
	fi

clean:
	rm -rf $(OUT_DIR)/*_pb2.py*
	rm -rf $(OUT_DIR)/*_pb2_grpc.py*
	find . -name "__pycache__" -type d -exec rm -rf {} +
	rm -rf ./server/logs/*.log
	rm -rf ../clients
	rm -rf ./server/logs
	rm -rf ./server/models
	rm ./server/fl_config_client.json
	rm ./server/fl_config_server.json
	rm ./server/fl_server.json
