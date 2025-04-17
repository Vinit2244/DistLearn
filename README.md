<p id="readme-top"></p>
<br />

<p align="center">
  <h1 align="center">⚡ Distributed Federated Learning ⚡</h1>
  <p align="center">
    A robust implementation of a Federated Learning Model for distributed environments
    <br />
    <strong>Team 10 · Distributed Systems · Spring 2025</strong>
  </p>
</p>

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![gRPC](https://img.shields.io/badge/gRPC-244c5a?style=for-the-badge&logo=google&logoColor=white)
![Consul](https://img.shields.io/badge/Consul-F24C53?style=for-the-badge&logo=consul&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

</div>

---

## 👥 Team Members

- **Abhinav Raundhal** (2022101089)
- **Archisha Panda** (2022111019)
- **Vinit Mehta** (2022111001)

## 📋 Project Overview

This repository contains our implementation for the *Distributed Systems* course project (Spring 2025). We've developed a **Federated Learning Model** that allows distributed training across multiple client nodes while preserving data privacy.

## 📋 What is Federated Learning?

Federated Learning is a distributed machine learning approach where models are trained locally on devices, preserving data privacy.

---

## 🗂️ Directory Structure

```plaintext
.
├── ablation/                  # Contains experimental results for various configurations
├── data/                      # Datasets and preprocessing scripts
│   ├── diabetes_dataset.csv
│   ├── fashion_mnist_dataset.csv
│   ├── mnist_dataset.csv
│   └── setup_data.py
├── docs/                      # Documentation and reference papers
├── src/                       # Source code for the project
│   ├── client/                # Client-side implementation
│   ├── server/                # Server-side implementation
│   │   ├── fl_server.py       # Main Federated Learning server logic
│   │   └── ...
│   ├── generated/             # Auto-generated gRPC files
│   ├── models/                # Model definitions and training scripts
│   ├── proto/                 # Protocol buffer definitions
│   └── Makefile               # Build and execution commands
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
└── ...
```

---

## `models` Directory

This directory contains data loaders, training and evaluation code for the 3 models implemented in this project namely DiabetesMLP, FashionMNISTCNN and MNISTMLP.

## `single_device_training` Directory
```bash
python3 train_all_models.py
```

Contains results of the models when trained with all the data on a single device. Also stores the trained models as pth files.

## `src` Directory  

The `src` directory contains the base code for a **server-client file transfer system** with **dynamic server discovery** using `Consul`.  

### Current Implementation  
- Both the **client** and **server** are **menu-based**, requiring manual startup for each client.  
- Clients assume that each file sent has a **unique filename** (i.e., it does not already exist on the server).  
- Before starting federated learning, the server needs to send training code to the clients (DiabetesMLP.py/FashionMNISTCNN.py/MNISTMLP.py). This can be done using the Transfer File function from the menu of the server.
- After training is done by all clients, the final model is stroed by the server in the 'models' directory with the name `global_model_round_{last_round}.pth`.

## 🔐 Security Features

The system includes SSL/TLS support with custom certificates. The repository includes a Certificate Authority (CA) setup for generating and signing certificates.

### CA Certificate Generation

```bash
# Inside CA folder
openssl genrsa -out ca.key 2048
openssl req -x509 -new -nodes -key ca.key -sha256 -days 365 -out ca.crt
```

### Server Certificate Generation

```bash
# Create server key and CSR
openssl genrsa -out server.key 2048
openssl req -new -key server.key -out server.csr -config server.cnf

# Get CSR signed by CA
openssl x509 -req -in server.csr -CA ../CA/ca.crt -CAkey ../CA/ca.key \
  -CAcreateserial -out server.crt -days 365 -sha256 -extfile server.cnf -extensions req_ext
```

Source: https://resonant-cement-f3c.notion.site/Self-Signed-Certificates-Create-your-own-Certificate-Authority-CA-for-local-HTTPS-sites-536636144b124904a52e4ac68973bb2c

---

## 🚀 How to Run the Code

Follow these steps to set up and run the Federated Learning system:

1. **Install Dependencies**  
   Ensure you have Python 3.8+ installed. Install the required dependencies using:  
   ```bash
   pip install -r requirements.txt
   ```

2. **Compile Protocol Buffers**  
   Generate gRPC files from `.proto` definitions:  
   ```bash
   make compile
   ```
   - This command uses the `protoc` compiler to generate Python code for gRPC communication based on the `.proto` files in the `proto/` directory.

3. **Set Up the Environment**  
   Prepare the directory structure and distribute datasets:  
   ```bash
   make do_setup_capabilities
   ```
   - This command ensures that all necessary directories are created and datasets are distributed to the appropriate locations for training.

4. **Start the Consul Server**  
   Start the Consul agent for dynamic service discovery:  
   ```bash
   make consul
   ```
   - This command launches the Consul server, which is used for service discovery, enabling clients to dynamically locate the server.

5. **Start the Federated Learning Server**  
   Launch the server with optional encryption:  
   ```bash
   make start_server
   ```
   - This command starts the Federated Learning server. If encryption is enabled (`ENCRYPT=1`), the server will use SSL/TLS for secure communication.

6. **Start the Clients**  
   Start multiple clients to connect to the server:  
   ```bash
   make start_clients
   ```
   - This command launches the client processes, which will connect to the server, receive training tasks, and send back model updates.

7. **Kill All Clients**  
   Stop all running clients:  
   ```bash
   make kill_clients
   ```
   - This command terminates all active client processes.

8. **Clean Up**  
   Remove generated files and logs:  
   ```bash
   make clean
   ```
   - This command deletes temporary files, logs, and other artifacts generated during the execution of the system.

---

## 🔄 Code Flow

Here’s how the system works:

1. **Client Registration**  
   - Clients connect to the server and register themselves.  
   - The server waits until all clients are registered before proceeding.

2. **Encryption**  
   - If encryption is enabled (`ENCRYPT=1`), RSA certificates are generated for secure communication.  
   - Each client has its own private key and certificate.

3. **Federated Learning Initialization**  
   - The server initializes the Federated Learning process by selecting a training algorithm (e.g., `FedSGD`, `FedAvg`, `FedAdp`, `FedModCS`).  
   - It distributes the training configuration (e.g., model type, optimizer, learning rate) to the clients.

4. **Local Training**  
   - Clients train the model locally on their datasets for a specified number of epochs.  
   - After training, clients send their weight updates to the server.

5. **Aggregation**  
   - The server aggregates the weight updates using algorithms like FedAvg or FedAdp.  
   - The global model is updated and saved after each round.

6. **Evaluation**  
   - The server evaluates the global model on a test dataset.  
   - Metrics like accuracy and loss are logged and visualized using Matplotlib.

7. **Repeat**  
   - Steps 4–6 are repeated for the specified number of rounds.

---

## 🛠️ Frameworks and Tools Used

- **Python**: Core programming language for the project.  
- **gRPC**: For communication between the server and clients.  
- **Consul**: For dynamic service discovery.  
- **PyTorch**: For model training and evaluation.
- **OpenSSL**: For generating RSA certificates for encryption.

---

## 📌 Assumptions

- Each client has access to its own local dataset.  
- The server and clients are running on the same device.  
- Encryption is optional and can be enabled using the `ENCRYPT` flag.  
- The server assumes that all clients will complete their training and send updates within the expected time.
- It is assumed that no client failures occur during the training process.

---

## 🎨 Visualizations

The system generates plots for metrics like loss and accuracy after each round of training. These plots are saved in the `server/metric_plots` directory.

---

## 🌐 Website

To run:
```bash
make website
```

---
