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

</div>

---

## 👥 Team Members

- **Abhinav Raundhal** (2022101089)
- **Archisha Panda** (2022111019)
- **Vinit Mehta** (2022111001)

## 📋 Project Overview

This repository contains our implementation for the *Distributed Systems* course project (Spring 2025). We've developed a **Federated Learning Model** that allows distributed training across multiple client nodes while preserving data privacy.

## `data` Directory
```bash
python3 setup_data.py
```

This code converts the FashionMNIST and MNIST datasets into CSV files by flattening each 28×28 image into a 1D array of 784 pixels, adding labels, and combining train/test sets. It saves the processed data for easier use in non-PyTorch environments.

## `models` Directory

This directory contains data loaders, training and evaluation code for the 3 models implemented in this project namely DiabetesMLP, FashionMNISTCNN and MNISTMLP.

## `server_data` Directory

This directory contains data for testing the global model after each round of federated training, once the server has aggregated weights from all clients.

## `single_device_training` Directory
```bash
python3 train_all_models.py
```

Contains results of the models when trained with all the data on a single device. Also stores the trained models as pth files.

## Setup For Federated Learning
```bash
python3 setup.py --num_clients 3
```
This script sets up the directory structure and distributes datasets for a Federated Learning setup. It does the following:
- Creates Necessary Directories:
   - A clients folder (../clients/) with subfolders for each client (e.g., ../clients/1/, ../clients/2/, etc.).
   - A server data folder (../server_data/) for storing global test data.

- Distributes the Data:
   - Reads the three datasets and splits each dataset for training and testing in the ration (90:10).
   - The training data is then equally divided among n clients.

- Places client.py inside each client's directory.


## `src` Directory  

The `src` directory contains the base code for a **server-client file transfer system** with **dynamic server discovery** using `Consul`.  

### Current Implementation  
- Both the **client** and **server** are **menu-based**, requiring manual startup for each client.  
- Clients assume that each file sent has a **unique filename** (i.e., it does not already exist on the server).  
- To ensure uniqueness, clients append `_client_<client_id>` to the filename before sending.  
- Before starting federated learning, the server needs to send training code to the clients (DiabetesMLP.py/FashionMNISTCNN.py/MNISTMLP.py). This can be done using the Transfer File function from the menu of the server.
- After training is done by all clients, the final model is stroed by the server in the 'models' directory with the name `global_model_round_{last_round}.pth`.

### Required Enhancements  
- Implement an **automatic mode** for testing with multiple clients to streamline the process.  

### How to Run?  
1. Navigate into the `/src` folder.  
2. Run:  
   ```sh
   make
   ```  
   This compiles all the files and installs the required dependencies.  
3. Run:  
   ```sh
   make consul
   ```  
   This starts the **Consul server** for dynamic service discovery.  
4. Run:  
   ```sh
   make start_server
   ```  
   This starts the **federated learning server**.  
5. Run:  
   ```sh
   make start_clients
   ```  
   This is intended to start multiple clients automatically. **(Not yet implemented—use manual startup instead.)**  
6. For now, manually start a client using:  
   ```sh
   python3 ./client/client.py --port 50052 --id 1
   ```
7. First register all clients with the server by choosing option 1 from the menu.

8. Send the training code to all clients by choosing option 1 (Transfer File) and entering 'all' when asked for client ID.

9. Then select option 3 (Initialize Federated Learning) to initialise federated learning by the server. Enter the command line inputs as follows:
   - num_epochs : Number of local epochs after which client should send weights
   - learning_rate : LR for local training
   - optimizer : SGD/Adam to be used during training
   - batch_size : Batch size to be used for DataLoader
   - model_type : The model we want to train (MNISTMLP, FashionMNISTCNN, DiabetesMLP)
   - client_fraction : Fraction of clients to involve in training in every round

10. After initialisation is done, start federated training by choosing option 4 from the server menu. Enter the number of rounds of federated learning you want to perform.

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
