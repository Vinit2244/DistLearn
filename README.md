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


Here’s a refined version for your `.md` file:  

---  

## `src` Directory  

The `src` directory contains the base code for a **server-client file transfer system** with **dynamic server discovery** using `Consul`.  

### Current Implementation  
- Both the **client** and **server** are **menu-based**, requiring manual startup for each client.  
- Clients assume that each file sent has a **unique filename** (i.e., it does not already exist on the server).  
- To ensure uniqueness, clients append `_client_<client_id>` to the filename before sending.  

### Required Enhancements  
- Implement an **automatic mode** for testing with multiple clients to streamline the process.  