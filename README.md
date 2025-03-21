<p id="readme-top"></p>
<br />

# <center><span style="color:cyan">Documentation</span></center>
## <center><span style="color:yellow">Federated Learning Model</span></center>
<center>
    <i>
        <b>
            Team Number: 10
        </b>
    </i>
    <br><br>
    <i>
        <b>
            Team Members:
        </b>
        <br> Abhinav Raundhal (2022101089)
        <br> Archisha Panda (2022111019)
        <br> Vinit Mehta (2022111001)
    </i>
    <br>
</center>
<br>

This repository contains the course project for the *Distributed Systems* course in the *Spring 2025 semester*. The project focuses on implementing a **Federated Learning Model**.


## For message passing directory
Run the code from message passing code (when code pasted to some other file run from the similar path or else change the paths in the files accordingly.) Please note that abhi maine sirf files send krne ka code likha hai, client can send files to server. To make it server sending files to all the clients we will have to implement the logic of clients registering themselves with the server at start and then the server can send the files to all the registered clients. Vo abhi nahi kiya hai that will be done in Phase 2 when we start implementing FedSGD.

---

To run my code:
Navigate to `file transfer code` repository

Run
```Bash
make compile
```
to compile

---

Run
```Bash
make start_server
```
in one of the terminals

---

Run
```Bash
make start_clients
```
in another terminal, there you can speicfy the number of clients by passing the argument `N_CLIENTS=3` for starting 3 clients