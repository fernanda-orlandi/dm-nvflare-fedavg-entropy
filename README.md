# Project for Master's Dissertation

#### Mitigating Non-IID Data impact in Federated Learning with Entropy

Project for Master's Dissertation, using SDK NVFlare technology from NVIDIA.
<br>Federated Learning model: FedAvg-BE
<br>Framework: PyTorch
<br>SDK: NVFlare 2.0

FedAvg-BE is a model that provides Federated Learning with the border entropy evaluation to select good input from a non-iid data environment.

The entropy measurement can be directly related to non-iid data to optimize the SGD calculation.
<br>Data can have lower uncertainty with lower entropy.
<br>The entropy measured in the node can be a criterion to define new training set.

The model is in the article of Orlandi *et al*, 2023:
<br>ORLANDI, F. C.; ANJOS, J. C. S.; LEITHARDT, V. R. Q.; SANTANA, J. F. de P.; GEYER, C. F. R. **Entropy to mitigate non-IID data problem on Federated Learning for the Edge Intelligence environment**. In IEEE Access, 2023, doi: 10.1109/ACCESS.2023.3298704.
<br><https://ieeexplore.ieee.org/document/10192897>


## MNIST FedAvg-BE

Folder with source code in Python for experiments with MNIST dataset:

/mnist_fedavg_entropy

Configuration files to run the application with NVFlare:

- /mnist_fedavg_entropy/config/config_fed_client.json
- /mnist_fedavg_entropy/config/config_fed_server.json

File with CNN Neural Network model, ModerateCNN class:

- /mnist_fedavg_entropy/custom/pt/networks/mnist_nets.py

Learner's source code, where variables are initialized, data are normalized and training and validation are performed:

- /mnist_fedavg_entropy/custom/pt/learners/mnist_learner.py

Useful files for preparing non-iid data on clients before running the Federated Learning application:

- /mnist_fedavg_entropy/custom/pt/utils/prepare_data.py
- /mnist_fedavg_entropy/custom/pt/utils/mnist_dataset.py


## CIFAR-10 FedAvg-BE

Folder with source code in Python for experiments with CIFAR-10 dataset

/cifar10_fedavg_entropy

Configuration files to run the application with NVFlare:

- /cifar10_fedavg_entropy/config/config_fed_client.json
- /cifar10_fedavg_entropy/config/config_fed_server.json

File with CNN Neural Network model, ModerateCNN class:

- /cifar10_fedavg_entropy/custom/pt/networks/cifar10_nets.py

Learner's source code, where variables are initialized, data are normalized and training and validation are performed:

- /cifar10_fedavg_entropy/custom/pt/learners/cifar10_learner.py

Useful files for preparing non-iid data on clients before running the Federated Learning application:

- /cifar10_fedavg_entropy/custom/pt/utils/prepare_data.py
- /cifar10_fedavg_entropy/custom/pt/utils/mnist_dataset.py

## Installation and Setup

#### 1. Commands to install software on a Virtual Machine (VM) with Ubuntu:

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.8
sudo apt install python3-pip
sudo pip3 install virtualenv
```

#### 2. Create a VirtualEnv for NVFlare SDK:

```
virtualenv nvflare-env
source nvflare-env/bin/activate
```

#### 3. Install softwares in "nvflare-env" virtual environment:

```
python3 -m pip install -U pip
python3 -m pip install -U setuptools
python3 -m pip install nvflare==2.0.16
python3 -m pip install protobuf==3.20.0
python3 -m pip install torch torchvision tensorboard
python3 -m pip install tensorboard==2.10
pip3 install torchsummary
```

#### 4. Example of commands to prepare data from a dataset, in the server VM:

MNIST:
```
python3 ./poc/admin/transfer/mnist_fedentropy/custom/pt/utils/prepare_data.py --data_dir="/home/ubuntu/dados/mnist_data" --num_sites="19" --alpha="0.1"
```

CIFAR-10:
```
python3 ./poc/admin/transfer/cifar10_fedentropy/custom/pt/utils/prepare_data.py --data_dir="/home/ubuntu/dados/cifar10_data" --num_sites="19" --alpha="0.1"
```

After preparing the data, copy the generated .npy files to each client VM.

#### 5. Initialize server and clients:

VM Server:

```
./poc/server/startup/start.sh
```

VM Site-1:

```
./poc/site-1/startup/start.sh
```

VM Site-2:

```
./poc/site-2/startup/start.sh
```

Other VMs, replacing the variable N with the client's ID number:

```
./poc/site-{N}/startup/start.sh
```

#### 6. Commands to run applications in NVflare:

Access the Administration environment, on the VM server:

```
./poc/admin/startup/fl_admin.sh
```

In Admin mode, commands to run the application on VMs:

```
set_run_number 1
upload_app cifar10_fedentropy
deploy_app cifar10_fedentropy all
start_app all
```

In Admin mode, command to check clients:

```
check_status client
```

In Admin mode, command to shut down clients and server:

```
shutdown all
```

#### 7. Command for TensorBoard:

```
tensorboard --logdir=poc/server/run_1/tb_events --bind_all
```