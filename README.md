# Projeto para Dissertação de Mestrado

#### Mitigating Non-IID Data impact in Federated Learning with Entropy

Projeto para Dissertação de Mestrado, usando a tecnologia SDK NVFlare da NVIDIA.
<br>Modelo de Federated Learning: FedAvg
<br>Framework: Pytorch

A medida de entropia pode ter uma relação direta com dados non-IID para otimizar o cálculo do SGD.

Os dados podem ter menor incerteza com menor entropia.

A entropia pode ser um critério de seleção de dados.

A entropia medida no nó pode ser um critério para definir novo conjunto de treinamento.
<br>Pois em um algoritmo de Rede Neural temos os dados de treinamento e dados de teste.


## MNIST FedAvg Entropy

Pasta com códigos fonte em Python para experimentos com dataset MNIST:

/mnist_fedavg_entropy

Arquivos de configuração da aplicação para executar com NVFlare:

- /mnist_fedavg_entropy/config/config_fed_client.json
- /mnist_fedavg_entropy/config/config_fed_server.json

Arquivo do modelo de Rede Neural CNN, classe ModerateCNN:

- /mnist_fedavg_entropy/custom/pt/networks/mnist_nets.py

Código fonte do Learner, onde é feita a inicialização das variáveis, normalização dos dados, treinamento e validação:

- /mnist_fedavg_entropy/custom/pt/learners/mnist_learner.py

Arquivos úteis para preparação dos dados Non-IID nos clientes, antes de executar o algoritmo de Federated Learning:

- /mnist_fedavg_entropy/custom/pt/utils/prepare_data.py
- /mnist_fedavg_entropy/custom/pt/utils/mnist_dataset.py


## CIFAR10 FedAvg Entropy

Pasta com códigos fonte em Python para experimentos com dataset CIFAR-10:

/cifar10_fedavg_entropy

Arquivos de configuração da aplicação para executar com NVFlare:

- /cifar10_fedavg_entropy/config/config_fed_client.json
- /cifar10_fedavg_entropy/config/config_fed_server.json

Arquivo do modelo de Rede Neural CNN, classe ModerateCNN:

- /cifar10_fedavg_entropy/custom/pt/networks/cifar10_nets.py

Código fonte do Learner, onde é feita a inicialização das variáveis, normalização dos dados, treinamento e validação:

- /cifar10_fedavg_entropy/custom/pt/learners/cifar10_learner.py

Arquivos úteis para preparação dos dados Non-IID nos clientes, antes de executar o algoritmo de Federated Learning:

- /cifar10_fedavg_entropy/custom/pt/utils/prepare_data.py
- /cifar10_fedavg_entropy/custom/pt/utils/mnist_dataset.py

## Instalação e Configuração

1. Comandos para instalar softwares em uma VM com Ubuntu:

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.8
sudo apt install python3-pip
sudo pip3 install virtualenv
```

2. Criar um VirtualEnv para SDK NVFlare:

```
virtualenv nvflare-env
source nvflare-env/bin/activate
```

3. Instalar softwares no nvflare-env:

```
python3 -m pip install -U pip
python3 -m pip install -U setuptools
python3 -m pip install nvflare==2.0.16
python3 -m pip install protobuf==3.20.0
python3 -m pip install torch torchvision tensorboard
python3 -m pip install tensorboard==2.10
pip3 install torchsummary
```

4. Exemplo de comandos para Preparar dados de um dataset, na VM do servidor:

MNIST:
```
python3 ./poc/admin/transfer/mnist_fedentropy/custom/pt/utils/prepare_data.py --data_dir="/home/ubuntu/dados/mnist_data" --num_sites="19" --alpha="0.1"
```

CIFAR-10:
```
python3 ./poc/admin/transfer/cifar10_fedentropy/custom/pt/utils/prepare_data.py --data_dir="/home/ubuntu/dados/cifar10_data" --num_sites="19" --alpha="0.1"
```

Após, preparar os dados, copiar os arquivos .npy gerados para cada cliente.

5. Inicializar servidor e clientes:

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

Demais VMs, substituindo a variável N pelo numero ID do cliente:

```
./poc/site-{N}/startup/start.sh
```

6. Comandos para executar as aplicações no NVflare:

Acessar ambiente de Administração, na VM server:

```
./poc/admin/startup/fl_admin.sh
```

No Admin, comandos para executar a aplicação nas VMs:

```
set_run_number 1
upload_app cifar10_fedentropy
deploy_app cifar10_fedentropy all
start_app all
```

No Admin, comando para verificar clientes:

```
check_status client
```

No Admin, comando para desligar clientes e servidor:

```
shutdown all
```

7. Comando para TensorBoard:

```
tensorboard --logdir=poc/server/run_1/tb_events --bind_all
```