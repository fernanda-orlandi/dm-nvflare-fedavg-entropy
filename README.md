# Projeto para Dissertação de Mestrado

#### Mitigating Non-IID Data impact in Federated Learning with Entropy

Projeto para Dissertação de Mestrado, usando a tecnologia SDK NVFlare da NVIDIA.
Modelo de Federated Learning: FedAvg
Framework: Pytorch

A medida de entropia pode ter uma relação direta com dados non-IID para otimizar o cálculo do SGD.

Os dados podem ter menor incerteza com menor entropia.

A entropia pode ser um critério de seleção de dados.

A entropia medida no nó pode ser um critério para definir novo conjunto de treinamento.
Pois em um algoritmo de Rede Neural temos os dados de treinamento e dados de teste.


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
