# Federated Learning framework for Graph Neural Networks

## Introduction
This small framework can be used to scale up and train any Tensorflow or Pytrorch based graph neural network in federated manner on partitioned graphs or distributed graphs. Some graphs are too large and consumes huge amount of time if we train a neural network on it. In this case distributed learning is a potential approach but in some cases (Ex;- distributed graphs located on different datacenters) federated learning can be better becase it minimizes the communication overhead and speed up the training.

## Requirements

* Python 3.5 or higher (A conda or a Virtual environment is preferred)
* Tensorflow 2
* Stellargraph
* Scikit-learn
* Pandas
* Numpy

## Installation

1. Clone the repository
2. Install dependencies as follows

``` pip3 install -r requirements.txt ```

## Model structure
To be trained using this framework a neural network should be wrapped with a python class that provides following methods. Few example models are in the models folder in the repo.

```
class Model:

    def __init__(self,nodes,edges):
        # define class variables here

    def initialize(self,**hyper_params):
        # define model initialization logic here
        # **hyper_params dictionary can be used to pass any variable

        return initial model weights

    def set_weights(self,weights):
        # set passed weights to your model

    def get_weights(self):
        # extract and return model weights
        return model weights

    def fit(self,epochs = 4):
        # define training logic here
        return model weights, training history

    def gen_embeddings(self):
        # this method is optional
        return embeddings as a pandas dataframe

```
## Repository structure

Repository <br />
├── data: partitioned CORA dataset for testing <br />
├── misc: miscellaneous <br />
├── models: where models are stored <br />
* upervised.py: supervised implementation of GRAPHSAGE
* supervised.py: unsupervised implementation of GRAPHSAGE

## Start training

#### Starting fl_server

Following arguments must be passed in following order

* path_weights - A location to extract and store model weights
* path_nodes - Where your graph nodes are stored
* path_edges - Where your graph edges are stored
* graph_id - ID for identify graphs
* partition_id - ID of the partition located in server that is used to initialize the weights
* num_clients - Number of clients that will be join for the federated training
* num_rounds - Number of federated rounds to be trained
* IP(optional - default localhost) - IP of the VM that fl_server is in
* PORT(optional - default 5000) - PORT that shuould be used to communicate with clients

```
python fl_server.py ./weights/ ./data/ ./data/ 4 0 2 3 localhost 5000
```


#### Starting fl_client s

* path_weights - A location to extract and store model weights
* path_embeddings - A location to store node embeddings if you want to generate them
* path_nodes - Where your graph nodes are stored
* path_edges - Where your graph edges are stored
* graph_id - ID for identify graphs
* partition_id - ID of the partition located in server that is used to initialize the weights
* epochs - number of epochs to train
* num_rounds - Number of federated rounds to be trained
* IP(optional - default localhost) - IP of the VM that fl_server is in
* PORT(optional - default 5000) - PORT that fl_server is listening to

Any number of clients can be started but number of clients should be passed in to fl_server when it is started as explained above. <br />

client 1
```
python fl_client.py ./weights/ ./embeddings/ ./data/ ./data/ 4 0 4 localhost 5000
```
client 2
```
python fl_client.py ./weights/ ./embeddings/ ./data/ ./data/ 4 1 4 localhost 5000
```


