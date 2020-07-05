# Federated Learning framework for Graph Neural Networks

## Introduction
This small framework can be used to scale up and train any Tensorflow or Pytrorch based graph neural network in federated manner on partitioned graphs or distributed graphs. Some graphs are too large and consumes huge amount of time if we train a neural network on it. In this case distributed learning is a potential approach but in some cases (Ex;- distributed graphs located on different datacenters) federated learning can be better becase it minimizes the communication overhead and speed up the training.

## Requirements

* Python 3.5 or higher (Conda or a Virtual environment is preferred)
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

## Start training

#### Starting server

python fl_server.py path_weights path_data graph_id partition_id num_clients num_rounds IP(optional - default localhost) PORT(optional - default 5000)

```
python fl_server.py ./weights/  ./data/ 4 0 2 3 localhost 5000
```


#### Starting clients

python fl_client.py path_weights path_embeddings path_data graph_id partition_id epochs(optional - default 10) IP(optional - default localhost) PORT(optional - default 5000) <br />

Client 1
```
python fl_client.py ./weights/ ./embeddings/ ./data/ 4 0 10 localhost 5000
```
Client 2
```
python fl_client.py ./weights/ ./embeddings/ ./data/ 4 1 10 localhost 5000
```


