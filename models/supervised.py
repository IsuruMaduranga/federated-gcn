# imports
import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, HinSAGE, link_classification
from stellargraph import globalvar
from stellargraph import datasets

from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection
import os
import sys
import numpy as np
import pandas as pd

class Model:

    def __init__(self,nodes,edges):
        self.model = None

        self.nodes =  nodes
        self.edges = edges
        self.graph = None

        self.train_flow = None

    def initialize(self,**hyper_params):

        if(not "batch_size" in hyper_params.keys()):
            batch_size = 20
        if(not "layer_sizes" in hyper_params.keys()):
            num_samples = [20, 10]
        if(not "num_samples" in hyper_params.keys()):
            layer_sizes = [20, 20]
        if(not "bias" in hyper_params.keys()):
            bias = True
        if(not "dropout" in hyper_params.keys()):
            dropout = 0.3
        if(not "lr" in hyper_params.keys()):
            lr = 1e-3
        if(not "train_split" in hyper_params.keys()):
            train_split = 0.2

        self.graph = sg.StellarGraph(nodes=self.nodes,edges=self.edges)

        # Train split
        edge_splitter_train = EdgeSplitter(self.graph)
        graph_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
            p=train_split, method="global", keep_connected=True
        )

        # Train iterators
        train_gen = GraphSAGELinkGenerator(graph_train, batch_size, num_samples)
        self.train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)

        # Model defining - Keras functional API + Stellargraph layers
        graphsage = GraphSAGE(
            layer_sizes=layer_sizes, generator=train_gen, bias=bias, dropout=dropout
        )

        x_inp, x_out = graphsage.in_out_tensors()

        prediction = link_classification(
            output_dim=1, output_act="relu", edge_embedding_method="ip"
        )(x_out)

        self.model = keras.Model(inputs=x_inp, outputs=prediction)

        self.model.compile(
            optimizer=keras.optimizers.Adam(lr=lr),
            loss=keras.losses.binary_crossentropy,
            metrics=["acc"],
        )

        return self.model.get_weights()

    def set_weights(self,weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def fit(self,epochs = 10):
        history = self.model.fit(self.train_flow, epochs=epochs, verbose=0)
        return self.model.get_weights(),history



if __name__ == "__main__":

    path_weights = "./weights/weights.npy"
    path_node_partition = "./data/4_attributes_0"
    path_edge_partition = "./data/4_0"

    nodes = pd.read_csv(path_node_partition , sep='\t', lineterminator='\n',header=None).loc[:,0:1433]
    nodes.set_index(0,inplace=True)

    edges = pd.read_csv(path_edge_partition , sep='\s+', lineterminator='\n', header=None)
    edges.columns = ["source","target"] 

    model = Model(nodes,edges)
    model.initialize()

    print("Training started")
    new_weights,history = model.fit()
    print("Training done")

    # Save weights
    np.save(path_weights,new_weights)
