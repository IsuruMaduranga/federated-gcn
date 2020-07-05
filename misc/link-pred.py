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



# Inputes
# path_weights = sys.argv[1]
# path_node_partition = sys.argv[2]
# path_edge_partition = sys.argv[3]

path_weights = "./weights/weights.npy"
path_node_partition = "./data/4_attributes_0"
path_edge_partition = "./data/4_0"

# Constructing the graph
nodes = pd.read_csv(path_node_partition , sep='\t', lineterminator='\n',header=None).loc[:,0:1433]
nodes.set_index(0,inplace=True)

edges = pd.read_csv(path_edge_partition , sep='\s+', lineterminator='\n', header=None)
edges.columns = ["source","target"]

G = sg.StellarGraph(nodes=nodes,edges=edges)

# Train split
edge_splitter_train = EdgeSplitter(G)
G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
    p=0.2, method="global", keep_connected=True
)

# Hyperparams
batch_size = 20
epochs = 20
num_samples = [20, 10]
layer_sizes = [20, 20]

# Train iterators
train_gen = GraphSAGELinkGenerator(G_train, batch_size, num_samples)
train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)



# Model defining - Keras functional API + Stellargraph layers
graphsage = GraphSAGE(
    layer_sizes=layer_sizes, generator=train_gen, bias=True, dropout=0.3
)

x_inp, x_out = graphsage.in_out_tensors()

prediction = link_classification(
    output_dim=1, output_act="relu", edge_embedding_method="ip"
)(x_out)

model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3),
    loss=keras.losses.binary_crossentropy,
    metrics=["acc"],
)

# Set weights
weights = np.load(path_weights,allow_pickle=True)
model.set_weights(weights)

print("Training started")
history = model.fit(train_flow, epochs=epochs, verbose=0)
print("Training done")

# Save weights
weights = model.get_weights()
np.save(path_weights,weights)
