import pandas as pd
import sys
from models.unsupervised import Model

if __name__ == "__main__":

    from models.unsupervised import Model

    arg_names = [
        'path_embeddings',
        'path_data',
        'graph_id',
        'partition_id',
        'epochs'
        ]

    args = dict(zip(arg_names, sys.argv[1:]))

    path_nodes = args['path_data'] + args['graph_id'] + '_nodes_' + args['partition_id'] + ".csv"
    nodes = pd.read_csv(path_nodes,index_col=0)

    path_edges = args['path_data'] + args['graph_id'] + '_edges_' + args['partition_id'] + ".csv"
    edges = pd.read_csv(path_edges) 

    model = Model(nodes,edges)
    model.initialize()

    embeddings = model.gen_embeddings()
                
    # embeddings file name : embeddings_nograd_graphid_workerid.npy
    embeddings_path = args['embeddings_path'] + 'embeddings_nograd_' + args['graph_id'] + '_' + args['partition_id'] + ".csv"
    embeddings.to_csv(embeddings_path)

    model.fit(int(args['epochs']))



