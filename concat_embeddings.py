import sys
import pandas as pd

arg_names = [
        'path_embeddings',
        'graph_id',
        'num_partitions'
    ]

args = dict(zip(arg_names, sys.argv[1:]))

embeddings = []

for i in range(int(args['num_partitions'])):
    path = args['path_embeddings'] + "embeddings_nograd_" + args['graph_id'] + '_' + str(i) + ".csv"
    emb = pd.read_csv(path,index_col=0)
    emb = emb.astype("float32")
    embeddings.append(emb)


emb = pd.concat(embeddings)
emb = emb.loc[~emb.index.duplicated(keep='first')]


path = args['path_embeddings'] + "embeddings_nograd_" + args['graph_id'] + ".csv"

emb.index.name = None
emb.to_csv(path)

