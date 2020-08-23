def mem(num_of_nodes,num_of_edges,num_of_features,feature_data_type,edge_data_type):
    
    edge_mem = 2 * num_of_edges * (edge_data_type/8)
    node_mem = num_of_nodes * num_of_features * (feature_data_type/8)

    graph_size = (edge_mem + node_mem) / (1024*1024*1024)

    return 3.6 * graph_size + 2


def mem_est(partition_data,num_of_features,feature_data_type,edge_data_type):

    mems = []

    for data in partition_data:
        mems.append(mem(data[0],data[1],num_of_features,feature_data_type,edge_data_type))

    return mems

if __name__ == "__main__":
    
    # num_of_features
    num_of_features = 1433

    # feature_data_type = int8
    feature_data_type = 64
    
    # edge_data_type = int64
    edge_data_type = 64

    # partiton_data = list of tuples (num_of_nodes,num_of_edges)
    partition_data = [(1452,2383),(1432,2593)]

    mems = mem_est(partition_data,num_of_features,feature_data_type,edge_data_type)

    print(mems)
