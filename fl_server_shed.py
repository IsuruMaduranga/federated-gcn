import socket
import pickle
import select
import time
import numpy as np
import pandas as pd
import sys
import logging
from timeit import default_timer as timer
import gc
import math

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s : [%(levelname)s]  %(message)s',
    handlers=[
        logging.FileHandler('server_shed.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class Server:

    def __init__(self, model_weights, ROUNDS , weights_path, graph_id, NUM_CLIENTS = 2, IP= socket.gethostname(), PORT = 5000, HEADER_LENGTH = 10 ):

        # Parameters
        self.HEADER_LENGTH =  HEADER_LENGTH
        self.IP = IP
        self.PORT = PORT
        self.NUM_CLIENTS = NUM_CLIENTS
        self.ROUNDS = ROUNDS

        self.weights_path = weights_path
        self.graph_id = graph_id

        # Global model
        self.GLOBAL_WEIGHTS = model_weights

        self.global_modlel_ready = False

        self.weights = []
        self.partition_sizes = []
        self.training_cycles = 0

        self.stop_flag = False
        self.finished_client_count = 0

        # List of sockets for select.select()
        self.sockets_list = []
        self.clients = {}
        self.client_ids = {}

        # Craete server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.IP, self.PORT))
        self.server_socket.listen(self.NUM_CLIENTS)

        self.sockets_list.append(self.server_socket)

    def update_model(self,new_weights,partition_ids):
        #self.partition_sizes.append(num_examples)
        #self.weights.append(num_examples * new_weights)
        self.weights.extend(new_weights)
        self.finished_client_count += 1

        if self.finished_client_count == self.NUM_CLIENTS:

            avg_weight = np.mean(self.weights, axis=0)
            #avg_weight = avg_weight / sum(self.partition_sizes)
            self.weights = []

            print(self.partition_sizes)
            self.partition_sizes = []
            self.finished_client_count = 0

            #self.GLOBAL_MODEL.set_weights(new_weights)
            self.GLOBAL_WEIGHTS = avg_weight

            self.training_cycles += 1

            # weights file name : global_weights_graphid.npy
            weights_path = self.weights_path + 'weights_' + 'graphID:' + self.graph_id + "_V" + str(self.training_cycles) + ".npy"
            np.save(weights_path,avg_weight)

            for soc in self.sockets_list[1:]:
                self.send_model(soc)
            
            logging.info("___________________________________________________ Training round %s done ______________________________________________________", self.training_cycles)
        

    def send_model(self, client_socket):

        if self.ROUNDS == self.training_cycles:
            self.stop_flag = True

        weights = np.array(self.GLOBAL_WEIGHTS)

        data = {"STOP_FLAG":self.stop_flag,"WEIGHTS":weights}

        data = pickle.dumps(data)
        data = bytes(f"{len(data):<{self.HEADER_LENGTH}}", 'utf-8') + data

        client_socket.sendall(data)

        logging.info('Sent global model to client-%s at %s:%s',self.client_ids[client_socket],*self.clients[client_socket])


    def receive(self, client_socket):
        try:
            
            message_header = client_socket.recv(self.HEADER_LENGTH)

            if not len(message_header):
                logging.error('Client-%s closed connection at %s:%s',self.client_ids[client_socket], *self.clients[client_socket])
                return False

            message_length = int(message_header.decode('utf-8').strip())

            #full_msg = client_socket.recv(message_length)

            full_msg = b''
            while True:
                msg = client_socket.recv(message_length)

                full_msg += msg

                if len(full_msg) == message_length:
                    break
            
            return pickle.loads(full_msg)

        except Exception as e:
            logging.error('Client-%s closed connection at %s:%s',self.client_ids[client_socket], *self.clients[client_socket])
            return False


    def run(self):

        while not self.stop_flag:

            read_sockets, write_sockets, exception_sockets = select.select(self.sockets_list, [], self.sockets_list)

            for notified_socket in read_sockets:

                if notified_socket == self.server_socket:

                    client_socket, client_address = self.server_socket.accept()
                    self.sockets_list.append(client_socket)
                    self.clients[client_socket] = client_address
                    self.client_ids[client_socket] = "new"

                    logging.info('Accepted new connection at %s:%s',*client_address)

                    self.send_model(client_socket)

                else:

                    message = self.receive(notified_socket)

                    if message is False:
                        self.sockets_list.remove(notified_socket)
                        del self.clients[notified_socket]
                        continue
                    else:
                        client_id = message['CLIENT_ID']
                        weights = message['WEIGHTS']
                        partitions = message["PARTITIONS"]
                        self.client_ids[notified_socket] = client_id
                    
                    logging.info('Recieved model from client-%s at %s:%s',client_id, *self.clients[notified_socket])
                    self.update_model(weights,partitions)

            for notified_socket in exception_sockets:
                self.sockets_list.remove(notified_socket)
                del self.clients[notified_socket]


if __name__ == "__main__":

    from models.supervised import Model

    arg_names = [
        'path_weights',
        'path_nodes',
        'path_edges',
        'graph_id',
        'partition_id',
        'num_clients',
        'num_rounds',
        'IP',
        'PORT'
        ]

    args = dict(zip(arg_names, sys.argv[1:]))

    logging.warning('####################################### New Training Session #######################################')
    logging.info('Server started , graph ID %s, number of clients %s, number of rounds %s',args['graph_id'],args['num_clients'],args['num_rounds'])

    if 'IP' not in args.keys()  or args['IP'] == 'localhost':
        args['IP'] = socket.gethostname()

    if 'PORT' not in args.keys():
        args['PORT'] = 5000

    path_nodes = args['path_nodes'] + args['graph_id'] + '_nodes_' + args['partition_id'] + ".csv"
    nodes = pd.read_csv(path_nodes,index_col=0)
    #nodes = nodes.astype("float32")

    path_edges = args['path_edges'] + args['graph_id'] + '_edges_' + args['partition_id'] + ".csv"
    edges = pd.read_csv(path_edges)
    #edges = edges.astype({"source":"uint32","target":"uint32"})
    
   
    model = Model(nodes,edges)
    model.initialize()

    logging.info('Model initialized')
    
    server = Server(model.get_weights(),ROUNDS=int(args['num_rounds']),weights_path=args['path_weights'],graph_id=args['graph_id'],NUM_CLIENTS=int(args['num_clients']),IP=args['IP'],PORT=int(args['PORT']))

    del nodes
    del edges
    del model
    gc.collect()
    
    logging.info('Federated training started!')

    start = timer()
    server.run()
    end = timer()

    elapsed_time = end -start
    logging.info('Federated training done!')
    logging.info('Training report : Elapsed time %s seconds, graph ID %s, number of clients %s, number of rounds %s',elapsed_time,args['graph_id'],args['num_clients'],args['num_rounds'])
    