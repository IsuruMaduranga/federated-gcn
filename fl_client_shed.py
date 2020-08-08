import sys
import logging

arg_names = [
    'client_id',
    'path_weights',
    'path_nodes',
    'path_edges',
    'graph_id',
    'partition_ids',
    'epochs',
    'IP',
    'PORT'
]

args = dict(zip(arg_names, sys.argv[1:]))

client_id = args['client_id']

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s : [%(levelname)s]  %(message)s',
    handlers=[
        logging.FileHandler(f'client_shed_{client_id}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

import socket
import pickle
import select
import numpy as np
import pandas as pd
from timeit import default_timer as timer
import time

from models.supervised import Model

class Client:

    def __init__(self, client_id, weights_path, graph_id, partition_ids, epochs = 10, IP = socket.gethostname(), PORT = 5000, HEADER_LENGTH = 10):

        self.HEADER_LENGTH =  HEADER_LENGTH
        self.IP = IP
        self.PORT = PORT
        self.client_id = client_id

        self.weights_path = weights_path
        self.graph_id = graph_id
        self.partition_ids = partition_ids
        self.epochs = epochs

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((IP, PORT))

        self.GLOBAL_MODEL = None
        self.MODEL = None

        self.LOCAL_MODELS = []
        self.partition_sizes = []

        self.STOP_FLAG = False
        self.rounds = 0


    def send_models(self):

        data = {"CLIENT_ID":self.client_id,"PARTITIONS":self.partition_ids,"PARTITION_SIEZES":self.partition_sizes,"WEIGHTS":self.LOCAL_MODELS}

        data = pickle.dumps(data)
        data = bytes(f"{len(data):<{self.HEADER_LENGTH}}", 'utf-8') + data
        self.client_socket.sendall(data)

        self.LOCAL_MODELS = []
        self.partition_sizes = []


    def fetch_model(self):
        
        message_header = self.client_socket.recv(self.HEADER_LENGTH)

        if not len(message_header):
            return False

        message_length = int(message_header.decode('utf-8').strip())

        full_msg = b''
        while True:
            msg = self.client_socket.recv(message_length)

            full_msg += msg

            if len(full_msg) == message_length:
                break
        
        data = pickle.loads(full_msg)

        self.STOP_FLAG = data["STOP_FLAG"]

        self.GLOBAL_MODEL = data["WEIGHTS"]

        return True


    def run(self):

        while not self.STOP_FLAG:

            read_sockets, _, exception_sockets = select.select([self.client_socket], [], [self.client_socket])

            success = False
            for soc in read_sockets:
                success = self.fetch_model()

                if success:
                    self.rounds += 1
                    logging.info('Global model v%s fetched',self.rounds - 1)
                else:
                    logging.error('Global model fetching failed')

            if not success:
                logging.error('Stop training')
                break

            if self.STOP_FLAG:
                self.MODEL.set_weights(self.GLOBAL_MODEL)
                
                logging.info('_____________________________________________________ Final model evalution ____________________________________________________________')

                for partition in self.partition_ids:
                    logging.info('********************************************************* Partition - %s ******************************************************',partition)
                    eval = self.MODEL.evaluate()

                    f1_train = (2 * eval[0][2] * eval[0][4]) / (eval[0][2] + eval[0][4])
                    f1_test = (2 * eval[1][2] * eval[1][4]) / (eval[1][2] + eval[1][4])
                    
                    logging.info('Final model (v%s) fetched',self.rounds)
                    logging.info('Training set : accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s',eval[0][1],eval[0][2],eval[0][3],f1_train,eval[0][4])
                    logging.info('Testing set : accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s',eval[1][1],eval[1][2],eval[1][3],f1_test,eval[1][4])
                
            else:
                
                logging.info('_____________________________________________________ Training Round %s ____________________________________________________________',self.rounds)

                for partition in self.partition_ids:

                    logging.info('********************************************************* Partition - %s ******************************************************',partition)

                    path_nodes = args['path_nodes'] + args['graph_id'] + '_nodes_' + partition + ".csv"
                    nodes = pd.read_csv(path_nodes,index_col=0)
                    nodes = nodes.astype("uint8")

                    path_edges = args['path_edges'] + args['graph_id'] + '_edges_' + partition + ".csv"
                    edges = pd.read_csv(path_edges)
                    edges = edges.astype({"source":"uint32","target":"uint32"})

                    logging.info('Model initialized')
                    self.MODEL = Model(nodes,edges)
                    num_train_ex,num_test_ex = self.MODEL.initialize()
                    self.partition_sizes.append(num_train_ex)

                    self.MODEL.set_weights(self.GLOBAL_MODEL)

                    logging.info('Number of training examples - %s, Number of testing examples %s',num_train_ex,num_test_ex)

                    eval = self.MODEL.evaluate()

                    f1_train = (2 * eval[0][2] * eval[0][4]) / (eval[0][2] + eval[0][4])
                    f1_test = (2 * eval[1][2] * eval[1][4]) / (eval[1][2] + eval[1][4])
                    logging.info('Global model v%s - Training set evaluation : accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s',self.rounds - 1,eval[0][1],eval[0][2],eval[0][3],f1_train,eval[0][4])
                    logging.info('Global model v%s - Testing set evaluation : accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s',self.rounds - 1,eval[1][1],eval[1][2],eval[1][3],f1_test,eval[1][4])

                
                    logging.info('Training started')
                    self.MODEL.fit(epochs = self.epochs)
                    self.LOCAL_MODELS.append(np.array(self.MODEL.get_weights()))
                    logging.info('Training done')

                    # eval = self.MODEL.evaluate()

                    # f1_train = (2 * eval[0][2] * eval[0][4]) / (eval[0][2] + eval[0][4])
                    # f1_test = (2 * eval[1][2] * eval[1][4]) / (eval[1][2] + eval[1][4])
                    # logging.info('After Round %s - Local model - Training set evaluation : accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s',self.rounds,eval[0][1],eval[0][2],eval[0][3],f1_train,eval[0][4])
                    # logging.info('After Round %s - Local model - Testing set evaluation : accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s',self.rounds,eval[1][1],eval[1][2],eval[1][3],f1_test,eval[1][4])

                logging.info('********************************************* All partitions trained **********************************************')

                logging.info('Sent local models to the aggregator')
                self.send_models()


if __name__ == "__main__":

    if 'IP' not in args.keys()  or args['IP'] == 'localhost':
        args['IP'] = socket.gethostname()

    if 'PORT' not in args.keys():
        args['PORT'] = 5000

    if 'epochs' not in args.keys():
        args['epoch'] = 10
    
    logging.warning('####################################### New Training Session #######################################')
    logging.info('Client started, graph ID %s, partition IDs %s , epochs %s',args['graph_id'],args['partition_ids'],args['epochs'])

   
    client = Client(args['client_id'],weights_path=args['path_weights'],graph_id=args['graph_id'],partition_ids=args['partition_ids'].split(","),epochs = int(args['epochs']) ,IP=args['IP'],PORT=int(args['PORT']))


    logging.info('Federated training started!')

    start = timer()
    client.run()
    end = timer()

    elapsed_time = end -start
    logging.info('Federated training done!')
    logging.info('Training report : Elapsed time %s seconds, graph ID %s, partition IDs %s, epochs %s',elapsed_time,args['graph_id'],args['partition_ids'],args['epochs'])

    
