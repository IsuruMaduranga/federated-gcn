import socket
import pickle
import select
import sys
import numpy as np
import pandas as pd
import logging
from timeit import default_timer as timer
import time

arg_names = [
    'path_weights',
    'path_nodes',
    'path_edges',
    'graph_id',
    'partition_id',
    'epochs',
    'IP',
    'PORT'
]

args = dict(zip(arg_names, sys.argv[1:]))

partition_id = args['partition_id']

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s : [%(levelname)s]  %(message)s',
    handlers=[
        logging.FileHandler(f'client_{partition_id}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
class Client:

    def __init__(self, MODEL, weights_path, graph_id, partition_id, epochs = 10, IP = socket.gethostname(), PORT = 5000, HEADER_LENGTH = 10):

        self.HEADER_LENGTH =  HEADER_LENGTH
        self.IP = IP
        self.PORT = PORT

        self.weights_path = weights_path
        self.graph_id = graph_id
        self.partition_id = partition_id
        self.epochs = epochs

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((IP, PORT))

        self.MODEL = MODEL
        self.STOP_FLAG = False
        self.rounds = 0


    def send_model(self):

        # svae model weights
        # weights file name : weights_graphid_workerid.npy
        weights_path = self.weights_path + 'weights_' + self.graph_id + '_' + self.partition_id + ".npy"
        np.save(weights_path,self.MODEL.get_weights())

        weights = np.array(self.MODEL.get_weights())

        data = {"CLIENT_ID":self.partition_id,"WEIGHTS":weights}

        data = pickle.dumps(data)
        data = bytes(f"{len(data):<{self.HEADER_LENGTH}}", 'utf-8') + data
        self.client_socket.sendall(data)


    def receive(self):
        try:

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

            return data["WEIGHTS"]

        except Exception as e:
            print(e)


    def fetch_model(self):
        data = self.receive()
        self.MODEL.set_weights(data)

    def train(self):
        self.MODEL.fit(epochs = self.epochs)

    def run(self):

        while not self.STOP_FLAG:

            read_sockets, _, exception_sockets = select.select([self.client_socket], [], [self.client_socket])

            for soc in read_sockets:
                self.fetch_model()
            
            if self.STOP_FLAG:
                eval = self.MODEL.evaluate()
                logging.info('Final Training eval : accuracy - %s, recall - %s, AUC - %s',eval[0][1],eval[0][2],eval[0][3])
                logging.info('Final Testing eval : accuracy - %s, recall - %s, AUC - %s',eval[1][1],eval[1][2],eval[1][3])
                
            else:
                logging.info('Model version %s fetched',self.rounds)

                self.rounds += 1
                logging.info('Training cycle %s started',self.rounds)
                self.train()

                eval = self.MODEL.evaluate()
                logging.info('Round %s - Training eval : accuracy - %s, recall - %s, AUC - %s',self.rounds,eval[0][1],eval[0][2],eval[0][3])
                logging.info('Round %s - Testing eval : accuracy - %s, recall - %s, AUC - %s',self.rounds,eval[1][1],eval[1][2],eval[1][3])

                logging.info('Training cycle %s done',self.rounds)

                logging.info('Sent local model to the server')
                self.send_model()


if __name__ == "__main__":

    from models.supervised import Model

    if 'IP' not in args.keys()  or args['IP'] == 'localhost':
        args['IP'] = socket.gethostname()

    if 'PORT' not in args.keys():
        args['PORT'] = 5000

    if 'epochs' not in args.keys():
        args['epoch'] = 10

    logging.info('Client started, graph ID %s, partition ID %s, epochs %s',args['graph_id'],args['partition_id'],args['epochs'])

    path_nodes = args['path_nodes'] + args['graph_id'] + '_nodes_' + args['partition_id'] + ".csv"
    nodes = pd.read_csv(path_nodes,index_col=0)
    nodes = nodes.astype("float32")

    path_edges = args['path_edges'] + args['graph_id'] + '_edges_' + args['partition_id'] + ".csv"
    edges = pd.read_csv(path_edges)
    edges = edges.astype({"source":"uint32","target":"uint32"})

    logging.info('Model initialized')
    model = Model(nodes,edges)
    model.initialize()

    client = Client(model,weights_path=args['path_weights'],graph_id=args['graph_id'],partition_id=args['partition_id'],epochs = int(args['epochs']) ,IP=args['IP'],PORT=int(args['PORT']))


    logging.info('Federated training started!')

    start = timer()
    client.run()
    end = timer()

    elapsed_time = end -start
    logging.info('Federated training done!')
    logging.info('Training report : Elapsed time %s seconds, graph ID %s, partition ID %s, epochs %s',elapsed_time,args['graph_id'],args['partition_id'],args['epochs'])

    
