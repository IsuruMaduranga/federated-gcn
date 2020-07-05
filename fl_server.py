import socket
import pickle
import select
import time
import numpy as np
import pandas as pd
import sys

class Server:

    def __init__(self, MODEL, ROUNDS , weights_path, graph_id, MAX_CONN = 2, IP= socket.gethostname(), PORT = 5000, HEADER_LENGTH = 10 ):

        # Parameters
        self.HEADER_LENGTH =  HEADER_LENGTH
        self.IP = IP
        self.PORT = PORT
        self.MAX_CONN = MAX_CONN
        self.ROUNDS = ROUNDS

        self.weights_path = weights_path
        self.graph_id = graph_id

        # Global model
        self.GLOBAL_WEIGHTS = MODEL.get_weights()

        self.global_modlel_ready = False

        self.weights = []
        self.training_cycles = 0

        self.stop_flag = False

        # List of sockets for select.select()
        self.sockets_list = []
        self.clients = {}

        # Craete server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.IP, self.PORT))
        self.server_socket.listen(self.MAX_CONN)

        self.sockets_list.append(self.server_socket)

    def update_model(self,new_weights):
        self.weights.append(new_weights)

        if len(self.weights) == self.MAX_CONN:

            new_weights = np.mean(self.weights, axis=0)
            self.weights = []

            #self.GLOBAL_MODEL.set_weights(new_weights)
            self.GLOBAL_WEIGHTS = new_weights

            self.training_cycles += 1

            # weights file name : global_weights_graphid.npy
            weights_path = self.weights_path + 'global_weights_' + self.graph_id + ".npy"
            np.save(weights_path,new_weights)
            
            print(f"Training cycle {self.training_cycles} done!")

            for soc in self.sockets_list[1:]:
                self.send_model(soc)
        

    def send_model(self, client_socket):

        if self.ROUNDS == self.training_cycles:
            self.stop_flag = True

        weights = np.array(self.GLOBAL_WEIGHTS)

        data = {"STOP_FLAG":self.stop_flag,"WEIGHTS":weights}

        data = pickle.dumps(data)
        data = bytes(f"{len(data):<{self.HEADER_LENGTH}}", 'utf-8') + data

        client_socket.sendall(data)
        print('Sent global model to: {}'.format(self.clients[client_socket]))


    def receive(self, client_socket):
        try:
            
            message_header = client_socket.recv(self.HEADER_LENGTH)

            if not len(message_header):
                print('Client closed connection from: {}'.format(self.clients[client_socket]))
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
            print('Client closed connection from: {}'.format(self.clients[client_socket]))
            return False


    def run(self):

        while not self.stop_flag:

            read_sockets, write_sockets, exception_sockets = select.select(self.sockets_list, [], self.sockets_list)

            for notified_socket in read_sockets:

                if notified_socket == self.server_socket:

                    client_socket, client_address = self.server_socket.accept()
                    self.sockets_list.append(client_socket)
                    self.clients[client_socket] = client_address

                    print('Accepted new connection from {}:{}'.format(*client_address))

                    self.send_model(client_socket)

                else:

                    message = self.receive(notified_socket)

                    if message is False:
                        self.sockets_list.remove(notified_socket)
                        del self.clients[notified_socket]

                        continue
                    
                    print('Recieved model from {}:{}'.format(*self.clients[notified_socket]))
                    self.update_model(message)

            for notified_socket in exception_sockets:
                self.sockets_list.remove(notified_socket)
                del self.clients[notified_socket]

        print("Federated training done!")


if __name__ == "__main__":

    from models.unsupervised import Model

    arg_names = [
        'path_weights',
        'path_data',
        'graph_id',
        'partition_id',
        'num_clients',
        'num_rounds',
        'IP',
        'PORT'
        ]

    args = dict(zip(arg_names, sys.argv[1:]))

    if 'IP' not in args.keys()  or args['IP'] == 'localhost':
        args['IP'] = socket.gethostname()

    if 'PORT' not in args.keys():
        args['PORT'] = 5000

    path_nodes = args['path_data'] + args['graph_id'] + '_nodes_' + args['partition_id'] + ".csv"
    nodes = pd.read_csv(path_nodes,index_col=0)

    path_edges = args['path_data'] + args['graph_id'] + '_edges_' + args['partition_id'] + ".csv"
    edges = pd.read_csv(path_edges)
   
    model = Model(nodes,edges)
    model.initialize()
    
    server = Server(model,ROUNDS=args['num_rounds'],weights_path=args['path_weights'],graph_id=args['graph_id'],MAX_CONN=int(args['num_clients']),IP=args['IP'],PORT=int(args['PORT']))

    del nodes
    del edges
    del model
    
    server.run()