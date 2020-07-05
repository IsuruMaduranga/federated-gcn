import socket
import pickle
import select
import time
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

class Server:

    def __init__(self, ROUNDS , HEADER_LENGTH = 10, HOST = socket.gethostname(), PORT = 5000, MAX_CONN = 5):

        # Parameters
        self.HEADER_LENGTH =  HEADER_LENGTH
        self.HOST= HOST
        self.PORT = PORT
        self.MAX_CONN = MAX_CONN
        self.ROUNDS = ROUNDS

        # Global model
        self.GLOBAL_MODEL = self.initialize_model()

        self.global_modlel_ready = False

        self.weights = []
        self.training_cycles = 0

        # List of sockets for select.select()
        self.sockets_list = []
        self.clients = {}

        # Craete server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.HOST, self.PORT))
        self.server_socket.listen(self.MAX_CONN)

        self.sockets_list.append(self.server_socket)


    def initialize_model(self):
        # Simple linear regression model
        model = Sequential()
        model.add(Dense(1, activation = 'linear', input_dim = 10))
        model.compile(optimizer=optimizers.RMSprop(lr=0.1), loss='mean_squared_error', metrics=['mae'])
        return model

    def update_model(self,new_weights):
        self.weights.append(new_weights)

        if len(self.weights) == 2:
            w1 = self.weights.pop()
            w2 = self.weights.pop()

            time.sleep(2)
            w = (w1+w2)/2
            self.GLOBAL_MODEL.set_weights(w)

            self.training_cycles += 1
            print(f"Training cycle {self.training_cycles} done!")

            for soc in self.sockets_list[1:]:
                self.send_model(soc)
        

    def send_model(self, client_socket):

        stop_flag = False
        if self.ROUNDS == self.training_cycles:
            stop_flag = True

        weights = np.array(self.GLOBAL_MODEL.get_weights())

        data = {"STOP_FLAG":stop_flag,"WEIGHTS":weights}

        data = pickle.dumps(data)
        data = bytes(f"{len(data):<{self.HEADER_LENGTH}}", 'utf-8') + data

        client_socket.sendall(data)


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

        while True:

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


if __name__ == "__main__":
    
    server = Server(ROUNDS=3)
    server.run()