import socket
import pickle
import select
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

HEADER_LENGTH = 10

IP = socket.gethostname()
PORT = 5000

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((IP, PORT))

MODEL = None
STOP_FLAG = False
rounds = 0

###########################################################################

def send_model(client_socket):
    data = np.array(MODEL.get_weights())
    data = pickle.dumps(data)
    data = bytes(f"{len(data):<{HEADER_LENGTH}}", 'utf-8') + data
    client_socket.sendall(data)


def receive(client_socket):
    try:


        message_header = client_socket.recv(HEADER_LENGTH)
        if not len(message_header):
            return False

        message_length = int(message_header.decode('utf-8').strip())

        full_msg = b''
        while True:
            msg = client_socket.recv(message_length)

            full_msg += msg

            if len(full_msg) == message_length:
                break
        
        data = pickle.loads(full_msg)

        STOP_FLAG = data["STOP_FLAG"]

        return data["WEIGHTS"]


    except Exception as e:
        print(e)


def fetch_model(client_socket):
    global MODEL

    data = receive(client_socket)
       
    if MODEL == None:
        model = Sequential()
        model.add(Dense(1, activation = 'linear', input_dim = 10))
        model.compile(optimizer=optimizers.RMSprop(lr=0.1), loss='mean_squared_error', metrics=['mae'])
        MODEL = model

    MODEL.set_weights(data)

def train():
    global MODEL

    df = pd.read_csv("data1.csv")
    x = df[["x1","x2","x3","x4","x5","x6","x7","x8","x9","x10"]]
    y = df["y"]

    MODEL.fit(x,y,epochs =200,batch_size = 50)


while not STOP_FLAG:

    read_sockets, _, exception_sockets = select.select([client_socket], [], [client_socket])

    for soc in read_sockets:
        fetch_model(soc)

    print(f"Model version: {rounds} fetched")

    rounds += 1
    print(f"Training cycle: {rounds}")
    train()

    print(f"Sent local model")
    send_model(client_socket)

print("Training Done")
