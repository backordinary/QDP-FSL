# https://github.com/shreeshjha/GroveMail/blob/5ba923e738db3bef15e73110ec1c39bfa348c27f/Reciever.py
from qiskit import *
import matplotlib.pyplot as plt

from channel_class import Channel

circ_reciever = QuantumCircuit(3)

reciever_channel = Channel(myport = 5001, remote_port = 5000)
circ_reciever, offset = reciever_channel.receive(circ_reciever)#,to_tpc)

circ_reciever.x(0+offset)

import time
time.sleep(2)
to_tpc = reciever_channel.send(circ_reciever,[1])
circ_reciever.draw()

