# https://github.com/shreeshjha/GroveMail/blob/5ba923e738db3bef15e73110ec1c39bfa348c27f/Sender_Teleportation.py
from qiskit import *

from channel_class import Channel

n_owner = 2
n_product = 1
owner_offset = 0
product_offset = n_owner

circ = QuantumCircuit(n_owner + n_product)

channel = Channel(product_offset, 5000, remote_port = 5001)

circ.rx(0.234,0 + channel._offset)
circ.rz(0.54,0 + channel._offset)
circ.ry(0.94,0 + channel._offset)
circ.rx(0.1,0 + channel._offset)

circ.h(1+channel._offset)
circ.cx(1+channel._offset,2+channel._offset)

circ.cx(0 + channel._offset, 1  + channel._offset)
circ.h(0 + channel._offset)

channel.send(circ,[1]) 

circ.draw(output='mpl',filename='teleport_sender.png')

