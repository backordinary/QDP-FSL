# https://github.com/kerenavnery/qmail/blob/b6b329f355f8645b67358ec81a79766fcfb3a0d1/Alice.py
from qiskit import *

from channel_class import Channel

n_master = 2
n_slave = 1
master_offset = 0
slave_offset = n_master

circ = QuantumCircuit(n_master + n_slave)

channel = Channel(slave_offset, 5000, remote_port = 5001)

## Master
circ.x(0 + channel._offset)
#circ.cx(0 + channel._offset, 1  + channel._offset)
#irc.h(1 + channel._offset)


to_tpc = channel.send(circ,[1])  ## TODO: remove
circ.draw()

"""
#Bob Part
circ_bob = QuantumCircuit(3)

bob_channel = Channel()
circ_bob, offset = bob_channel.receive(circ_bob)#,to_tpc)
circ_bob.draw()

# Add new gates to circ2
circ_bob.h(0+offset)
#circ_bob.cx(0+offset, 1+offset)
#psi2 = Statevector.from_instruction(circ_bob)

to_tpc = bob_channel.send(circ_bob,[1])
circ_bob.draw()
"""

#Alice Part
circ_alice = QuantumCircuit(3)

alice_channel = channel #Channel()
alice_channel._slave_offset = 0
circ_alice , offset = alice_channel.receive(circ_alice)#,to_tpc)
circ_alice.draw(output='mpl',filename='test.png')

