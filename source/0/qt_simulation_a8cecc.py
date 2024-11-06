# https://github.com/Talkal13/Quantum/blob/ccda55776da0a3f5bd212a8566f0a1e367061a6f/QCP/simulations/QT_simulation.py
import threading
from protocols.QuantumTeleport import QuantumTeleport
from time import sleep
from qiskit import QuantumCircuit, QuantumRegister
from math import pi

class agent(threading.Thread):
    def __init__(self, name, protocol):
        threading.Thread.__init__(self)
        self.name = name
        self.protocol = protocol


class Alice(agent):
    def __init__(self, name, protocol):
        super().__init__(name, protocol)

    def run(self):
        res = self.protocol.send(self.set_state())
        print("Alice has teleported the data with result: " + res)

    def set_state(self):
        q = QuantumRegister(1)
        qc = QuantumCircuit(q)

        # Set up state
        qc.ry(2/3 * pi, q)
        return qc


class Bob(agent):
    def __init__(self, name, protocol):
        super().__init__(name, protocol)

    def run(self):
        self.protocol.recive()
        print("Bob has recive the qubit")
        print(self.protocol.measure_b())

def exec():
    protocol = QuantumTeleport()
    alice = Alice("alice", protocol)
    bob = Bob("bob", protocol)
    alice.start()
    sleep(5)
    bob.start()
    alice.join()
    bob.join()