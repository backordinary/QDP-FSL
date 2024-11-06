# https://github.com/pablo-cardenas/QInvaders/blob/fed66c1901740fcbc67da6a88c4e750fbf7dd1d4/WEBversion/api/utils.py
from qiskit.quantum_info import Statevector
import numpy as np 
from qiskit import QuantumCircuit

db_key = {
    81:'q',
    87:'w',
    65:'a',
    83:'s',
    90:'z',
    88:'x',
}

key_list = [
    'q',
    'w',
    'a',
    's',
    'z',
    'x',
]

class InputQuantum():
    def __init__(self):
        self.__gates = 3 * [0]

    def event_quantum(self,eventKey):
        key_dict = {key: i for i, key in enumerate(key_list)}
        event_gate = key_dict.get(db_key[eventKey])
        # puerta XOR 
        self.__gates[event_gate // 2] ^= (event_gate % 2) + 1
        return self.__gates

    def update_quantum(self,eventKey):
        self.__gates = self.event_quantum(eventKey)

        circuit = QuantumCircuit(3)        
        for i, gate in enumerate(self.__gates):
            if gate & 1:
                circuit.x(i)
                gate_name = "X"
            if gate & 2:
                circuit.h(i)
                gate_name = "H"
        
        psi = Statevector(circuit)
        return (np.abs(psi.data)).astype(bool).astype(int) , np.abs(psi.data), self.__gates