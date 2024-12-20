# https://github.com/AndrewLitteken/neutral-atom-compilation/blob/eec1350bf6ef6600b03683b36b4c1500c28e63db/neutralatomcompilation/compiler/compiler.py
import qiskit

from ..interaction_model import InteractionModel
from ..hardware import Hardware

class Compiler:

    def __init__(self,
                 interaction_model: InteractionModel,
                 hardware: Hardware) -> None:
        self.interaction_model = interaction_model
        self.hardware = hardware

    def compile(self,
                circuit: qiskit.circuit.QuantumCircuit,
                debug: bool=False) \
        -> qiskit.circuit.QuantumCircuit:

        '''
            Returns an equivalent circurit that is constrained to the hardware
            model and the interaction model provided to the compiler

            Qubits in the returned circuit are those used in the hardware
            specification
        '''
        raise NotImplementedError()