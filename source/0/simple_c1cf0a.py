# https://github.com/MichalReznak/quep/blob/b60b3294a175af667f29793b2b7d5f1310904eff/python/simple.py
import time
from enum import Enum

from qiskit import *


class CircuitSchemaType(str, Enum):
    OpenQasm = 'OpenQasm'
    Qiskit = 'Qiskit'


class Simple:
    meta_info: dict[str, any] = None
    backend: any = None
    circuits: [QuantumCircuit] = []

    def get_meta_info(self):
        return self.meta_info

    def auth(self):
        self.backend = Aer.get_backend('aer_simulator')

    def clear_circuits(self: 'Simple'):
        self.circuits = []

    def append_circuit(self: 'Simple', circuit: str, t: str, log: bool):
        parsed_c = None

        if t == CircuitSchemaType.OpenQasm:
            parsed_c = QuantumCircuit.from_qasm_str(circuit)

        elif t == CircuitSchemaType.Qiskit:
            exec_res = {}
            exec(circuit, None, exec_res)
            parsed_c = exec_res["circ"]

        self.circuits.append(parsed_c)

        if log:
            print(parsed_c)

    def run_all(self: 'Simple') -> str:
        start = time.time()
        job = execute(self.circuits, self.backend, shots=1024, memory=True, optimization_level=0)
        end = time.time()

        self.meta_info = {
            'time': end - start
        }

        return job.result().get_counts()
