# https://github.com/wweronika/qram-grover-search/blob/78d13a07d0782452d804eeec662ce7170b476525/qram_grover.py
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister

from qiskit.tools.monitor import job_monitor
from qiskit import execute, IBMQ

import numpy as np

from qram import QRAM
from mcz import MCZ

class QRAMGrover:
    def __init__(self, numbers, n_address_bits, n_memory_cell_bits) -> None:

        self._numbers = numbers
        # Toggle to enable additional qubits
        # n_address_bits += 1
        self._n_trigger_bits = 2 ** n_address_bits
        self._n_memory_cell_bits = n_memory_cell_bits
        self._n_address_bits = n_address_bits
        self._n_qubits = n_address_bits + self._n_trigger_bits + self._n_trigger_bits * n_memory_cell_bits + n_memory_cell_bits
        print("N " + str(self._n_qubits))

        self._q_reg = QuantumRegister(self._n_qubits)
        self._c_reg = ClassicalRegister(self._n_qubits)
        self._qc = QuantumCircuit(self._q_reg, self._c_reg)

        # First n qubits
        self._address_qubits = [i for i in range(n_address_bits)]
        # Last n qubits
        self._output_qubits = [i for i in range(self._n_qubits - n_memory_cell_bits, self._n_qubits)]
        # Memory qubits
        self._memory_qubits = [i for i in range(n_address_bits + self._n_trigger_bits, n_address_bits + self._n_trigger_bits + self._n_trigger_bits * n_memory_cell_bits )]


        # self._oracle = AdjacentBitsOracle(n_memory_cell_bits)

        self.build_circuit()

    """
    Applies Hadamard gate to all address qubits (initial step of Grover's algorithm).
    """
    def _initialize_h(self) -> None:
        self._qc.h(self._address_qubits)

    """
    Constructs the circuit for generalised QRAM-based Grover's search. 
    High-level structure of the circuit is, from left to right:
    1. Hadamard on address qubits
    2. Bucket-brigade QRAM
    3. Oracle on QRAM output qubits
    4. Diffuser on QRAM output qubits 
    """
    def build_circuit(self) -> None:
        self._initialize_h()
        self._qc.x(self._n_address_bits)
        self.initialize_memory()
        qram = QRAM(self._n_address_bits, self._n_memory_cell_bits, self._numbers)
        qram_instruction = qram.get_instruction()
        print(self._qc.qubits)
        self._qc.append(qram_instruction, self._qc.qubits)
        self.apply_oracle()
        qram_inverse_instruction = qram.get_inverse_instruction()
        self._qc.append(qram_inverse_instruction, self._qc.qubits)
        self.apply_diffuser()
        # self._qc.append(qram_instruction, self._qc.qubits)

    def apply_diffuser(self):
        self._qc.h(self._address_qubits)
        self._qc.x(self._address_qubits)

        mcz_gate = MCZ().get_mcz(self._n_address_bits)
        self._qc.append(mcz_gate, self._address_qubits)

        self._qc.x(self._address_qubits)
        self._qc.h(self._address_qubits)

    def apply_oracle(self):
        if self._n_memory_cell_bits == 2:
            self._qc.cx(self._output_qubits[0], self._output_qubits[1])
            self._qc.z(self._output_qubits[1])
            self._qc.cx(self._output_qubits[0], self._output_qubits[1])
        else:
            mcz_gate = MCZ.get_mcz(self._n_memory_cell_bits)
            mcxzx_gate = MCZ.get_mcxzx(self._n_memory_cell_bits)
            
            self._qc.x(self._output_qubits[0:self._n_memory_cell_bits-1:2])
            if self._n_memory_cell_bits % 2 == 0:
                self._qc.append(mcz_gate, self._output_qubits)
            else:
                self._qc.append(mcxzx_gate, self._output_qubits)
            self._qc.x(self._output_qubits[0:self._n_memory_cell_bits-1:2])

            self._qc.x(self._output_qubits[1:self._n_memory_cell_bits-1:2])
            if self._n_memory_cell_bits % 2 == 0:
                self._qc.append(mcxzx_gate, self._output_qubits)
            else:
                self._qc.append(mcz_gate, self._output_qubits)
            self._qc.x(self._output_qubits[1:self._n_memory_cell_bits-1:2])

    def initialize_memory(self):
        for i, number in enumerate(self._numbers):
            if number > 2 ** self._n_memory_cell_bits:
                print("Number too large: " + str(number), + ", please enter a smaller one.")
            else:
                j = 0
                while number > 0:
                    if number % 2 == 1:
                        self._qc.x(self._n_address_bits + self._n_trigger_bits + i*self._n_memory_cell_bits + j)
                    number = number // 2
                    j += 1


    def get_circuit(self):
        return self._qc

# Example input

n_address_bits = 2
n_memory = 3
sol = QRAMGrover([1,0,5], n_address_bits, n_memory)

provider = IBMQ.get_provider(hub='ibm-q')

backend = provider.get_backend('ibmq_qasm_simulator')
print('\nExecuting job....\n')
qc = sol.get_circuit()

for i in range(n_address_bits):
    qc.measure(i,i)

print(qc)
job = execute(qc, backend, shots=5000)

job_monitor(job)
counts = job.result().get_counts()

print('RESULT: ',counts,'\n')