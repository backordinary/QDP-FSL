# https://github.com/prixt/Q-Mok/blob/3f98845e2c4c63d19d49af27fbee68595c008cf1/Scripts/Quantum.py
import qiskit
from qiskit import Aer, QuantumRegister, ClassicalRegister, QuantumCircuit, IBMQ
from qiskit.providers import JobStatus
from qiskit.providers.ibmq import least_busy
from qiskit.exceptions import QiskitError
from qiskit.compiler import transpile, assemble
from Scripts.ibmq_account import account_key

IBMQ.enable_account(account_key)
q_device = IBMQ.backends(filters=lambda x: x.configuration().n_qubits == 5 and x.configuration().memory and not x.configuration().simulator)
real_backend = least_busy(q_device)
real_name = real_backend.name()

sim_backend = Aer.get_backend('qasm_simulator')

class CircuitBuilder():
    def __init__(self, register_size):
        self.size = register_size
        self.qr = QuantumRegister(register_size)
        self.cr = ClassicalRegister(register_size)
        self.circuit = QuantumCircuit(self.qr, self.cr)
    
    def add_operation(self, op, backend):
        op_name = op[0]
        if op_name == 'h':
            self.circuit.h(self.qr[ op[1] ])
        elif op_name == 'x':
            self.circuit.x(self.qr[ op[1] ])
        elif op_name == 'gate':
            conf = backend.configuration()
            coupling_map = conf.coupling_map
            has_coupling = False
            a, b = op[1], op[2]
            if coupling_map:
                for c in coupling_map:
                    if c[0] == a and c[1] == b or c[0] == b and c[1] == a:
                        has_coupling = True
                        break
            a, b = self.qr[ a ], self.qr[ b ]
            if conf.simulator or has_coupling:
                self.circuit.cx(a,b)

                self.circuit.h(a)

                self.circuit.t(a)
                self.circuit.cx(b,a)
                self.circuit.tdg(a)

                self.circuit.h(a)

                self.circuit.cx(a,b)
            else:
                tmp = self.qr[2]
                self.circuit.swap(b, tmp)
                self.circuit.cx(a,tmp)

                self.circuit.h(a)

                self.circuit.t(a)
                self.circuit.cx(tmp,a)
                self.circuit.tdg(a)

                self.circuit.h(a)

                self.circuit.cx(a,tmp)
                self.circuit.swap(b, tmp)

    
    def build(self):
        self.circuit.barrier()
        self.circuit.measure(self.qr, self.cr)
        return self.qr, self.cr, self.circuit