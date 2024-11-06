# https://github.com/Talkal13/Quantum/blob/ccda55776da0a3f5bd212a8566f0a1e367061a6f/QCP/protocols/QuantumTeleport.py
from ERP.states.bell_state import bell_state
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.aqua.circuits import StateVectorCircuit
from threading import Lock

class QuantumTeleport:
    def __init__(self):
        self.q = QuantumRegister(1)
        self.a = QuantumRegister(1)
        self.b = QuantumRegister(1)
        self.qc = QuantumCircuit(self.q, self.a, self.b, name="Quantum Teleport")
        erps = bell_state(2)
        self.qc.append(erps.qc, [self.a] + [self.b])
        self.lock = Lock()
        self.result = None

    def send(self, phi):
        self.lock.acquire()
        
        # encode the qubit
        self.qc.append(phi, self.q)


        self.qc.cx(self.q, self.a)
        self.qc.h(self.q)
        self.result = self.measure()
        self.lock.release()
        return self.result

    def recive(self):
        self.lock.acquire()
        if self.result is None: return
        if self.result == "00":
            pass
        elif self.result == "10":
            self.qc.x(self.b)
        elif self.result == "01":
            self.qc.z(self.b)
        else:
            self.qc.x(self.b)
            self.qc.z(self.b)
        self.lock.release()

    def measure(self):
        cb = ClassicalRegister(2)
        self.qc.add_register(cb)
        self.qc.measure(self.q, cb[0])
        self.qc.measure(self.a, cb[1])
        (result, statevector) = self.execute()
        self.reset_qc(statevector)
        return result

    def measure_b(self):
        cb = ClassicalRegister(1)
        self.qc.add_register(cb)
        self.qc.measure(self.b, cb)
        result = self.execute_counts()
        return result

    def reset_qc(self, statevector):
        st = StateVectorCircuit(statevector)
        self.qc = st.construct_circuit()
        self.q = self.qc.qubits[0]
        self.a = self.qc.qubits[1]
        self.b = self.qc.qubits[2]

    def execute(self):
        backend = Aer.get_backend("statevector_simulator")
        job = execute(self.qc, backend, shots=1)
        result = job.result()
        counts = result.get_counts()
        statevector = result.get_statevector()
        return (list(counts)[0], statevector)

    def execute_counts(self):
        backend = Aer.get_backend("qasm_simulator")
        job = execute(self.qc, backend, shots=1024)
        result = job.result()
        counts = result.get_counts()
        return counts