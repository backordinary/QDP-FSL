# https://github.com/Talkal13/Quantum/blob/ccda55776da0a3f5bd212a8566f0a1e367061a6f/QCP/ERP/ERPS.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.aqua.circuits import StateVectorCircuit
from threading import Lock


class ERPS:

    '''Abstract class to generate ERP pairs, since qiskit does not allow for sequential measures on different qubits without reseting the system 
    a workaround has been found to collapse the system on the first measure and use its result for the following to obtain the same functionality as a ERP Source'''

    def __init__(self, n=2):
        self.q = QuantumRegister(2)
        self.n = 2
        self.qc = QuantumCircuit(self.q, name="ERP Source")
        self.lock = Lock()

    def prepare(self):
        raise NotImplementedError

    def measure(self, qbits):
        self.lock.acquire()
        i = 0
        cb = ClassicalRegister(len(qbits))
        self.qc.add_register(cb)
        for qb in qbits:
            self.qc.measure(self.q[qb], cb[i])
            i += 1
        (result, statevector) = self.execute()
        self.reset_qc(statevector)
        self.lock.release()
        return result


    def reset_qc(self, statevector):
        st = StateVectorCircuit(statevector)
        self.qc = st.construct_circuit(register=self.q)
        


    def execute(self):
        backend = Aer.get_backend("statevector_simulator")
        job = execute(self.qc, backend, shots=1)
        result = job.result()
        counts = result.get_counts()
        statevector = result.get_statevector()
        return (list(counts)[0], statevector)

