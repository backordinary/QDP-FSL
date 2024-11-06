# https://github.com/Talkal13/Quantum/blob/ccda55776da0a3f5bd212a8566f0a1e367061a6f/MTF/MR.py
from typing import Callable, List
from qiskit import QuantumCircuit, QuantumRegister, execute
from qiskit.providers.aer.backends import StatevectorSimulator

class MR:
    
    def execute(self, qc: QuantumCircuit, sv: list) -> list:
        backend = StatevectorSimulator()
        sr = []
        for state in sv:
            q = QuantumRegister(len(state))
            qci = QuantumCircuit(q)
            for j in range(len(state)):
                qci.initialize(state[j], q[j])
            qci.append(qc, q)
            qci.measure_all()
            result = execute(qci, backend, shots=1).result().get_counts()
            sr.append(result)
        return sr 
    
        

    def test(self, qc: QuantumCircuit) -> bool:
        sv = self.generate()
        rv = self.execute(qc, sv)
        return self.compare(sv, rv)


    def generate(self) -> list:
        return []

    def compare(self, sv, rv) -> bool:
        return None

    def __call__(self, qc) -> bool:
        return self.test(qc)

class d_MR(MR):
    def __init__(self, mr : Callable[[QuantumCircuit, list], bool], gen : Callable[[], list]):  
        self.execute = mr
        self.generate = gen
