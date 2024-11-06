# https://github.com/prthmsh77/qtgame/blob/db25aaad9b6a9bc466f950b30fac5f7f1083abeb/qtgame/superposition.py
from qiskit import QuantumCircuit,Aer

def quantum_superposition():
    ckt = QuantumCircuit(1,1)
    ckt.h(0)
    ckt.measure(0,0) 
    qsim = Aer.get_backend('aer_simulator')
    result = qsim.run(ckt).result().get_counts()
    
    return result