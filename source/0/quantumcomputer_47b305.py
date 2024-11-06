# https://github.com/ninadgawad/quantum-computing/blob/a429164ea6909756e8c7e31540b589043fe6c1dc/Qiskit-Apps/org/ninad/QuantumComputer.py
import os
import warnings
import qiskit
import random
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import Shor
from qiskit import QuantumCircuit,Aer,execute



def callQuantumFactor():
    backend = Aer.get_backend('qasm_simulator')
    quantum_instance = QuantumInstance(backend, shots=1000)
    my_shor = Shor(N=15,a=2,quantum_instance=quantum_instance)
    return Shor.run(my_shor)

def callQuantumCoinToss():
    print('Inside callQuantumCoinToss')
    machine = os.getenv('COMPUTERNAME','MISSING')
    bits = 1
    qr = qiskit.QuantumRegister(bits)
    cr = qiskit.ClassicalRegister(bits)
    program = qiskit.QuantumCircuit(qr,cr)
    program.h(0)
    program.measure(qr,cr)
    job = qiskit.execute(program, qiskit.BasicAer.get_backend('qasm_simulator'))
    print("Calling from: "+machine)
    return job.result().get_counts()


def callClassicalCoinToss():
    print('\nInside callClassicalCoinToss')
    rounds = 1024
    recordList = []
    for amount in range(rounds):
        flip = random.randint(0,1)
        if (flip == 0):
            recordList.append("0")
        else:
            recordList.append("1")
    return print("Heads=",str(recordList.count("0")),"Tails=",str(recordList.count("1")))

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    print('------------------------- Quantum Coin Toss    ---------------------------')
    print(callQuantumCoinToss())
    print('------------------------- Classical Coin Toss  ---------------------------')
    print(callClassicalCoinToss())
    print('-----------------------------------------------------')
    print(callQuantumFactor())



