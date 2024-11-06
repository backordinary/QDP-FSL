# https://github.com/ntallar/quantum-teleport/blob/9d5ae6d0ac60922270904f00e759655f4e61cbba/swaptest.py
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit
import numpy as np
from utils import *

def initSwapTest(qc):
    aux = QuantumRegister(1, 'aux')
    auxMeasurement = ClassicalRegister(1, 'auxMeasurement')

    qc.add_register(aux)
    qc.add_register(auxMeasurement)
    return (aux, auxMeasurement)

# @require len(phis)==len(rhos)
def runSwapTest(qc, phis, rhos, aux, auxMeasurement):
    qc.h(aux[0])
    for (phi, rho) in zip(phis, rhos):
        qc.cswap(aux[0], phi, rho)
    qc.h(aux[0])

    qc.barrier()

    qc.measure(aux[0], auxMeasurement[0])

def swapTestTwoQubits(qubitBuilders):
    qc = QuantumCircuit()

    phi = QuantumRegister(1, 'phi')
    rho = QuantumRegister(1, 'rho')
    qc.add_register(phi)
    qc.add_register(rho)

    qubitBuilders(qc, phi, rho)

    # Create circuit to check equality
    (aux, auxMeasurement) = initSwapTest(qc)
    qc.barrier()
    runSwapTest(qc, [phi], [rho], aux, auxMeasurement)

    # Show the quantum circuit before executing
    drawCircuit(qc)

    # Execute the circuit
    classicalValuesResult = executeQuantumCircuit('qasm_simulator', 1024, qc)
    print(parseGetCounts(classicalValuesResult, 0))

# Simple testing to see how swap test works
if __name__ == "__main__":

    swapTestTwoQubits(lambda qc, phi, rho : constructBellBase(qc, phi, rho, False, False))