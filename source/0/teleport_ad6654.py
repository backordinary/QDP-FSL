# https://github.com/ntallar/quantum-teleport/blob/9d5ae6d0ac60922270904f00e759655f4e61cbba/teleport.py
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit
import numpy as np
from utils import *

def initTeleport(qc, phi, suffix):
    alice = QuantumRegister(1, 'alice' + suffix)
    bob = QuantumRegister(1, 'bob' + suffix)
    phiMeasurement = ClassicalRegister(1, 'phiMeasurement' + suffix)
    aliceMeasurement = ClassicalRegister(1, 'aliceMeasurement' + suffix)

    qc.add_register(alice)
    qc.add_register(bob)
    qc.add_register(phiMeasurement)
    qc.add_register(aliceMeasurement)
    return (alice, bob, phiMeasurement, aliceMeasurement)

def runTeleport(qc, phi, alice, bob, phiMeasurement, aliceMeasurement):
    # Built the entangled pair that alice and bob will share
    transformCompToBell(qc, alice[0], bob[0])

    # Interact phi with alice qubit
    transformBellToComp(qc, phi[0], alice[0])

    qc.barrier()

    # Alice measures her qubits
    qc.measure(alice[0], aliceMeasurement[0])
    qc.measure(phi[0], phiMeasurement[0])

    # Bob interacts with his qubits
    qc.x(bob[0]).c_if(aliceMeasurement, 1)
    qc.z(bob[0]).c_if(phiMeasurement, 1)

    return bob

def teleportOneQubit(stateToTeleportBuilder):
    qc = QuantumCircuit()

    # State to teleport
    phi = QuantumRegister(1, 'phi')
    qc.add_register(phi)

    stateToTeleportBuilder(qc, phi)

    # Create circuit to teleport
    (alice, bob, phiMeasurement, aliceMeasurement) = initTeleport(qc, phi, "")
    qc.barrier()
    resultState = runTeleport(qc, phi, alice, bob, phiMeasurement, aliceMeasurement)

    # Measure the result
    resultMeasurement = ClassicalRegister(1, 'resultMeasurement')
    qc.add_register(resultMeasurement)
    qc.barrier()
    qc.measure(resultState, resultMeasurement[0])

    # Show the quantum circuit before executing
    drawCircuit(qc)

    # Execute the circuit
    classicalValuesResult = executeQuantumCircuit('qasm_simulator', 1024, qc)
    print(parseGetCounts(classicalValuesResult, 0))

def teleportTwoQubits(stateToTeleportBuilder):
    qc = QuantumCircuit()

    # State to teleport
    phi = QuantumRegister(1, 'phi')
    qc.add_register(phi)

    rho = QuantumRegister(1, 'rho')
    qc.add_register(rho)

    stateToTeleportBuilder(qc, phi, rho)

    # Create circuit to teleport
    (alice1, bob1, phiMeasurement1, aliceMeasurement1) = initTeleport(qc, phi, "1")
    (alice2, bob2, phiMeasurement2, aliceMeasurement2) = initTeleport(qc, rho, "2")
    qc.barrier()
    resultState1 = runTeleport(qc, phi, alice1, bob1, phiMeasurement1, aliceMeasurement1)
    qc.barrier()
    resultState2 = runTeleport(qc, rho, alice2, bob2, phiMeasurement2, aliceMeasurement2)

    # Measure the result
    resultMeasurement = ClassicalRegister(2, 'resultMeasurement')
    qc.add_register(resultMeasurement)
    qc.barrier()
    qc.measure(resultState1, resultMeasurement[0])
    qc.measure(resultState2, resultMeasurement[1])

    # Show the quantum circuit before executing
    drawCircuit(qc)

    # Execute the circuit
    classicalValuesResult = executeQuantumCircuit('qasm_simulator', 1024, qc)
    print(parseGetCounts(classicalValuesResult, 0))


# Simple testing to see how teleporting works
if __name__ == "__main__":

    teleportOneQubit(lambda qc, phi : constructMixedState(qc, phi))

    teleportTwoQubits(lambda qc, phi, rho : constructBellBase(qc, phi, rho, True, True))