# https://github.com/stewmcl/Quantum-Circuits/blob/0a49217f82a14fa64a1b41075a30528680baec9d/Circuit_Basics.py
import numpy as np
import time
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator

def basic_circuit():
    #creation of a quantum circuit acting on a quantum register of three qubits
    circ = QuantumCircuit(3)

    #Applying a Hadamard gate on qubit 0, this places the qubit in superposition
    circ.h(0)

    #Add a controlled-Not operation (CNOT \ CX) gate on the control qubit 0 and targate bit 1.
    #This puts the qubits in a Bell State
    circ.cx(0, 1)

    #Add a controlled-Not operation (CNOT \ CX) gate on the control qubit 0 and targate bit 2.
    #This puts the qubits in a GHZ state.
    circ.cx(0, 2)

    #draws the circuit
    #print(circ.draw('mpl'))
    print(circ)
    return circ

def simulating_circuits(circ):
    #setting the initial state of the simulator to the ground state using from_int
    state = Statevector.from_int(0, 2**3)

    #evolve the state by the quantum circuit
    state = state.evolve(circ)

    #state.draw('latex')

    print(state)
    state.draw('qsphere')
    state.draw('hinton')

def unitary_rep(circ):
    Un = Operator(circ)

    print(Un.data)

def QASM(circ):
    # Create a Quantum Circuit
    meas = QuantumCircuit(3, 3)
    meas.barrier(range(3))
    # map the quantum measurement to the classical bits
    meas.measure(range(3), range(3))

    # The Qiskit circuit object supports composition.
    # Here the meas has to be first and front=True (putting it before)
    # as compose must put a smaller circuit into a larger one.
    qc = meas.compose(circ, range(3), front=True)

    #drawing the circuit
    #qc.draw('mpl')
    print(qc)


def main():
    circ = basic_circuit()
    time.sleep(1)
    simulating_circuits(circ)
    time.sleep(1)
    unitary_rep(circ)
    time.sleep(1)
    QASM(circ)

if __name__ == "__main__":
    main()
