# https://github.com/Salvo1108/Quantum_Programming_ASService/blob/b80462df050218490e521ea16b4ac88008b242b1/gateway/walkSearch.py
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, IBMQ
# Loading your IBM Q account(s)

# Shift operator function for 4d-hypercube
def shift_operator(circuit):
    circuit.x(4)
    if 0%2==0:
        circuit.x(5)
    circuit.ccx(4,5,0)
    circuit.x(4)
    if 1%2==0:
        circuit.x(5)
    circuit.ccx(4,5,1)
    circuit.x(4)
    if 2%2==0:
        circuit.x(5)
    circuit.ccx(4,5,2)
    circuit.x(4)
    if 3%2==0:
        circuit.x(5)
    circuit.ccx(4,5,3)

def ws_algorithm(qubits):
    one_step_circuit = QuantumCircuit(qubits, name=' ONE STEP')
    # Coin operator
    one_step_circuit.h([4,5])
    one_step_circuit.z([4,5])
    one_step_circuit.cz(4,5)
    one_step_circuit.h([4,5])
    one_step_circuit.draw()

    shift_operator(one_step_circuit)
    one_step_gate = one_step_circuit.to_instruction()

    return one_step_circuit
