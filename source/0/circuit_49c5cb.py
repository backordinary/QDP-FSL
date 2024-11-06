# https://github.com/LuisMi1245/QPath-and-Snakes/blob/a215bf9fbd7e70abadceef7f2a3f49f01bd52b4d/circuit.py
from qiskit import QuantumCircuit
from PIL import Image
import numpy as np

def StartCircuit():
    global elements
    qc = QuantumCircuit(2,2)
    qc.x(1)
    elements = 1

    return qc

def MeasureCircuit(circuit):
    backend = Aer.get_backend("qasm_simulator")
    job = execute(circuit,backend, shots = 1)
    result = job.result()
    count = result.get_counts(circuit)
    return list(count.keys())[0]

def AssembleCircuit(circuit, action):
    global elements
    state_qb = 0
    if action == "Rx":
        circuit.rxx(np.pi/2,0,1)
        elements += 1
    if action == "Ry":
        circuit.ryy(np.pi/3,0,1)
        elements += 1
    if action == "Swap":
        circuit.swap(0,1)
        elements += 1
    if action == "M":
        circuit.measure((0,1),(0,1))
        state_qb = MeasureCircuit(circuit)
        circuit = StartCircuit()
    #return (circuit, state_qb)
    return circuit
    
def DrawCircuit(circuit, elements):
    circuit.draw(output = "mpl", filename = "__stored_img__/circuit.png")
    img = Image.open('__stored_img__/circuit.png')
    new_img = img.resize((168*elements,168))
    new_img.save('__stored_img__/circuit.png','png')
    return True