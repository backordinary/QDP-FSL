# https://github.com/A-n-irudhT/QubitMapping/blob/2b7d4468892df29525aa41a5826b3751a445ecb4/utils.py
import random
import numpy as np
from qiskit import *
from qiskit import Aer
from qiskit import IBMQ
import matplotlib.pyplot as plt
from qiskit.circuit import Reset
from IPython.display import display
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.exceptions import CircuitError
from sympy.utilities.iterables import multiset_permutations
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library.standard_gates import (IGate, U1Gate, U2Gate, U3Gate, XGate, YGate, ZGate, HGate, SGate, SdgGate, 
                                                   TGate, TdgGate, RXGate, RYGate, RZGate, CXGate, CYGate, CZGate, CHGate, 
                                                   CRZGate, CU1Gate, CU3Gate, SwapGate, RZZGate, CCXGate, CSwapGate)

def random_circuit(num_qubits, depth, max_operands=3, measure=False, conditional=False, reset=False, seed=None):

    if max_operands < 1 or max_operands > 3:
        raise CircuitError("max_operands must be between 1 and 3")

    one_q_ops = [IGate, U1Gate, U2Gate, U3Gate, XGate, YGate, ZGate, HGate, SGate, SdgGate, TGate, TdgGate, RXGate, RYGate, RZGate]
    one_param = [U1Gate, RXGate, RYGate, RZGate, RZZGate, CU1Gate, CRZGate]
    two_param = [U2Gate]
    three_param = [U3Gate, CU3Gate]
    two_q_ops = [CXGate]
    three_q_ops = [CCXGate, CSwapGate]

    qr = QuantumRegister(num_qubits, "q")
    qc = QuantumCircuit(num_qubits)

    if measure or conditional:
        cr = ClassicalRegister(num_qubits, "c")
        qc.add_register(cr)

    if reset:
        one_q_ops += [Reset]

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    # apply arbitrary random operations at every depth
    for _ in range(depth):
        # choose either 1, 2, or 3 qubits for the operation
        remaining_qubits = list(range(num_qubits))
        rng.shuffle(remaining_qubits)
        while remaining_qubits:
            max_possible_operands = min(len(remaining_qubits), max_operands)
            num_operands = 2
            if max_possible_operands < 2:
                break
            operands = [remaining_qubits.pop() for _ in range(num_operands)]
            if num_operands == 1:
                operation = rng.choice(one_q_ops)
            elif num_operands == 2:
                operation = rng.choice(two_q_ops)
            elif num_operands == 3:
                operation = rng.choice(three_q_ops)
            if operation in one_param:
                num_angles = 1
            elif operation in two_param:
                num_angles = 2
            elif operation in three_param:
                num_angles = 3
            else:
                num_angles = 0
            angles = [rng.uniform(0, 2 * np.pi) for x in range(num_angles)]
            register_operands = [qr[i] for i in operands]
            op = operation(*angles)

            # with some low probability, condition on classical bit values
            if conditional and rng.choice(range(10)) == 0:
                value = rng.integers(0, np.power(2, num_qubits))
                op.condition = (cr, value)

            qc.append(op, register_operands)

    if measure:
        qc.measure(qr, cr)

    return qc

def get_cnot_count(qc, layout, backend):
    transpiled_circuit = transpile(qc, backend, initial_layout=layout)
    data = transpiled_circuit.count_ops()
    l = list(data.values())
    return l[0]

def get_circuit_from_x(x, backend):
    qc = QuantumCircuit(5,5)
    
    x = np.reshape(x, (-1,2))
    dimension = np.shape(x)[0]
    for i in range(dimension):
        if(x[i,0] != 0 or x[i,1] != 0):
            qc.cx(x[i,0], x[i,1])
        else:
            break
    return qc

def get_training_element(qc):
    gatedata = []
    for gate in qc.data:
        if (gate[0].name == 'cx'):
            gatedata.append(gate[1])
    gatedata = np.array(gatedata)
    l = np.shape(gatedata)[0]

    cnotarray = np.zeros([l,2], dtype=int)
    for i in range(l):
        cnotarray[i][0] = gatedata[i][0].index
        cnotarray[i][1] = gatedata[i][1].index
    return cnotarray.flatten()

def get_designspace(qc, backend):
    qubits = np.array([0, 1, 2, 3, 4])
    designspace = np.empty(120, dtype=int)
    i = 0
    for p in multiset_permutations(qubits):
        designspace[i] = get_cnot_count(qc, p, backend)
        i = i+1
        
    return np.array(designspace)

def get_optimal_layouts(designspace, margin, backend):
    accepted_range = np.amin(designspace) + margin
    indexes = np.where(designspace <= accepted_range)[0]
    
    return indexes

def one_hot_encode_layout_data(layouts):
    encoded = np.zeros(120, dtype=int)
    for i in layouts:
        encoded[i] = 1
        
    return encoded

def get_target_element(qc, backend):
    designspace = get_designspace(qc, backend)
    margin = np.std(designspace)
    indexes = get_optimal_layouts(designspace, margin, backend)
    target_element = one_hot_encode_layout_data(indexes)
    
    return target_element

def get_x_and_y_element(qubitno, depth, backend):
    qc = random_circuit(qubitno, depth)
    x = get_training_element(qc)
    y = get_target_element(qc, backend)
    
    return x, y

def get_traindata_and_targetdata(datasetsize, qubitno, backend):
    x_data = []
    y_data = []
    for i in range(datasetsize):
        x_element, y_element = get_x_and_y_element(qubitno=qubitno, depth=random.randint(8,16), backend=backend)
        x_data.append(x_element)
        y_data.append(y_element)
        
        if(i%1000 == 0):
            print("Iteration", i+1)
        
    max_length = max(len(row) for row in x_data)
    x_data = np.array([np.pad(row, (0, max_length-len(row))) for row in x_data])
    y_data = np.array(y_data)
    
    return x_data, y_data