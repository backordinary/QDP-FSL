# https://github.com/tgag17/BTechProject/blob/acd78762970626db191ef474b83d463bdd367d71/QSim/error/depolarization.py
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute, BasicAer
import numpy as np
import time

l = np.linspace(0, 2*np.pi, 21)
y = []
backend = BasicAer.get_backend('dm_simulator')

for k in l:
#Options & Noise goes here - Don't change options variable name & block
    options = { 
    "rotation_error": {'rx':[1.0, 0.0], 'ry':[1.0, 0.0], 'rz':[1.0,0.0]},
    "tsp_model_error": [1.0, 0.0],
    "thermal_factor": 1.0,
    "decoherence_factor": 1.0,
    "depolarization_factor": k,
    "bell_depolarization_factor": 1.0,
    "decay_factor": 1.0,
    }
   
    s = []
    qc = QuantumCircuit()
    q = QuantumRegister(2, 'q')
    c = ClassicalRegister(2, 'c')

    qc.add_register(q)
    qc.add_register(c)

    a = np.pi/10

    qc.h(0)
    qc.cx(0, 1)
    qc.ry(a, 0)

    ##########################

    qc1 = QuantumCircuit()
    q1 = QuantumRegister(2, 'q')
    c1 = ClassicalRegister(2, 'c')

    qc1.add_register(q1)
    qc1.add_register(c1)

    qc1.h(0)
    qc1.cx(0, 1)
    qc1.ry(a, 0)
    qc1.h(0)

    ############################

    qc2 = QuantumCircuit()
    q2= QuantumRegister(2, 'q')
    c2 = ClassicalRegister(2, 'c')

    qc2.add_register(q2)
    qc2.add_register(c2)


    qc2.h(0)
    qc2.cx(0, 1)
    qc2.ry(a, 0)
    qc2.h(1)

    #############################

    qc3 = QuantumCircuit()
    q3 = QuantumRegister(2, 'q')
    c3 = ClassicalRegister(2, 'c')

    qc3.add_register(q3)
    qc3.add_register(c3)

    qc3.h(0)
    qc3.cx(0, 1)
    qc3.ry(a, 0)
    qc3.h(0)
    qc3.h(1)

    qc.measure(q[0:2],c[0:2],basis='Expect',add_param='ZZ')
    qc1.measure(q1[0:2],c1[0:2],basis='Expect',add_param='ZZ')
    qc2.measure(q2[0:2],c2[0:2],basis='Expect',add_param='ZZ')
    qc3.measure(q3[0:2],c3[0:2],basis='Expect',add_param='ZZ')
   
    run = execute(qc,backend = backend, **options)
    result = run.result()
    s.append(result['results'][0]['data']['Pauli_string_expectation'])
    run1 = execute(qc1,backend = backend, **options)
    result1 = run1.result()
    s.append(result1['results'][0]['data']['Pauli_string_expectation'])
    run2 = execute(qc2,backend = backend, **options)
    result2 = run2.result()
    s.append(result2['results'][0]['data']['Pauli_string_expectation'])
    run3 = execute(qc3,backend = backend, **options)
    result3 = run3.result()
    s.append(result3['results'][0]['data']['Pauli_string_expectation'])
    y.append(s[0]+s[1]-s[2]+s[3])
print(y)
