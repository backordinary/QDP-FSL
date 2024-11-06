# https://github.com/pc2/rdmft_on_qc/blob/00a461db1fe12e771e144f854ab65316d3b015d1/utils/perturb.py
import numpy as np
from qiskit import *
from qiskit import Aer
from qiskit.circuit import Gate
from math import pi
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller
from qiskit.circuit.library import TwoLocal,EfficientSU2
import os
import copy
import math
import sys
import time
from scipy import sparse
from scipy.optimize import minimize,BFGS
from math import pi
import numpy as np
import dotenv
import matplotlib.pyplot as plt
import qiskit
from qiskit.circuit.library import TwoLocal,EfficientSU2
from qiskit.circuit import Parameter, ParameterVector, ParameterExpression
from qiskit import execute
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import L_BFGS_B,SPSA,COBYLA,QNSPSA
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.opflow.state_fns import CircuitStateFn
from qiskit.quantum_info import Pauli
from qiskit.opflow.gradients import Gradient, NaturalGradient, QFI, Hessian
from qiskit.opflow import Z, X, I, StateFn, CircuitStateFn, SummedOp
import qiskit_nature
from qiskit import IBMQ, assemble, transpile,Aer
from qiskit import BasicAer
from qiskit_nature.circuit.library import HartreeFock
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper, BravyiKitaevMapper    
from qiskit_nature.operators.second_quantization import FermionicOp
import pylatexenc
import configparser
import itertools
from qiskit.opflow import PauliSumOp

def paulis_for_op(m,qubit_converter,qc,num_particles=0,tmeasure=True):
    m_op = qubit_converter.convert(m,num_particles=num_particles)
    pauliops=[]
    coeff=[]
    const=0

    #build measuring programs here
    p=m_op.to_pauli_op()
    for op in p:
        if op.to_circuit().depth()>0:
            coeff.append(op.coeff)
            pauliops.append(op)
        else:
            const+=op.coeff
    return {"pauliops":pauliops,"coeff":coeff,"const":const}

def getexp(circ,sp):
    backend_check = BasicAer.get_backend('statevector_simulator')
    job=execute(circ,backend_check)
    result=job.result()
    psi=result.get_statevector()
    return np.dot(np.conj(psi),sp.dot(psi)).real

nq=8
N=(4,4)
seed=238745

ansatz = TwoLocal(nq, reps=3, rotation_blocks='ry', entanglement_blocks='cx', entanglement='linear', parameter_prefix = 'y',insert_barriers=False)
q = QuantumRegister(nq)
c = ClassicalRegister(nq)
qc=QuantumCircuit(q,c)
qc=qc.compose(ansatz)
qc=qc.decompose()
print(qc)


np.random.seed(seed)
initial_point = np.random.random(ansatz.num_parameters)
circ=ansatz.bind_parameters(initial_point)
print(circ)

qubit_converter = QubitConverter(mapper=JordanWignerMapper(),two_qubit_reduction=False)
#qubit_converter = QubitConverter(mapper=ParityMapper(),two_qubit_reduction=False)

c_ops=[]
for i in range(nq):
    c_ops.append(FermionicOp("-_"+str(i),register_length=nq))


HF=~c_ops[0] @ ~c_ops[1]
print(HF)
PHF=paulis_for_op(HF,qubit_converter,qc,num_particles=N,tmeasure=False)
print(PHF)

bitstr = [True,True,True,False,True,False,False,False] 
label = ["+" if bit else "I" for bit in bitstr]
print("label=",label)
bitstr_op = FermionicOp("".join(label))
print("bitstr_op",bitstr_op)
qubit_op: PauliSumOp = qubit_converter.convert_match(bitstr_op)
print("qubit_op",qubit_op)

# Add gates in the right positions: we are only interested in the `X` gates because we want
# to create particles (0 -> 1) where the initial state introduced a creation (`+`) operator.
for i, bit in enumerate(qubit_op.primitive.table.X[0]):
    if bit:
        print("x at",i)

hf=HartreeFock(nq,N,qubit_converter)
print(hf)



hf=QuantumCircuit(q,c)
hf.x(0)
hf.x(1)
hf.x(2)
hf.x(4)

qc=hf
print(qc)

for i in range(nq):
    for j in range(nq):
        if i>j:
            continue
        #real part
        m=~c_ops[i]@ c_ops[j]
        cop=0.5*(m+~m)
        op=qubit_converter.convert(cop,num_particles=N)
        sp=sparse.csr_matrix(op.to_matrix())
        #c['pauli']=paulis_for_op(c['op'],qubit_converter,qc,num_particles)
        v1=getexp(qc,sp)
        if i!=j:
            m=~c_ops[i]@ c_ops[j]
            cop=0.5/1j*(m-~m)
            op=qubit_converter.convert(cop,num_particles=N)
            sp=sparse.csr_matrix(op.to_matrix())
            #c['pauli']=paulis_for_op(c['op'],qubit_converter,qc,num_particles)
            v2=getexp(qc,sp)
            print("rdm",i,j,v1,v2)
        else:
            print("rdm",i,j,v1)

quit()

#Changing the simulator 
#backend = Aer.get_backend('unitary_simulator')
backend = Aer.get_backend('statevector_simulator')

#The circuit without measurement

circ = QuantumCircuit(4)

circ.x(0)
circ.x(1)
circ.x(2)
circ.draw()
print(circ)
print(circ.decompose())

#job execution and getting the result as an object
job = execute(circ, backend)
result = job.result()

#pass_ = Unroller(['u1', 'u2', 'u3', 'cx'])
#pm = PassManager(pass_)
#new_circ = pm.run(circ)
#print(new_circ)

#get the unitary matrix from the result object
#u1=result.get_unitary(circ, decimals=3)
v1=result.get_statevector(circ, decimals=3)

circ = QuantumCircuit(4)

circ.x(0)
circ.x(1)
circ.x(3)
circ.draw()
print(circ)

#job execution and getting the result as an object
job = execute(circ, backend)
result = job.result()

#get the unitary matrix from the result object
#u2=result.get_unitary(circ, decimals=3)
v2=result.get_statevector(circ, decimals=3)
#print(u1)
#print(u2)
#print(u1-u2)


circ = QuantumCircuit(4)

circ.x(0)
circ.x(1)
circ.x(3)
circ.h(2)
circ.cnot(2,3)
circ.draw()
print(circ)

#job execution and getting the result as an object
job = execute(circ, backend)
result = job.result()

#get the unitary matrix from the result object
#u2=result.get_unitary(circ, decimals=3)
v3=result.get_statevector(circ, decimals=3)

for i in range(len(v1)):
    print(i,v1[i],v2[i],v3[i])

pass_ = Unroller(['u1', 'u2', 'u3', 'cx'])
pm = PassManager(pass_)
new_circ = pm.run(circ)
print(new_circ)
