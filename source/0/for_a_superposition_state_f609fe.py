# https://github.com/pv6234/Qiskit-Qauntum-Teleportation/blob/0b827430ff0e0513e32489341d09da80e08b2abf/For_a_Superposition_State.py
import numpy as np
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info import random_statevector
import math
# Loading your IBM Quantum account(s)
provider = IBMQ.load_account()

state = [1/math.sqrt(3), math.sqrt(2/3)]
psi = state
display(array_to_latex(psi,prefix="|\\psi\\rangle = "))
plot_bloch_multivector(psi)
#Bloch Vector for (𝟏/√𝟑|𝟎>+√𝟐/𝟑|𝟏>):
def bellpair(qc,a,b):
  qc.h(a)
  qc.cx(a,b)
#1st bell pair
def alice(qc,psi,a):
  qc.cx(psi,a)
  qc.h(psi)
def measure_send(qc,a,b):
  qc.barrier()
  qc.measure(a,0)
  qc.measure(b,1)
def bob(qc,qubit,crz,crx):
  qc.x(qubit).c_if(crx,1)
  qc.z(qubit).c_if(crz,1)
qr=QuantumRegister(3,name='q')
crz= ClassicalRegister(1,name='crz')
crx=ClassicalRegister(1,name='crx')
qc = QuantumCircuit(qr,crz,crx)
qc.initialize(psi,0)
bellpair(qc,1,2)
qc.barrier()
alice(qc,0,1)
measure_send(qc,0,1)
bob(qc,2,crz,crx)
qc.draw()
sim = Aer.get_backend("aer_simulator")
qc.save_statevector()
out_vector=sim.run(qc).result().get_statevector()
plot_bloch_multivector(out_vector)
