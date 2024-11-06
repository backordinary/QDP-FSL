# https://github.com/harry1357/Quantum/blob/dbd17cd49bed9ef5d331df6ac2edca02550cc383/Density%20Matrix.py
# Importing standard Qiskit Libraries

from qiskit import QuantumCircuit, transpile, Aer, IBMQ,execute
from qiskit.tools.jupyter import *

from qiskit.quantum_info import*

from qiskit.visualization import *

from ibm_quantum_widgets import *

from qiskit.providers.aer import QasmSimulator
# Loading your IBM Quantum account(s)

provider = IBMQ. load_account()
qc=QuantumCircuit(2, 2)

qc.cx(0,1)

qc.h(1)

qc.cx(1,0)

qc.draw()

from qiskit.quantum_info import Statevector
state=Statevector.from_int(1,4)
state=state.evolve(qc)

state.draw('latex')

rho_AB = DensityMatrix.from_instruction(qc)
rho_AB.draw('latex', prefix='\\rho_{AB} = ')
