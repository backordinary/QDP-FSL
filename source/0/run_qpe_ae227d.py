# https://github.com/compmolsci/Hubbard_model/blob/6a1525bf6f01a125a4e929e5e2f49b80f7d13511/run_qpe.py
import qiskit
from qiskit import QuantumCircuit
from qiskit.aqua.components.initial_states import Custom
from qiskit.aqua import QuantumInstance
from qiskit.circuit.library import QFT
from qiskit.aqua.algorithms import QPE
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.transpiler import PassManager
import inp
from qiskit.transpiler.passes import Unroller
pass_ = Unroller(['u3', 'cx'])
pm = PassManager(pass_)
#from qiskit.tools.jupyter import *
from qiskit.visualization import *
#provider = IBMQ.load_account()
#print(qiskit.__qiskit_version__)
circ=QuantumCircuit(4)
#circ.x(1)
#circ.h(0)
#circ.x(2)
#circ.y(3)
#from mol_hamilt_qiskit import qubit_op #as qubit_op1
from write_mol_hamiltonian import summed_op_home as qubit_op1
#import 
circ.x(0)
#circ.h(0)
#circ.x(1)
#circ.cx(0,1)
circ.x(2)
#circ.cx(0,2)
#circ.cx(0,3)
circ.barrier()
a = Custom(4,state = 'zero'   , circuit=circ)  
#dictb = [{'paulis': [{"coeff": {"imag": 0.00, "real": 0.456789 }, "label": "ZZXY" }]},[{"coeff": {"imag": 0.00, "real": 0.456789 }, "label": "ZZXZ" }]]
#b=Pauli.x
qubit_op = WeightedPauliOperator.from_dict(qubit_op1)
#print(qubit_op)
quantum_instance = QuantumInstance(backend=Aer.get_backend("statevector_simulator"))#, shots=10000)
n_ancillae = 6
iqft = QFT(n_ancillae).inverse()
#iqft = QFT(n_ancillae,inverse=True)
qpe = QPE(qubit_op,a, iqft, num_time_slices=2, num_ancillae=n_ancillae,expansion_mode='trotter', expansion_order=2, shallow_circuit_concat=False)
qpe_result = qpe.run(quantum_instance)
circuit = qpe.construct_circuit(measurement=True)
#unrolled_circuit = pm.run(circuit)
#gates = unrolled_circuit.count_ops() 
#print('u3=',gates['u3'],'cx=',gates['cx'])
#print(circuit.decompose())
#print(circuit.global_phase)
#print(circuit)
#print(qpe_result.eigenvalue)
print(qpe_result.eigenvalue/inp.total_time+inp.enuc)
