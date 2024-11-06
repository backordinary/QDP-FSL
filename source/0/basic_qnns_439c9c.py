# https://github.com/Rin-The-QT-Bunny/quantum_networks/blob/4ff59a4fd107a07ef1253d59827701940907af7d/basic_qnns.py
import numpy as np

from qiskit import Aer, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.opflow import StateFn, PauliSumOp,AerPauliExpectation,ListOp,Gradient

from qiskit.utils import QuantumInstance,algorithm_globals

algorithm_globals.random_seed = 42

# set method to calculate expeceted values
expval  = AerPauliExpectation()

# define gradient method
gradient = Gradient()

# define quantum instances (statevector and sample based)
qi_sv = QuantumInstance(Aer.get_backend("aer_simulator_statevector"))

# we set shots to 10 as this will determine the number of samples later on.
qi_qasm = QuantumInstance(Aer.get_backend("aer_simulator"),shots = 10)

from qiskit_machine_learning.neural_networks import OpflowQNN

# constuct parameterized circuit
params1 = [Parameter("input1"),Parameter("weights1")]
qc1 = QuantumCircuit(1)
qc1.h(0)
qc1.ry(params1[0],0)
qc1.rx(params1[1],0)
qc_sfn1 = StateFn(qc1)

# construct the cost operator
H1 = StateFn(PauliSumOp.from_list([("Z",1.0),("X",1.0)]))

# combine operator and circuit to objective function
op1 = ~H1 @ qc_sfn1
print(op1)
# construct opflowqnn with the operator, the input parameters, the weight parameters
qnn1 = OpflowQNN(op1, [params1[0]], [params1[1]], expval, gradient, qi_sv)

input1 = algorithm_globals.random.random(qnn1.num_inputs)
weights1 = algorithm_globals.random.random(qnn1.num_weights)

out = qnn1.forward(input1,weights1)
print(out)
back = qnn1.backward(input1, weights1)
print(back)
print("End of the calculation")