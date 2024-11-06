# https://github.com/amirarfan/MATH275/blob/b57506323fe024749af6f3c8602af3de2d2ab4b0/QSVM/qsvm_mnist.py
# https://qiskit.org/documentation/stable/0.26/tutorials/machine_learning/02_qsvm_multiclass.html
# Updated by: Amir Arfan for MATH275 Projec

#%%
import numpy as np

from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.components.multiclass_extensions import AllPairs
from qiskit.aqua.utils.dataset_helper import get_feature_dimension

from qiskit.ml.datasets import digits

#%%
n = 2  # dimension of each data point
sample_Total, training_input, test_input, class_labels = digits(
    training_size=500, test_size=100, n=2, plot_data=True
)


temp = [test_input[k] for k in test_input]
total_array = np.concatenate(temp)


aqua_globals.random_seed = 10598

#%%
backend = BasicAer.get_backend("qasm_simulator")
feature_map = PauliFeatureMap(
    feature_dimension=get_feature_dimension(training_input),
    reps=2,
    entanglement="full",
)
svm = QSVM(
    feature_map,
    training_input,
    test_input,
    total_array,
    multiclass_extension=AllPairs(),
)
quantum_instance = QuantumInstance(
    backend,
    shots=1024,
    seed_simulator=aqua_globals.random_seed,
    seed_transpiler=aqua_globals.random_seed,
)
#%%
result = svm.run(quantum_instance)


# %%
for k, v in result.items():
    print(f"{k} : {v}")

# %%
print(f"Accuracy: {result['test_accuracy']}")
