# https://github.com/keamanansiber/qiskit/blob/6f1459aa130100251201ea8f0a7d78f97134919e/3QuantumMachineLearning/Tutorial/QSVM%204%20-%20Quantum%20SVM%20algorithm%20multiclass%20classifier%20extension.py
import numpy as np
from qiskit.ml.datasets import wine
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.components.multiclass_extensions import AllPairs
from qiskit.aqua.utils.dataset_helper import get_feature_dimension


n = 2  # dimension of each data point
sample_Total, training_input, test_input, class_labels = wine(
    training_size=40,
    test_size=10, n=n, plot_data=True
)
temp = [test_input[k] for k in test_input]
total_array = np.concatenate(temp)


aqua_globals.random_seed = 10598

backend = BasicAer.get_backend('qasm_simulator')
feature_map = SecondOrderExpansion(feature_dimension=get_feature_dimension(training_input),
                                   depth=2, entangler_map=[[0, 1]])
svm = QSVM(feature_map, training_input, test_input, total_array,
          multiclass_extension=AllPairs())
quantum_instance = QuantumInstance(backend, shots=1024,
                                    seed_simulator=aqua_globals.random_seed,
                                    seed_transpiler=aqua_globals.random_seed)

result = svm.run(quantum_instance)
for k,v in result.items():
    print("'{}' : {}".format(k, v))


# 'testing_accuracy' : 1.0

# sumber = https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/machine_learning/qsvm_multiclass.ipynb
