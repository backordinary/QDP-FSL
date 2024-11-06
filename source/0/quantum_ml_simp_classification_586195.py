# https://github.com/cgrant093/Coding-with-Qiskit/blob/ed2e1b859a6412b419046724cfdaf9e7226d4aea/Quantum_ML_simp_classification.py
# quantum machine learning: classification problem

import numpy as np
import matplotlib.pyplot as plt

from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.ml.datasets import ad_hoc_data, sample_ad_hoc_data

# parameters
feature_dim = 2
training_dataset_size = 20
testing_dataset_size = 10
shots = 10000
random_seed = 10598

# setup training data
sample_total, training_input, test_input, class_labels,  = ad_hoc_data(
    training_size=training_dataset_size,
    test_size=testing_dataset_size,
    n=feature_dim,
    gap=0.3,
    plot_data=False )
    
extra_test_data = sample_ad_hoc_data(sample_total, 10, n=feature_dim)
datapoints, class_to_label = split_dataset_to_data_and_labels(extra_test_data)
print(class_to_label)

# setup backend, feature map, and plugged into quantum support vector machine
backend = BasicAer.get_backend('qasm_simulator')
feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2, entanglement='linear')
svm = QSVM(feature_map, training_input, test_input, datapoints[0])
svm.random_seed = random_seed
quantum_instance = QuantumInstance(backend, shots=shots, seed_simulator=random_seed, seed_transpiler=random_seed)
result = svm.run(quantum_instance)

print(f'Testing success ratio: {result["testing_accuracy"]}')
print()
print('Prediction from datapoints set:')
print(f'  ground truth: {map_label_to_class_name(datapoints[1], qsvm.label_to_class)}')
print(f'  prediction:   {result["predicted_classes"]}')
predicted_labels = result["predicted_labels"]
print(f'  success rate: {100*np.count_nonzero(predicted_labels == datapoints[1])/len(predicted_labels)}%')


print("kernel matrix during the trainings:")
kernel_matrix = result('kernel_matrix_training')
img = plt.imshow(np.asmatrix(kernel_matrix), interpolation='nearest', origin='upper', cmap='bone_r')
plt.show()