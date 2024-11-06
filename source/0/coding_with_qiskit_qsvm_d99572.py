# https://github.com/arshpreetsingh/Qiskit-cert/blob/117a7ebcd8b757432a3ec61ef23bf3634e0c1d62/coding_with_qiskit_QSVM.py
"""
We will use Classification to identify/Get Classification between two labels.
We will use QSVM() to predict and understand the Data.
"""
import qiskit
from qiskit.ml.datasets import ad_hoc_data
import numpy as np
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.circuit.library import ZZFeatureMap
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name

feature_dim = 2
training_dataset_size = 20
testing_dataset_size = 10
random_seed = 10598
shots = 10000

sample_Total, training_input, test_input, class_labels = ad_hoc_data(training_size=training_dataset_size,
                                                                     test_size=testing_dataset_size,
                                                                     gap=0.3,
                                                                     n=feature_dim,
                                                                     plot_data=True)

datapoints, class_to_label = split_dataset_to_data_and_labels(test_input)
# print("*******************")
# print(class_to_label)

backend = BasicAer.get_backend('qasm_simulator')
feature_map =ZZFeatureMap(feature_dim, reps=2)
svm = QSVM(feature_map, training_input, test_input)
svm.random_seed = random_seed
quantum_instance = QuantumInstance(backend, shots=shots, seed_simulator=random_seed, seed_transpiler=random_seed)
result = svm.run(quantum_instance)
# print(result)

# print("Kernerl Matrix During the training")
# kernel_matrix = result['kernel_matrix_training']

predicted_labels = svm.predict(datapoints[0])
predicted_classes = map_label_to_class_name(predicted_labels,svm.label_to_class)
print("Actal-Values")
print(datapoints[1])
print("Predicted_Values")
print(predicted_labels)