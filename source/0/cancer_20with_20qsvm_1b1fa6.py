# https://github.com/alielfilali01/Quantum-Machine-Learning/blob/776f11a307f355c536dfbb312cc0cd59e7f66884/cancer%20with%20QSVM.py
### BREAST CANCER CLASSIFIER WITH QSVM


from datasets import *
from qiskit import Aer
from qiskit import BasicAer
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.aqua.input import ClassificationInput
from qiskit.aqua import run_algorithm, QuantumInstance
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.components.feature_maps import SecondOrderExpansion

# setup aqua logging
import logging
from qiskit.aqua import set_qiskit_aqua_logging
# set_qiskit_aqua_logging(logging.DEBUG)  # choose INFO, DEBUG to see the log
import numpy as np

n = 2  # dimension of each data point
sample_Total, training_input, test_input, class_labels = Breast_cancer(training_size=40,
                                                              test_size=10, n=n, PLOT_DATA=True)

temp = [test_input[k] for k in test_input]
total_array = np.concatenate(temp)

aqua_dict = {
    'problem': {'name': 'classification', 'random_seed': 100},
    'algorithm': {
        'name': 'QSVM'
    },
    'backend': {'provider': 'qiskit.BasicAer', 'name': 'qasm_simulator', 'shots': 256},
    'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2, 'entanglement': 'linear'}
} 

algo_input = ClassificationInput(training_input, test_input, total_array)
result = run_algorithm(aqua_dict, algo_input)

for k,v in result.items():
    print("'{}' : {}".format(k, v))