# https://github.com/HectorMenezes/UndergraduateProject/blob/ccc0ba201fa94db2211cfcac45119660a2928218/src/SVMs/qiskit_svm.py
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.components.multiclass_extensions import MulticlassExtension, AllPairs
from qiskit.circuit.library import ZZFeatureMap
from qiskit.ml.datasets import breast_cancer
import pandas as pd
import numpy as np

from definitions import SEED, ROOT_DIR

# TODO: change data to the banking data
_, training_input, test_input, _ = breast_cancer(
    training_size=20,
    test_size=10,
    n=9,
)


data = pd.read_csv(ROOT_DIR+ '/data/phishing_websites.txt', header=None)
dict_data = {'0': data[data[9] == 0].iloc[:, :-1].to_numpy(),
             '-1': data[data[9] == -1].iloc[:, :-1].to_numpy(),
             '1': data[data[9] == 1].iloc[:, :-1].to_numpy()}

feature_map = ZZFeatureMap(feature_dimension=9, reps=2, entanglement='linear')
qsvm = QSVM(feature_map, dict_data, dict_data, multiclass_extension=AllPairs())

backend = BasicAer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=SEED, seed_transpiler=SEED)

result = qsvm.run(quantum_instance)

print(f'Testing success ratio: {result["testing_accuracy"]}')
