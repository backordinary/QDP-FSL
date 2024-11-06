# https://github.com/TheAldex/QSVM/blob/36a91b1206ef0c808877e8ad26aaa4f4b33f202b/Real-Dataset-Example/qsvm-cleveland.py
import numpy as np
import pandas as pd

#sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

#qiskit
from qiskit import BasicAer
from qiskit.circuit.library import ZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('Cleveland Dataset.csv')

X = data[['ca','cp','thal','exang','slope']]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

samples = np.append(X_train, X_test, axis=0)
minmax_scaler = MinMaxScaler((0, 1)).fit(samples)
X_train = minmax_scaler.transform(X_train)
X_test = minmax_scaler.transform(X_test)

# number of qubits is equal to the number of features
num_qubits = 5
# regularization parameter
C = 1000

algorithm_globals.random_seed = 12345

backend = QuantumInstance(
    BasicAer.get_backend("qasm_simulator"),
    seed_simulator=algorithm_globals.random_seed,
    seed_transpiler=algorithm_globals.random_seed,
)
feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=2)
qkernel = QuantumKernel(feature_map=feature_map, quantum_instance=backend)
qsvc = QSVC(quantum_kernel=qkernel, C=C)

# training
qsvc.fit(X_train, y_train)

# testing
qsvc_score = qsvc.score(X_test, y_test)
print(f"QSVC classification test score: {qsvc_score}")

# classification report of QSVC
expected_y  = y_test
predicted_y = qsvc.predict(X_test) 

# print classification report and confusion matrix for the classifier
print("Classification report: \n", metrics.classification_report(expected_y, predicted_y))
print("Confusion matrix: \n", metrics.confusion_matrix(expected_y, predicted_y))

