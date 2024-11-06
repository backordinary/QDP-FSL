# https://github.com/Ike-adebi/Personal-Portfolio/blob/d69923d62fa349fb40135a63628dd6819520703e/Quantum%20Machine%20Learning%20Model/my_quantum_ML_machine.py
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.algorithms import QSVM
from qiskit import IBMQ
from sklearn import model_selection

import numpy as np
import pandas as pd

IBMQ.load_accounts()

survival_data = pd.read_csv("survival_data.data")
survival_data.drop(['Year'], axis=1, inplace=True)

predict_label = "Survival Status"

X = np.array(survival_data.drop([predict_label], 1))
y = np.array(survival_data[predict_label])
# print("X", X)
# print("y", y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
# X_train = X[:30]
# X_test = X[30:45]
# y_train = y[:30]
# y_test = y[30:45]
# X_test2 = X[45:65]
# y_test2 = y[45:65]

print("X Train:", len(X_train))
print("X Test:", len(X_test))
print("Y train:", len(y_train))
print("Y test:", len(y_test))

training_data = {1:[], 2:[]}
testing_data = {1:[],2:[]}
for x,y in zip(X_train, y_train):
    training_data[y].append(x)
for x,y in zip(X_test, y_test):
    testing_data[y].append(x)

for label in training_data:
    training_data[label] = np.array(training_data[label])

for label in testing_data:
    testing_data[label] = np.array(testing_data[label])

backend = BasicAer.get_backend("qasm_simulator")
feature_map = SecondOrderExpansion(feature_dimension=2, depth=2, entangler_map=[[0,1]])
svm = QSVM(feature_map, training_data, testing_data, None)

random_seed = 10598
svm.random_seed = random_seed
quantum_instance = QuantumInstance(backend,shots=1024,seed=random_seed,seed_transpiler=random_seed)
result = svm.run(quantum_instance)

print(result)
print("testing success ratio: ", result['testing_accuracy'])
# predicted_labels = svm.predict(X_test2)
#
# predicted_classes = map_label_to_class_name(predicted_labels, {0:1, 1:2})
#
# print(f"ground truth: {y_test2}")
# print(f"prediction: {np.array(predicted_classes)}")
# acc = 0
# count = 0
# for a, b in zip(y_test2, predicted_classes):
#     if a == b:
#         acc += 1
#     count += 1
# acc /= count
# print(acc)