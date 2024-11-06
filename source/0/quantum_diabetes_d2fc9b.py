# https://github.com/Rxtreem65/Quantum_diabetes_prediction/blob/d368dbec8981218229f1d52026fa793603c6efd4/Quantum_diabetes.py
import qiskit
import numpy as np
from matplotlib import pyplot as plt
from qiskit.ml.datasets import ad_hoc_data
from qiskit.aqua import QuantumInstance
from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.components.multiclass_extensions import (ErrorCorrectingCode,AllPairs,OneAgainstRest)
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
df= pd.read_csv("diabetes.csv")

feature_dim = 9
y = df["Outcome"]
x_dia = df[df["Outcome"]==1]
x_no_dia = df[df["Outcome"]==0]

x_dia_train, x_dia_test = train_test_split(x_dia)
x_no_dia_train, x_no_dia_test = train_test_split(x_no_dia)

x_7 = {'1':np.asanyarray(x_dia_train[:7]),
           '0':np.asanyarray(x_no_dia_train)[:7]}
x_3 = {'1':np.asanyarray(x_no_dia_train[7:10]),
          '0':np.asanyarray(x_no_dia_test[7:10])}

num_qubits=9
shots = 5

feature_map = ZZFeatureMap(feature_dimension= num_qubits, reps = 2, entanglement = 'full')
svm = QSVM(feature_map,x_7,x_3)
quantum_instance=QuantumInstance(backend = device, shots=shots, skip_qobj_validation=False)


#Run the QSVM for accuracy results
result = svm.run(quantum_instance)

print('Prediction of diabetes disease\n')
print('Accuracy: ' , result['testing_accuracy'],'\n')
