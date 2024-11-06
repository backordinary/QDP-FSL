# https://github.com/Lucacontr/MalwareDetection/blob/2c408e7330a6f626bfccf517e13308fae5af7c71/PegasosQSVC.py
import time
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, KFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from qiskit.circuit.library import ZFeatureMap
from qiskit import IBMQ
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms import PegasosQSVC
from qiskit import Aer
from qiskit.utils import QuantumInstance, algorithm_globals
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import precision_score

data = pd.read_csv("dataset/Small&Balanced.csv")
data = data.drop('hash', axis=1)
features = data.drop(columns='malware').values
labels = data['malware'].values

train_features, test_features, train_labels, test_labels = train_test_split(
            features, data['malware'].values, test_size=0.20, shuffle=True)


print("Feature Selection..")
#Feature Selection
kbest = SelectKBest(score_func=chi2, k=750)
train_features = kbest.fit_transform(train_features, train_labels)
test_features = kbest.transform(test_features)

# Feature Extraction
print("Feature Extraction..")
pca = PCA(32)
pca.fit(train_features)
train_features_pca = pca.transform(train_features)
test_features_pca = pca.transform(test_features)

# Data Balancing
sm = SMOTE()
train_features_bal, train_labels_bal = sm.fit_resample(train_features_pca, train_labels)

num_qubits = 32

#Classification PegasosQSVC
tau = 100
C = 1000
algorithm_globals.random_seed = 12345
backend = QuantumInstance(Aer.get_backend("aer_simulator"))
feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=1)
qkernel = QuantumKernel(feature_map=feature_map, quantum_instance=backend)
qsvc = PegasosQSVC(quantum_kernel=qkernel, C=C, num_steps=tau)

# training
print("Running...")
start_time = time.time()
qsvc.fit(train_features_bal, train_labels_bal)
total_time = time.time() - start_time
print("Train effettuato in " + str(total_time))

# testing
start_time = time.time()
y_predict = qsvc.predict(test_features_pca)
cm = confusion_matrix(test_labels, y_predict)
print(cm)
print(classification_report(test_labels, y_predict))
total_time = time.time() - start_time
print("Test effettuato in " + str(total_time))


