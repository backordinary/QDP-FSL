# https://github.com/Lucacontr/MalwareDetection/blob/2c408e7330a6f626bfccf517e13308fae5af7c71/CircuitQNN.py
import time
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier


def callback_graph(weights, obj_func_eval, objective_func_vals=[]):
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

data = pd.read_csv("dataset/Small&Balanced.csv")
data = data.drop('hash', axis=1)
features = data.drop(columns='malware').values
labels = data['malware'].values

train_features, test_features, train_labels, test_labels = train_test_split(
            features, data['malware'].values, test_size=0.20, shuffle=True)

#Feature Selection
print("Feature Selection..")
pca = SelectKBest(score_func=chi2, k=250)
train_features = pca.fit_transform(train_features, train_labels)
test_features = pca.fit_transform(test_features, test_labels)

#Feature Extraction
print("Feature Extraction..")
pca = PCA(10)
pca.fit(train_features)
train_features_pca = pca.transform(train_features)
test_features_pca = pca.transform(test_features)

plt.scatter(train_features_pca[:, 0], train_features_pca[:, 1], c=train_labels, cmap=plt.cm.prism, edgecolor='k', alpha=0.7)
plt.show()

#Data Balancing
print("Data Balancing..")
sm = SMOTE()
print("TRAIN FEATURES: \n", train_features_pca)
train_features_bal, train_labels_bal = sm.fit_resample(train_features_pca, train_labels)

num_qubits = 10

quantum_instance = QuantumInstance(Aer.get_backend("aer_simulator"))

# construct feature map
feature_map = ZZFeatureMap(num_qubits)

# construct ansatz
ansatz = RealAmplitudes(num_qubits, reps=1)

# construct quantum circuit
qc = QuantumCircuit(num_qubits)
qc.append(feature_map, range(num_qubits))
qc.append(ansatz, range(num_qubits))
qc.decompose().draw(output="mpl")

# parity maps bitstrings to 0 or 1
def parity(x):
    return "{:b}".format(x).count("1") % 2

output_shape = 2  # corresponds to the number of classes, possible outcomes of the (parity) mapping.
# construct QNN
circuit_qnn = CircuitQNN(
    circuit=qc,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    interpret=parity,
    output_shape=output_shape,
    quantum_instance=quantum_instance,
)
# construct classifier
circuit_classifier = NeuralNetworkClassifier(
    neural_network=circuit_qnn, optimizer=COBYLA(maxiter=50), callback=callback_graph
)
# create empty array for callback to store evaluations of the objective function
objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)

print("Running..")
start_time = time.time()
# fit classifier to data
result = circuit_classifier.fit(train_features_pca, train_labels)
print(result)
total_time = time.time() - start_time
print("Train effettuato in " + str(total_time))

# return to default figsize
plt.rcParams["figure.figsize"] = (6, 4)

# score classifier
start_time = time.time()
y_predict = circuit_classifier.predict(test_features_pca)
cm = confusion_matrix(test_labels, y_predict)
print("Test effettuato in " + str(total_time))
print(cm)
print(classification_report(test_labels, y_predict))
total_time = time.time() - start_time


for x, y_target, y_p in zip(test_features_pca, test_labels, y_predict):
    if y_target == 1:
        plt.plot(x[0], x[1], "bo")
    else:
        plt.plot(x[0], x[1], "go")
    if y_target != y_p:
        plt.scatter(x[0], x[1], s=200, facecolors="none", edgecolors="r", linewidths=2)
plt.show()



