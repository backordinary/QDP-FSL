# https://github.com/Inverseit/quantum/blob/87bc8b3f6c74f202037b0cad08a79d52b81df955/knn/test.py
from qiskit_quantum_knn.qknn import QKNeighborsClassifier
from qiskit_quantum_knn.encoding import analog
from qiskit import terra
from sklearn import datasets
import qiskit as qk

# initialising the quantum instance
backend = qk.BasicAer.get_backend('qasm_simulator')
instance = terra.QuantumInstance(backend, shots=10000)

# initialising the qknn model
qknn = QKNeighborsClassifier(
   n_neighbors=3,
   quantum_instance=instance
)

n_variables = 2        # should be positive power of 2
n_train_points = 4     # can be any positive integer
n_test_points = 2      # can be any positive integer

# use iris dataset
iris = datasets.load_iris()
labels = iris.target
data_raw = iris.data

# encode data
encoded_data = analog.encode(data_raw[:, :n_variables])

# now pick these indices from the data
train_data = encoded_data[:n_train_points]
train_labels = labels[:n_train_points]

test_data = encoded_data[n_train_points:(n_train_points+n_test_points), :n_variables]
test_labels = labels[n_train_points:(n_train_points+n_test_points)]

qknn.fit(train_data, train_labels)
qknn_prediction = qknn.predict(test_data)

print(qknn_prediction)
print(test_labels)