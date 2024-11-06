# https://github.com/QuantumApplicationLab/qradio/blob/f284c21fdb39fd2422986fc61a9c69e229344c4f/qradio/pulsar_search/machine_learning/knn/knn_quantum.py
from numpy.random.mtrand import shuffle
from qiskit_quantum_knn.qknn import QKNeighborsClassifier
from qiskit_quantum_knn.encoding import analog
from qiskit import aqua
from sklearn import datasets
import qiskit as qk


import numpy as np
from dataclasses  import dataclass
from knn_data import read_dataset, divide_dataset




class QKNN:

    def __init__(self, n_neighbor=3):
        """Handles the q-knn training/testing

        Reference :

        Quantum k-nearest neighbors algorithm
        https://arxiv.org/abs/2003.09187
        https://github.com/GroenteLepel/qiskit-quantum-knn
        https://www.ru.nl/publish/pages/913395/daniel_kok_4_maart_2021.pdf

        Args:
            train_dataset (dataclass): train dataset
            test_dataset (dataclass): test dataset
            n_neighbor (int, optional): number of neighbors. Defaults to 3.
        """

        self.n_neighbor = n_neighbor
        self.model = self.initialize_quantum_circuit(n_neighbor)

    def initialize_quantum_circuit(self, n_neighbor):
        """Init the quantum circuit

        Args:
            n_neighbor (int, optional): number of neighbors
        """


        # initialising the quantum instance
        backend = qk.BasicAer.get_backend('qasm_simulator')
        instance = aqua.QuantumInstance(backend, shots=1000)

        # initialising the qknn model
        return QKNeighborsClassifier(n_neighbors=n_neighbor,
                                     quantum_instance=instance)


    def get_circuits(self):
        """Returns  the circuits for the knn

        Returns:
            list: list of qiskit circuits
        """
        return self.model.instance._qalgo.construct_circuits(self.test_data, self.model.instance._qalgo.training_dataset)

    def get_circuit_results(self, circuits):
        return self.model.instance._qalgo.get_circuit_results(circuits)

    def get_contrasts(self, circuits):
        circuit_results = self.get_circuit_results(circuits)
        return self.model.instance._qalgo.get_all_contrasts(circuit_results)

    def encode_data(self, train_dataset, test_dataset):
        """Encode the data in the circuit

        Args:
            dataset ([type]): [description]
            fraction (list, optional): [description]. Defaults to [0.8,0.2].
        """

        # select the features
        train_dataset.features = train_dataset.features
        test_dataset.features = test_dataset.features

        # encode data
        encoded_train_data = analog.encode(train_dataset.features)
        encoded_test_data = analog.encode(test_dataset.features)

        # now pick these indices from the data
        self.train_data = encoded_train_data
        self.train_labels = train_dataset.labels

        self.test_data = encoded_test_data
        self.test_labels = test_dataset.labels

    def fit(self):
        self.model.fit(self.train_data, self.train_labels)

    def test(self):
        predict = self.model.predict(self.test_data)
        percent = 100*np.sum(predict == self.test_labels)/len(self.test_labels)
        print(" ==> Classification succesfull at %f percent" %percent)
        return predict

if __name__ == "__main__":
    dataset = read_dataset(shuffle=True)

    qknn = QKNN()
    qknn.encode_data(dataset)
    qknn.fit()
    prediction = qknn.test()
    print(qknn.test_labels)
    print(prediction)