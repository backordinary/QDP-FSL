# https://github.com/martin00001313/Quantum-machine-learning/blob/881447374e009c3e33c59cc4cf8e5eda061ea0b7/QK-mean/kmeans.py
import copy
import pandas as pd
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit
from qiskit import Aer, execute
from numpy import pi
import random


class QKM:
    def __init__(self, K, input_path, feature_list,  output_path, max_iter_count=10):
        if feature_list is None or len(feature_list) != 2:
            self.K = None
            self.feature_list = None
            self.input_path = None
            self.output_path = None
            self.max_iter_count = None
        else:
            self.K = K
            self.input_path = input_path
            self.feature_list = feature_list
            self.output_path = output_path
            self.max_iter_count = max_iter_count
        self.encoded_centroid_features = None

    def train(self):
        base_data = self._get_input_data()
        train_data = [[ft1, ft2] for ft1, ft2 in zip(base_data[self.feature_list[0]], base_data[self.feature_list[1]])]

        if len(train_data) < self.K:
            raise Exception("The classes count should be more than the count of classes need to be predicted!")

        # get random centroids
        tmp_centroids = random.sample(train_data, self.K)

        encoded_centroid_features = self._get_phi_theta_for_clusters(tmp_centroids)
        classification, base_class_idxs = self._classify_instances(train_data, encoded_centroid_features)
        print(base_class_idxs)

        iter_count = 0
        while True:
            centroids = self._calculate_centroids(classification)
            print("MA13: start - " + str(iter_count) + ",  " + str(centroids))
            encoded_centroid_features = self._get_phi_theta_for_clusters(centroids)
            classification, class_idxs = self._classify_instances(train_data, encoded_centroid_features)
            iter_count += 1
            if all(x == y for x, y in zip(base_class_idxs, class_idxs)):
                print("The training process has been finished successfully!")
                break
            elif iter_count > self.max_iter_count:
                print("The training was interrupted based on the provided restrictions related to iterations count!")
                break

        # keep centroids data for future predictions
        self.encoded_centroid_features = encoded_centroid_features

        print("Finish: " + str(classification))
        print("Finish2: " + str(class_idxs))
        return classification, class_idxs

    def _classify_instances(self, samples, encoded_centroid_features):
        clusters = [[] for i in range(self.K)]
        class_idxs = []

        for sample in samples:
            class_idx = self._classify_sample(sample, encoded_centroid_features)
            class_idxs.append(class_idx)
            clusters[class_idx].append(sample)
        return clusters, class_idxs

    def _classify_sample(self, sample, encoded_centroid_features):
        sample_pi = (sample[0] + 1) * pi / 2
        sample_theta = (sample[1] + 1) * pi / 2
        distance_from_centroids = self._qdistance_calculation(sample_pi, sample_theta, encoded_centroid_features)
        print(str(distance_from_centroids))
        # get the element having minimum distance
        samples_class_idx = distance_from_centroids.index(min(distance_from_centroids))
        return samples_class_idx

    @staticmethod
    def _get_phi_theta_for_clusters(centroids):
        return [[(ft1 + 1) * pi / 2, (ft2 + 1) * pi / 2] for ft1, ft2 in centroids]

    @staticmethod
    def _calculate_centroids(clusters):
        if any(len(i) == 0 for i in clusters):
            # case when there is a tmp class having no entities
            return None

        return [[sum([ft[0] for ft in cluster]) / len(cluster), sum([ft[1] for ft in cluster]) / len(cluster)] for cluster in clusters]

    @staticmethod
    def _qdistance_calculation(input_pi_val, input_theta_val, encoded_centroid_features):
        # Create a 3 qubit QuantumRegister - two for the vectors, and
        # one for the ancillary qubit
        qreg = QuantumRegister(3, 'qreg')

        # Create a one bit ClassicalRegister to hold the result
        # of the measurements
        creg = ClassicalRegister(1, 'creg')

        qc = QuantumCircuit(qreg, creg, name='qc')

        # Get backend using the Aer provider
        backend = Aer.get_backend('qasm_simulator')

        # Create list to hold the results
        results_list = []

        for centroid in encoded_centroid_features:
            # Apply a Hadamard to the ancillary
            qc.h(qreg[2])

            # Encode new point and centroid
            qc.u3(input_theta_val, input_pi_val, 0, qreg[0])
            qc.u3(centroid[1], centroid[0], 0, qreg[1])

            # Perform controlled swap
            qc.cswap(qreg[2], qreg[0], qreg[1])
            # Apply second Hadamard to ancillary
            qc.h(qreg[2])

            qc.draw()
            # Measure ancillary
            qc.measure(qreg[2], creg[0])

            # Reset qubits
            qc.reset(qreg)

            # Register and execute job
            job = execute(qc, backend=backend, shots=10224)
            q_result = job.result().get_counts(qc)
            print(str(q_result))
            results_list.append(q_result.get('1', 0))

        return results_list

    def predict(self, sample):
        clusters, class_idxs = self._classify_instances([sample], self.encoded_centroid_features)
        return class_idxs

    def _get_input_data(self):
        if self.input_path is None:
            return None
        input_data = pd.read_csv(self.input_path, usecols=self.feature_list)
        return input_data
