# https://github.com/alanmark13579/qiskit/blob/7a31b1208072ae1e544eb3b0c6b144a3041fc63a/qiskit_breast_cancer.py
import sys
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit_machine_learning.datasets.dataset_helper import features_and_labels_transform
from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.aqua.algorithms import SklearnSVM
import pandas as pd

seed = 10599
aqua_globals.random_seed = seed


def breast_cancer(training_size, test_size, n, plot_data=False, one_hot=True):
    """returns breast cancer dataset"""

    df = pd.read_csv("./breast_cancer_source.csv", header=0)
    array_temp= []
    array1_temp= []
    str_temp= []
    columns = df.shape[1]-1
    data = []
    target = []
    for i in range(0,len(df)):
      array_temp.append(df.iloc[i])
      for j in range(0,columns):
        array1_temp.append(array_temp[0][j])
      data.append(np.array(array1_temp))

      target.append(np.int(array_temp[0][8]))
      array_temp = []
      array1_temp = []
    data = np.array(data)
    target = np.array(target)

   

    class_labels = [r"A", r"B"]
    #data, target = datasets.load_breast_cancer(return_X_y=True)
    sample_train, sample_test, label_train, label_test = train_test_split(
        data, target, test_size=0.3, random_state=12
    )

    # Now we standardize for gaussian around 0 with unit variance
    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)

    # Now reduce number of features to number of qubits
    pca = PCA(n_components=n).fit(sample_train)
    sample_train = pca.transform(sample_train)
    sample_test = pca.transform(sample_test)

    # Scale to the range (-1,+1)
    samples = np.append(sample_train, sample_test, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    sample_train = minmax_scale.transform(sample_train)
    sample_test = minmax_scale.transform(sample_test)

    # Pick training size number of samples from each distro
    training_input = {
        key: (sample_train[label_train == k, :])[:training_size]
        for k, key in enumerate(class_labels)
    }
    test_input = {
        key: (sample_test[label_test == k, :])[:test_size] for k, key in enumerate(class_labels)
    }

    
    training_feature_array, training_label_array = features_and_labels_transform(
        training_input, class_labels, one_hot
    )
    test_feature_array, test_label_array = features_and_labels_transform(
        test_input, class_labels, one_hot
    )

    if not plot_data:
        try:
            import matplotlib.pyplot as plt
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                libname="Matplotlib",
                name="breast_cancer",
                pip_install="pip install matplotlib",
            ) from ex
        plt.scatter(
            sample_train[label_train == 0, 0][:training_size],
            sample_train[label_train == 0, 1][:training_size],
        )
        plt.scatter(
            sample_train[label_train == 1, 0][:training_size],
            sample_train[label_train == 1, 1][:training_size],
        )

        plt.title("PCA dim. reduced Breast cancer dataset")
        plt.show()
        print(f"維度:{n}維")
        feature_dim = n
        feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2, entanglement='linear')
        print(feature_map)

        qsvm = QSVM(feature_map, training_input, test_input)

        backend = BasicAer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed)

        result = qsvm.run(quantum_instance)
        print("QSVM")
        print(f'Testing success ratio: {result["testing_accuracy"]}')
        feature_dim = n
        feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=3, entanglement='linear')
        print(feature_map)

        qsvm = QSVM(feature_map, training_input, test_input)

        backend = BasicAer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed)

        result = qsvm.run(quantum_instance)
        print("QSVM")
        print(f'Testing success ratio: {result["testing_accuracy"]}')
        feature_dim = n
        feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=4, entanglement='linear')
        print(feature_map)

        qsvm = QSVM(feature_map, training_input, test_input)

        backend = BasicAer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed)

        result = qsvm.run(quantum_instance)
        print("QSVM")
        print(f'Testing success ratio: {result["testing_accuracy"]}')
        feature_dim = n
        feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=5, entanglement='linear')
        print(feature_map)

        qsvm = QSVM(feature_map, training_input, test_input)

        backend = BasicAer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed)

        result = qsvm.run(quantum_instance)
        print("QSVM")
        print(f'Testing success ratio: {result["testing_accuracy"]}')
        feature_dim = n
        feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=6, entanglement='linear')
        print(feature_map)

        qsvm = QSVM(feature_map, training_input, test_input)

        backend = BasicAer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed)

        result = qsvm.run(quantum_instance)
        print("QSVM")
        print(f'Testing success ratio: {result["testing_accuracy"]}')
        feature_dim = n
        feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=7, entanglement='linear')
        print(feature_map)

        qsvm = QSVM(feature_map, training_input, test_input)

        backend = BasicAer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed)

        result = qsvm.run(quantum_instance)
        print("QSVM")
        print(f'Testing success ratio: {result["testing_accuracy"]}')
        kernel_matrix = result['kernel_matrix_training']
        #plt.imshow(np.asmatrix(kernel_matrix),interpolation='nearest',origin='upper',cmap='bone_r')


        #Classical approach

        result = SklearnSVM(training_input, test_input).run()
        print("Classical approach")
        print(f'Testing success ratio: {result["testing_accuracy"]}')

        kernel_matrix = result['kernel_matrix_training']

        #plt.imshow(np.asmatrix(kernel_matrix), interpolation='nearest', origin='upper', cmap='bone_r');


        return (
        
        training_feature_array,
        training_label_array,
        test_feature_array,
        test_label_array,
        
        )
breast_cancer(52,26,2)
breast_cancer(52,26,3)
breast_cancer(52,26,4)
breast_cancer(52,26,5)
breast_cancer(52,26,6)
breast_cancer(52,26,7)