# https://github.com/Ike-adebi/Personal-Portfolio/blob/d69923d62fa349fb40135a63628dd6819520703e/Quantum%20Machine%20Learning%20Model/Practice/qsvm_direct.py
from Practice.datasets import *
from qiskit import BasicAer
from qiskit.aqua import run_algorithm, QuantumInstance
from qiskit.aqua.input import ClassificationInput
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit import IBMQ

IBMQ.load_accounts()

feature_dim = 2  # dimension of each data point
training_dataset_size = 20
testing_dataset_size = 10
random_seed = 10598
shots = 1024

run = 1
sample_Total, training_input, test_input, class_labels = ad_hoc_data(
    training_size=training_dataset_size,
    test_size=testing_dataset_size,
    n=feature_dim,
    gap=0.3,
    PLOT_DATA=False
)

datapoints, class_to_label = split_dataset_to_data_and_labels(test_input)
print(test_input)
print(type(test_input['A']))
# print(class_to_label)
# pprint.pprint(training_input)
# pprint.pprint(test_input)

if run == 0:
    print("DECLARATIVE APPROACH")
    '''
    APPROACH I: DECLARATIVE APPROACH

    Now we can create the svm in the declarative approach. In the following json, we config:
    - the algorithm name
    - the feature map
    '''

    params = {
        'problem': {'name': 'classification', 'random_seed': random_seed},
        'algorithm': {
            'name': 'QSVM'
        },
        'backend': {'shots': shots},
        'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2, 'entanglement':'linear'}
    }
    backend = BasicAer.get_backend('qasm_simulator')

    algo_input = ClassificationInput(training_input, test_input, datapoints[0])

    # Now everything is setup, so we can run the algorithm
    '''
    Run method includes training, testing, and predicting on unlabeled data
    For the testing, the result includes the success ratio.
    For the prediction, the result includes the predicted class names for each data.
    After that, the trained model is also stored in the svm instance,
        and you can use it for future predictions.
    '''
    result = run_algorithm(params, algo_input, backend=backend)
    print("kernel matrix during the training:")
    kernel_matrix=result['kernel_matrix_training']
    img = plt.imshow(np.asmatrix(kernel_matrix), interpolation="nearest", origin='upper', cmap="bone_r")
    plt.show()

    print("testing success ratio: ", result['testing_accuracy'])
    print("predicted classes:", result['predicted_classes'])

else:
    print("PROGRAMMATIC APPROACH")
    '''
    APPROACH II: PROGRAMMATIC APPROACH
    
    We construct the svm instance directly from the classes.
    The programmatic approach offers the users 
    better accessibility, e.g., the users can 
    access the internal state of svm instance 
    or invoke the methods of the instance. 
    We will demonstrate this advantage soon.
    '''

    # Now we create the svm in the programmatic approach
    '''
    We build the svm by instantianting the class QSVM
    We build hte feature map instance (required by the
    svm instance) by instantiating the class
    SecondOrder Expansion.
    '''
    backend = BasicAer.get_backend('qasm_simulator')
    feature_map = SecondOrderExpansion(feature_dimension=feature_dim,
                                       depth=2,
                                       entangler_map=[[0,1]])
    svm = QSVM(feature_map, training_input, test_input, None) # the data for prediction can be fed later
    svm.random_seed = random_seed
    quantum_instance = QuantumInstance(backend, shots=shots, seed=random_seed, seed_transpiler=random_seed)
    result = svm.run(quantum_instance)

    print("kernel matrix during the training:")
    kernel_matrix = result['kernel_matrix_training']
    img = plt.imshow(np.asmatrix(kernel_matrix), interpolation="nearest",
                     origin="upper", cmap="bone_r")
    plt.show()
    print("testing success ratio: ", result['testing_accuracy'])
    predicted_labels = svm.predict(datapoints[0])
    predicted_classes = map_label_to_class_name(predicted_labels, svm.label_to_class)
    print(f"ground truth: {datapoints[1]}")
    print(f"prediction: {predicted_labels}")