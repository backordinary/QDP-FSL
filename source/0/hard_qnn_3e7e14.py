# https://github.com/ghuioio/effective_dimension_test/blob/e652138b1544ce8c403d53ad2ecd35ecb557674d/Loss_plots/generate_data/hard_qnn.py
from pickletools import optimize
from time import time
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, ZFeatureMap
from qiskit.quantum_info import Statevector
import numpy as np

from adam import ADAM
from adam.gradient_descent import GradientDescent
from math import log
from warnings import filterwarnings
filterwarnings(action='ignore')
# Train the quantum neural network with feature map = ZZFeatureMap, variational form = RealAmplitudes

# size of training data set
training_size = 100

# dimension of data sets
n = 4

from sklearn import datasets
from sklearn import preprocessing
iris = datasets.load_iris()

# load iris and normalise
x = preprocessing.normalize(iris.data)
x1_train = x[0:49, :] # class A
x2_train = x[50:99, :] # class B
training_input = {'A':x1_train, 'B':x2_train}
class_labels = ['A', 'B']
blocks = 1
sv = Statevector.from_label('0' * n)
circuit = QuantumCircuit(n)
# feature_map = ZZFeatureMap(n, reps=2)
# var_form = RealAmplitudes(n, reps=blocks)
feature_map = ZZFeatureMap(n, reps=2)
var_form = RealAmplitudes(4, entanglement= [[0,1], [2,3], [0,2], [0,3]], reps=1)
circuit = feature_map.combine(var_form)


def get_data_dict(params, x):
    """Get the parameters dict for the circuit"""
    parameters = {}
    for i, p in enumerate(feature_map.ordered_parameters):
        parameters[p] = x[i]
    for i, p in enumerate(var_form.ordered_parameters):
        parameters[p] = params[i]
    return parameters


def assign_label(bit_string, class_labels):
    hamming_weight = sum([np.int32(k) for k in list(bit_string)])
    is_odd_parity = hamming_weight & 1
    if is_odd_parity:
        return class_labels[1]
    else:
        return class_labels[0]


def return_probabilities(counts, class_labels):
    shots = sum(counts.values())
    result = {class_labels[0]: 0,
              class_labels[1]: 0}
    for key, item in counts.items():
        label = assign_label(key, class_labels)
        result[label] += counts[key]/shots
    return result


def classify(x_list, params, class_labels):
    qc_list = []
    for x in x_list:
        circ_ = circuit.assign_parameters(get_data_dict(params, x))
        qc = sv.evolve(circ_)
        qc_list += [qc]
    probs = []
    for qc in qc_list:
        counts = qc.to_counts()
        prob = return_probabilities(counts, class_labels)
        probs += [prob]
    return probs


def CrossEntropy(yHat, y):
    if y == 'A':
      return -log(yHat['A'])
    else:
      return -log(1-yHat['A'])


def cost_function(training_input, class_labels, params, shots=100, print_value=False):
    # map training input to list of labels and list of samples
    cost = 0
    training_labels = []
    training_samples = []
    for label, samples in training_input.items():
        for sample in samples:
            training_labels += [label]
            training_samples += [sample]
    # classify all samples
    probs = classify(training_samples, params, class_labels)

    # evaluate costs for all classified samples
    for i, prob in enumerate(probs):
        # cost += cost_estimate_sigmoid(prob, training_labels[i])
        cost += CrossEntropy(yHat=prob, y=training_labels[i])
    cost /= len(training_samples)
    # return objective value
    return cost

# setup the optimizer
#optimizer = ADAM(maxiter=100, lr=0.1)
optimizer = GradientDescent(maxiter=100, learning_rate=0.1)
# define objective function for training
objective_function = lambda params: cost_function(training_input, class_labels, params, print_value=True)

for i in range(100):
    np.random.seed(i)
    d = 8  # num of trainable params
    # randomly initialize the parameters
    init_params = 2 * np.pi * np.random.rand(n * (1) * 2)
    # train classifier
    init_params = 2 * np.pi * np.random.rand(n * (1) * 2)
    # opt_params, value, _, loss = optimizer.optimize(len(init_params), objective_function, initial_point=init_params)
    loss = optimizer.optimize(len(init_params), objective_function, initial_point=init_params)
    print("success, round = %d" %i)
    # print results
    f1 = 'quantum_loss_hard_full_grades_%d.npy' %i
    np.save(f1, loss)
    # f2 = 'opt_params_hard_dep2_%d.npy'%i
    # np.save(f2, opt_params)
   