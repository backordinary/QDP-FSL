# https://github.com/mdsinar/TeamMagic8/blob/e81564052b3c271c881d7e4c8aceacfdec0629c7/titanic_zz.py
from filecmp import clear_cache
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import  preprocessing
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

import qiskit
from qiskit.algorithms.optimizers import SPSA, COBYLA, GradientDescent
from qiskit.circuit.library import TwoLocal, ZZFeatureMap, ZFeatureMap
from qiskit.opflow import StateFn, Gradient, I, Z
from qiskit.utils import QuantumInstance

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import CircuitQNN, OpflowQNN
from qiskit_machine_learning.utils.loss_functions import L2Loss, CrossEntropyLoss


# * need to laod the training data
df = pd.read_csv('titanic_dataset/train.csv') # read csv file as pandas data frame
df.head(8)

# * let's see, what features are available in the dataset
feature_name_list = list(df)
list(feature_name_list)

# * ignore useless features. Also, we only have max 7 qubit, so reduce to fit
df_selected = df[[ 'Pclass', 'Survived','Age', 'Sex', 'Parch']] # Choose desired features and labels(classes)
df_selected

# * make sure all data is viable
print('Total empty values in the Dataset :', df_selected.isnull().sum().sum())
clean_data = df_selected.dropna()
for col in clean_data.columns:
    print('Unique values in {} :'.format(col),len(clean_data[col].unique()))
print(clean_data)

print('Unique values in updated Gender column :', clean_data.Sex.unique())
print('Range of column Age :', (clean_data.Age.min(), clean_data.Age.max()))
print('Unique values in parent/child column :', clean_data.Parch.unique())
print('Unique values in passenger class column :', clean_data.Pclass.unique())

# * we need all values to be numerical
clean_data['Sex'].replace(to_replace = 'male', value = 0, inplace=True)
clean_data['Sex'].replace(to_replace = 'female', value = 1, inplace=True)

# Define features and labels (that contin class information)
# * we need to divide the data into input and output and each of those into training and testing sets
test_ratio = 0.2
seed = 1430
np.random.seed(seed)
X = np.array(clean_data.drop('Survived', axis =1)) # X contains all feature values as array
y = np.array(clean_data.Survived) # y contains class values as array
X = preprocessing.normalize(X, axis=0)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed, stratify=y)


# * to do any quantum operation, we need to define the circuit for data embedding...
feature_dim = X.shape[1]
feature_map_rep = 2
ent = 'linear'

fmap_zz = ZZFeatureMap(feature_dimension=feature_dim, reps=feature_map_rep, entanglement=ent)
# fmap_zz.decompose().draw('mpl', scale=1)


# * ... and for the neural netowrk (learning)
rotations = ['ry', 'rz']
var_form_rep = 2

ansatz_tl = TwoLocal(num_qubits=feature_dim, rotation_blocks=rotations, entanglement_blocks='cx', entanglement=ent, reps=var_form_rep)
ansatz_tl.decompose().draw('mpl', scale=1)

# * the total circuit to run is the combination of the previous two
var_circuit = fmap_zz.compose(ansatz_tl)
# var_circuit.draw('mpl')


# * these callback functions will post-process the measurement data dn show us the progress
def parity(x, num_classes):
    return f"{x:b}".count("1") % num_classes

def one_qubit_binary(x):
    return x % 2

# # * You might wanna use a GPU if available, so let's see what we can use
# # print(providers.aer.StatevectorSimulator().available_devices())
# 
# # * choosing a simulator that can be parallelized
# statevec_sim = qiskit.providers.aer.backends.StatevectorSimulator(max_parallel_threads = 16, max_parallel_experiments=0)

statevec_sim = qiskit.BasicAer.get_backend("qasm_simulator")


# * We need the instance that is actually running the circuit. To get statistically relevant data (shot noise), we need many samples!
qinst = QuantumInstance(statevec_sim, 2048)

# * 
num_classes = 2
qnn = CircuitQNN(circuit=var_circuit,
                 input_params=fmap_zz.parameters,  # if your embedding strategy is not a parametrized circuit 
                                                   # (e.g. amplitude encoding) you will have to do some extra work!
                 weight_params=ansatz_tl.parameters,  # if you train your embedding as well, the trainable
                                                      # parameters of the embedding must be added
                 interpret=one_qubit_binary,
                 output_shape=num_classes,
                 gradient=None,
                 quantum_instance=qinst)

def get_one_hot_encoding(y):
    unique_labels = np.unique(y, axis=0)
    y_one_hot = [(np.eye(len(unique_labels))[np.where(unique_labels == y_i)]).reshape(len(unique_labels)) for y_i in y]

    return np.array(y_one_hot)

y_train_1h = get_one_hot_encoding(y_train)
y_test_1h = get_one_hot_encoding(y_test)

def callback(nfev, params, fval, stepsize, accepted=None):
    """
    Can be used for SPSA and GradientDescent optimizers
    nfev: the number of function evals
    params: the current parameters
    fval: the current function value
    stepsize: size of the update step
    accepted: whether the step was accepted (not used for )
    """
    global loss_recorder

    loss_recorder.append(fval)
    print(f'{nfev} - {fval}')

# max_itr = 500
max_itr = 50
spsa_opt = SPSA(maxiter=max_itr, callback=callback)
loss_recorder = []
initial_point = np.random.random((len(ansatz_tl.parameters),))
vqc = NeuralNetworkClassifier(neural_network=qnn,
                              loss=CrossEntropyLoss(),
                              one_hot=True,
                              optimizer=spsa_opt,
                              initial_point=initial_point)

x_train_norm = np.array([x/np.linalg.norm(x) for x in x_train])
x_test_norm = np.array([x/np.linalg.norm(x) for x in x_test])

# * finally, train the system
vqc = vqc.fit(x_train_norm, y_train_1h)


# * how well did we do?
score_train = vqc.score(x_train_norm, y_train_1h)
score_test = vqc.score(x_test_norm, y_test_1h)
print(f'Score on the train set {score_train}')
print(f'Score on the test set {score_test}')

plt.figure(1000)
plt.plot(loss_recorder)
plt.show()