# https://github.com/cepc/HiggsQML/blob/66cd2fce54823dc1ef928b655f7920ffc06080a0/Higgs_vqc.py
#variational quantum algorithm for 2 qubits with simulator of ibm
import pandas as pd
import numpy as np
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import SPSA
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name

data1 = pd.read_csv('signal.txt', usecols=['invariant_mass_of_qq', 'invariant_mass_of_vv'])
data1['type'] = 'A'
data2 = pd.read_csv('background.txt', usecols=['invariant_mass_of_qq', 'invariant_mass_of_vv'])
data2['type'] = 'B'
data3 = pd.concat([data1, data2])
data3 = data3.reset_index(drop=True)

list1=[i*0.1 for i in data3.invariant_mass_of_qq]
data3.invariant_mass_of_qq=list1
list2=[i*0.06 for i in data3.invariant_mass_of_vv]
data3.invariant_mass_of_vv=list2

train = data3.sample(frac=0.5)
test = data3[~data3.index.isin(train.index)]

colors = {'A':'red', 'B':'blue'}
train.plot.scatter(x='invariant_mass_of_qq', y='invariant_mass_of_vv', c=train.type.apply(lambda x: colors[x])) 

train_input = {'A': [], 'B': []}
test_input = {'A': [], 'B': []}
for i in train.values:
    train_input[i[2]].append([i[0], i[1]])
for i in train_input:
    train_input[i] = np.array(train_input[i])
for i in test.values:
    test_input[i[2]].append([i[0], i[1]])
for i in test_input:
    test_input[i] = np.array(test_input[i])
    
seed = 10599
aqua_globals.random_seed = seed

feature_dim=2
feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2)
datapoints, class_to_label = split_dataset_to_data_and_labels(test_input)
optimizer = SPSA(maxiter=40, c0=4.0, skip_calibration=True)
var_form = TwoLocal(feature_dim, ['ry', 'rz'], 'cz', reps=3)

vqc = VQC(optimizer, feature_map, var_form, train_input, test_input, datapoints[0])
backend = BasicAer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed)
result = vqc.run(quantum_instance)

print('Prediction from datapoints set:')
print(f'  ground truth: {map_label_to_class_name(datapoints[1], vqc.label_to_class)}')
print(f'  prediction:   {result["predicted_classes"]}')

a = map_label_to_class_name(datapoints[1], vqc.label_to_class)
b = result["predicted_classes"]
r = len(a)

result = {}
for i in range(r):
    if a[i] == b[i]:
        if a[i] not in result:
            result[a[i]] = 1
        else:
            result[a[i]] += 1
a1 = (result['A']/a.count('A'))*100
b1 = (result['B']/a.count('B'))*100

print(f'The background rejection:{round(a1,3)}%.The signal efficiency:{round(b1,3)}%')

      
      
#variational quantum algorithm for 5 qubits with the real quantum machine of ibm
import pandas as pd
import numpy as np
from qiskit import BasicAer,IBMQ,assemble
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.providers.ibmq import least_busy
from qiskit.aqua.algorithms import VQC
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name

data1 = pd.read_csv('test20(signal).txt', usecols=['dijet_m', 'vis_all_rec_m','dijet_rec_m','jet_sub_e','vis_all_m'])
data1['type'] = 'A'
data2 = pd.read_csv('test20(background).txt', usecols=['dijet_m', 'vis_all_rec_m','dijet_rec_m','jet_sub_e','vis_all_m'])
data2['type'] = 'B'
data3 = pd.concat([data1, data2])
data3 = data3.reset_index(drop=True)

list1=[i*2*np.pi/70 for i in data3.dijet_m]
data3.dijet_m=list1
list2=[i*2*np.pi/120 for i in data3.vis_all_rec_m]
data3.vis_all_rec_m=list2
list3=[i*2*np.pi/225 for i in data3.dijet_rec_m]
data3.dijet_rec_m=list3
list4=[i*2*np.pi/40 for i in data3.jet_sub_e]
data3.jet_sub_e=list4
list5=[i*2*np.pi/190 for i in data3.vis_all_m]
data3.vis_all_m=list5

train = data3.sample(frac=0.5)
test = data3[~data3.index.isin(train.index)]

colors = {'A':'red', 'B':'blue'}
train.plot.scatter(x='dijet_m', y='vis_all_rec_m', c=train.type.apply(lambda x: colors[x])) 
test.plot.scatter(x='dijet_m', y='vis_all_rec_m', c=test.type.apply(lambda x: colors[x])) 

train_input = {'A': [], 'B': []}
test_input = {'A': [], 'B': []}
for i in train.values:
    train_input[i[5]].append([i[0], i[1], i[2],i[3],i[4]])
for i in train_input:
    train_input[i] = np.array(train_input[i])
for i in test.values:
    test_input[i[5]].append([i[0], i[1], i[2],i[3],i[4]])
for i in test_input:
    test_input[i] = np.array(test_input[i])
    
seed = 10599
algorithm_globals.random_seed = seed

feature_dim=5
feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2)
datapoints, class_to_label = split_dataset_to_data_and_labels(test_input)
optimizer = SPSA(maxiter=40)
var_form = TwoLocal(feature_dim, ['ry', 'rz'], 'cz', reps=3)

vqc = VQC(optimizer, feature_map, var_form, train_input, test_input, datapoints[0])

shots = 1024
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 2 
                                       and not x.configuration().simulator 
                                       and x.status().operational==True))
print("least busy backend: ", backend)
quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed)

result = vqc.run(quantum_instance)

print(f'Testing success ratio: {result["testing_accuracy"]}')
print()
print('Prediction from datapoints set:')
print(f'  ground truth: {map_label_to_class_name(datapoints[1], vqc.label_to_class)}')
print(f'  prediction:   {result["predicted_classes"]}')

a = map_label_to_class_name(datapoints[1], vqc.label_to_class)
b = result["predicted_classes"]
r = len(a)

result = {}
for i in range(r):
    if a[i] == b[i]:
        if a[i] not in result:
            result[a[i]] = 1
        else:
            result[a[i]] += 1
a1 = (result['A']/a.count('A'))*100
b1 = (result['B']/a.count('B'))*100

print(f'The background rejection:{round(a1,3)}%.The signal efficiency:{round(b1,3)}%')
