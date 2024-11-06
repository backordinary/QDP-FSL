# https://github.com/marcinjastrzebski8/QKE_for_tracking/blob/5d57736f43bd1fea140b1f16111482e0c01b6505/first_try_with_trackml_data.py
"""
Example of how quantum kernel estimation with a custom kernel can be used to classify data.
Edge data originates from TrackML, pre-processing by Tuysuz et al, adapted for QKE by Marcin Jastrzebski
"""

#general imports
import argparse
import os
import numpy as np
from datetime import datetime
#qiskit
from qiskit import (Aer,IBMQ)
IBMQ.load_account()
IBMQ.providers()
provider = IBMQ.get_provider(group='open')
from qiskit.utils import QuantumInstance
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.kernels import QuantumKernel
#machine learning (external to qiskit)
from sklearn.svm import SVC
#my own library
from quantum_circuit import (param_U_gate,param_feature_map) #quantum_circuit isn't the most original and descriptive name, easy to mistake with qiskit's stuff


startTime = datetime.now()

#Parser is specific to the edge data provided. Events numbers should be from 0 to 49 
#They are divided into train and test folders
parser = argparse.ArgumentParser(description='Load one (!) event for training and one for testing. Size decides how many edges from the event are used.')
add_arg = parser.add_argument

add_arg('--train_event_number', default = 0, type = int)
add_arg('--test_event_number', default = 0, type = int)  
add_arg('--train_size', default = 1000, type = int)
add_arg('--test_size', default = 1000, type = int)
args = parser.parse_args()

train_event_number = args.train_event_number
test_event_number = args.test_event_number
train_size = int(args.train_size)
test_size = int(args.test_size)

#Make sure you're pointing to the right directory
train_files_folder_str = '/Users/marcinjastrzebski/Desktop/ACADEMIA/TrackML/Simple_QKE/tuy_data/ready_train'
train_files_folder_as_list = os.listdir(train_files_folder_str)
train_files_folder_as_list.sort()

train_data_file = train_files_folder_as_list[int(train_event_number)] #this is a string
train_data = np.load(train_files_folder_str+'/'+train_data_file) #this is a numpy array
train_label_file = train_files_folder_as_list[int(train_event_number)+50]
train_label = np.load(train_files_folder_str+'/'+train_label_file)

test_files_folder_str = '/Users/marcinjastrzebski/Desktop/ACADEMIA/TrackML/Simple_QKE/tuy_data/ready_test'
test_files_folder_as_list = os.listdir(test_files_folder_str)
test_files_folder_as_list.sort()

test_data_file = test_files_folder_as_list[int(test_event_number)] #this is a string
test_data = np.load(test_files_folder_str+'/'+test_data_file) #this is a numpy array
test_label_file = test_files_folder_as_list[int(test_event_number)+50]
test_label = np.load(test_files_folder_str+'/'+test_label_file)

#if you pass more edges than exist in event, whole event is used
if train_size < np.shape(train_data)[0]:
    train_features = train_data[:train_size]
    train_labels = train_label[:train_size]
else:
    train_features = train_data
    train_labels = train_label

if test_size < np.shape(test_data)[0]:
    test_features = test_data[:test_size]
    test_labels = test_label[:test_size]
else:
    test_features = test_data
    test_labels = test_label

#define what simulator we're using
statevector_backend = QuantumInstance(Aer.get_backend("statevector_simulator"))

#data dimension is 6 - two 3D points which make an edge
data_dim = 6

#one parameter per feature 
params_j = ParameterVector('phi',data_dim)
my_U = param_U_gate.U_flexible(data_dim,params_j,single_mapping=1,pair_mapping=1)
my_map = param_feature_map.feature_map(data_dim,my_U)
my_kernel = QuantumKernel(feature_map=my_map,quantum_instance = statevector_backend) 

#train part
adhoc_matrix_train = my_kernel.evaluate(x_vec=train_features)
adhoc_svc = SVC(kernel="precomputed")
adhoc_svc.fit(adhoc_matrix_train, train_labels)

#test part 
adhoc_matrix_test = my_kernel.evaluate(x_vec=test_features, y_vec=train_features)
adhoc_score = adhoc_svc.score(adhoc_matrix_test, test_labels)

time_taken = datetime.now() - startTime

#format adapted from other analysis so might not be ideal
results = np.array([
{'train_event': train_event_number, 'test_event': test_event_number},
{'train_total_edges': np.shape(train_data)[0], 'test_total_edges': np.shape(test_data)[0]},
{'time_taken': time_taken},
{'score': adhoc_score}
])
results_file_name = 'results_event_'+str(train_event_number)+'.npy'

np.save(results_file_name,results) #has to be loaded with allow_pickle = True 


