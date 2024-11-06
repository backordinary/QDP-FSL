# https://github.com/BriefCud/QML-for-btag/blob/fa3f5d2353496abb18aae7abfbeb7cfeb6ad3323/harware_pred.py
# Imports
import pandas as pd
import numpy as np
from load_dataset import load_dataset
from sklearn.metrics import roc_curve, roc_auc_score 
from sklearn.preprocessing import MinMaxScaler
import pennylane as qml
from functools import partial
import jax
import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import pennylane_qiskit
from qiskit import IBMQ

# Constants
SEED=0              # Fix it for reproducibility (kind of)
TRAIN_SIZE = 300
TEST_SIZE = 300  # Number of jets (has to be even) for testing
N_QUBITS = 16       # One qubit per feature
N_LAYERS = 2        # Add more layers for extra complexity
LR=1e-3           # Learning rate of the ADAM optimizer
N_EPOCHS = 10     # Number of training epochs
BATCH_SIZE = 300
N_PARAMS_B = 3

# user token, input manually
t = ""
provider = IBMQ.enable_account(t,hub='ibm-q-cern',group='infn', project='qlm4btag')
ibm_device = qml.device('qiskit.ibmq', provider=provider,backend='ibmq_qasm_simulator',wires=N_QUBITS,imbqx_token=t,shots=4096)


def Get_Path():
  input_model = input("Which model would you like to choose from, Strong=0, MPS=1, TTN=2? Enter a number:")
  path="/content/drive/MyDrive/Colab Notebooks/BTR/"
  n=0
  file_name = ''
  if(input_model=='0'):
    path = path + "strong_w/"
    n=0
    file_name = 'strong_data/strong_loss_accuracy_data_hardware_testsize'+str(TEST_SIZE)+'.csv'
  elif(input_model=='1'):
    path = path + "mps_w/"
    n=1
    file_name = 'mps_data/mps_loss_accuracy_data_hardware_testsize'+str(TEST_SIZE)+'.csv'
  elif(input_model=='2'):
    path = path + "ttn_w/"
    n=2
    file_name = 'ttn_data/ttn_loss_accuracy_data_hardware_testsize'+str(TEST_SIZE)+'.csv'
  else:
    print("Model does not exist")
    exit()

  return path, n

def Get_Weights(path):
  fweights = os.scandir(path) # Get the .npy weight files
  with fweights as entries:
    for entry in entries:
      print(entry.name)
  print("Please Choose a file from above to load the weights for the Strong model, otherwise press the space bar, then enter to pass this stage.")
  w_f = input("Enter file name:  ")

  return np.load(path+"/"+w_f)

path, n = Get_Path()
WEIGHTS = Get_Weights(path)
N_PARAMS_B = WEIGHTS.shape[1]

def Block(weights,wires):
  qml.RZ(weights[0], wires=wires[0])
  qml.RY(weights[1], wires=wires[1])
  qml.U1(weights[2], wires=wires[0])
  #qml.RX(weights[3], wires=wires[1])
  qml.CZ(wires=wires)
    
@qml.batch_input(argnum=0)
@qml.qnode(ibm_device)
def Strong_Circuit(x):
    qml.AngleEmbedding(x, wires=range(N_QUBITS))
    qml.StronglyEntanglingLayers(WEIGHTS,wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

@qml.batch_input(argnum=0)
@qml.qnode(ibm_device, interface=None, diff_method=None)
def Mps_Circuit(x):
  qml.AngleEmbedding(x, wires=range(N_QUBITS))
  qml.MPS(wires=range(N_QUBITS), n_block_wires=2,block=Block, n_params_block=N_PARAMS_B, template_weights=WEIGHTS) 
  return qml.expval(qml.PauliZ(N_QUBITS-1))

@qml.batch_input(argnum=0)
@qml.qnode(ibm_device)
def Ttn_Circuit(x):
  qml.AngleEmbedding(x, wires=range(N_QUBITS))
  qml.TTN(wires=range(N_QUBITS), n_block_wires=2,block=Block, n_params_block=N_PARAMS_B, template_weights=WEIGHTS)
  return qml.expval(qml.PauliZ(N_QUBITS-1))

jax_device =  qml.device("default.qubit.jax", wires=N_QUBITS,prng_key = jax.random.PRNGKey(737))

@partial(jax.vmap,in_axes=[0])
@qml.qnode(jax_device,interface='jax')  # Create a Pennylane QNode
def Jax_Circuit(x):
  qml.AngleEmbedding(x,wires=range(N_QUBITS))
  qml.MPS(wires=range(N_QUBITS), n_block_wires=2,block=Block, n_params_block=N_PARAMS_B, template_weights=WEIGHTS) # Variational layer
  # qml.TTN(wires=range(N_QUBITS), n_block_wires=2,block=block_one, n_params_block=N_PARAMS_B, template_weights=w) 
  return qml.expval(qml.PauliZ(N_QUBITS-1))

def Accuracy(pred,y):
  return np.mean(jax.numpy.sign(pred) == y)

def Batch(x,y):
  z = int(len(x) / BATCH_SIZE)
  data = np.column_stack([x,y])
  return np.split(data[:,0:N_QUBITS],z), np.split(data[:,-1],z),z

def Hardware_Predictions(x,n):
  path = "/content/drive/MyDrive/Colab Notebooks/BTR/"
  rank = np.linalg.matrix_rank(x)
  print("Getting Hardware Predictions...")   
  if(n==0):
    z = (int(TEST_SIZE/BATCH_SIZE))
    predictions = np.zeros((z,BATCH_SIZE))
    for i in range(z):
      predictions[i] = Strong_Circuit(x[i])
      fname = path + "strong_data/strong_hardware_pred_batch"+str(i+1)+"_size"+str(TEST_SIZE)+".npy"
      np.save(fname,pred)
    return np.reshape(predictions,(z*BATCH_SIZE))
  elif(n==1):
    z = (int(TEST_SIZE/BATCH_SIZE))
    predictions = np.zeros((z,BATCH_SIZE))
    for i in range(z):
      predictions[i] = Mps_Circuit(x[i])
      fname = path + "mps_data/mps_hardware_pred_batch"+str(i+1)+"_size"+str(TEST_SIZE)+".npy"
      np.save(fname,pred)
    return np.reshape(predictions,(z*BATCH_SIZE))
  elif(n==2):
    z = (int(TEST_SIZE/BATCH_SIZE))
    predictions = np.zeros((z,BATCH_SIZE))
    for i in range(z):
      predictions[i] = Ttn_Circuit(x[i])
      fname = path + "ttn_data/ttn_hardware_pred_batch"+str(i+1)+"_size"+str(TEST_SIZE)+".npy"
      np.save(fname,pred)
    return np.reshape(predictions,(z*BATCH_SIZE))


train_features,train_target,test_features,test_target = load_dataset(TRAIN_SIZE,TEST_SIZE,SEED)
x,y,z=Batch(test_features,test_target)
ibm_pred = Hardware_Predictions(x,n)
fname = "full_pred_circuit"+str(n)+".npy"
np.save(fname,ibm_pred)
ibm_acc = Accuracy(ibm_pred,test_target)
jax_pred = np.zeros((z,BATCH_SIZE))
for i in range(z):
  jax_pred[i] = Jax_Circuit(x[i])
jax_pred = np.reshape(jax_pred,(z*BATCH_SIZE))
jax_acc = Accuracy(jax_pred, test_target)
print('IBM accuracy: ' + str(ibm_acc))
print('JAX accuracy: ' + str(jax_acc))
print(np.mean(np.abs(ibm_pred - jax_pred)))
