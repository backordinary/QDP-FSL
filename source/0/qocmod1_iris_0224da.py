# https://github.com/shukob/qocc/blob/f07577c8c27fc53611e8533a2d7f9db192158562/QOCmod1/qocmod1_iris.py
"""
Classificador que recebe vetores com 4 features
"""

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, BasicAer
import numpy as np
import util

# beta angles for 4 feature vectors
def betaAngles(v):
    
    beta0 = 2 * np.arcsin(np.sqrt(v[1] ** 2) / np.sqrt(v[0] ** 2 + v[1] ** 2))
    beta1 = 2 * np.arcsin(np.sqrt(v[3] ** 2) / np.sqrt(v[2] ** 2 + v[3] ** 2))
    beta2 = 2 * np.arcsin(np.sqrt(v[2] ** 2 + v[3] ** 2) / np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2 + v[3] ** 2))
    
    angles = [beta2, beta1, beta0]
    
    return angles

def initQC(n):
    q = QuantumRegister(n)
    c = ClassicalRegister(n)
    qc = QuantumCircuit(q, c)
    
    return q, c, qc

def simQasm(qcircuit):
    backend_sim = BasicAer.get_backend('qasm_simulator')
    job_sim = execute(qcircuit, backend_sim, shots=1000)
    result = job_sim.result()
    result_counts = result.get_counts(qcircuit)
    
    return result_counts

# load 4 feature test vector
def loadTest(beta, qcircuit, c, d1, d2):
    
    qcircuit.cry(-beta[2], c, d1)
    qcircuit.barrier()
    
    qcircuit.mcry(-beta[1], [c, d1], d2, None, mode='noancilla')
    qcircuit.barrier()
    
    qcircuit.x(d1)
    qcircuit.barrier()
    qcircuit.mcry(-beta[0], [c, d1], d2, None, mode='noancilla')
    qcircuit.barrier()
    qcircuit.x(d1)
    qcircuit.barrier()
    
# load 4 feature training vectors
def loadTraining(beta, qcircuit, c1, c2, d1, d2):
    
    qcircuit.mcry(-beta[2], [c1, c2], d1, None, mode='noancilla')
    qcircuit.barrier()
    
    qcircuit.mcry(-beta[1], [c1, c2, d1], d2, None, mode='noancilla')
    qcircuit.barrier()
    
    qcircuit.x(d1)
    qcircuit.barrier()
    qcircuit.mcry(-beta[0], [c1, c2, d1], d2, None, mode='noancilla')
    qcircuit.barrier()
    qcircuit.x(d1)
    qcircuit.barrier()
    
def circuit(test, training):
    
    q, c, qc = initQC(4)
    
    ancilla = q[0]
    index = q[1]
    data_1 = q[2]
    data_2 = q[3]
    
    # ancilla and index in uniform superposition
    qc.h(ancilla)
    qc.h(index)
    qc.barrier()
    
    # load test vector
    loadTest(test, qc, ancilla, data_1, data_2)
    
    # flip the ancilla qubit moves the input vector to the |0> state of the ancilla
    qc.x(ancilla)
    qc.barrier()
    
    # load first training vector
    loadTraining(training[0], qc, ancilla, index, data_1, data_2)
    
    # flip the index qubit moves the first training vector to the |0> state of the index qubit
    qc.x(index)
    qc.barrier()
    
    # load second training vector
    loadTraining(training[1], qc, ancilla, index, data_1, data_2)
    
    # the Hadamard gate interferes the copies of the test vector with the training vectors
    qc.h(ancilla)
    qc.barrier()
    
    # measure ancilla qubit
    qc.measure(ancilla, c[0])
    
    return qc

def classifySim(test_vector, training_set):
    
    qc = circuit(test_vector, training_set)
    
    result_counts = simQasm(qc)
    total_samples = sum(result_counts.values())
    
    # Define lambda function that retrieves only results where the ancilla is in the |0> state
    ancilla_func = lambda counts: [(state, occurences) for state, occurences in counts.items() if state[-1] == '0']
    
    # Retrive counting values for the ancilla qubit by applying ancilla_func
    ancilla_counts = dict(ancilla_func(result_counts))
    ancilla_samples = sum(ancilla_counts.values())
    
    # Probability of ancilla qubit is in the |0> state
    ancilla_prob = ancilla_samples/total_samples
    
    return total_samples, ancilla_counts, ancilla_samples, ancilla_prob

def experiment(classe):
    
    # classes and features of the iris dataset
    classes = [0,1]
    features = [0,1,2,3]
    
    X, y = util.load_iris(classes=classes, features=features)
    X = util.preprocess(X)
    
    if(classe==0):
        cent = util.centroid2(X[0:50])
    elif(classe==1):
        cent = util.centroid2(X[50:100])
        
    training_set = util.minDistances(X[0:50], cent, 2)
    
    training_angles = []
    
    for i in range(len(training_set)):
        tv = betaAngles(X[training_set[i]])
        training_angles.append(tv)
    
    success = 0
    predict = []
    
    for i in range(100):
        print(i)
        
        if i not in training_set:
            classify = betaAngles(X[i])
            
            total_samples, ancilla_counts, ancilla_samples, ancilla_prob = classifySim(
                        test_vector=classify, training_set=training_angles)
            
            if(classe==0):
                if (i < 50):
                    if (ancilla_prob > 0.5):
                        success += 1
                else:
                    if (ancilla_prob < 0.5):
                        success += 1
                    
            if(classe==1):
                if ((i < 50) and (ancilla_prob < 0.5)):
                    success += 1
                elif((i >= 50) and (ancilla_prob > 0.5)):
                    success += 1
            
            predict.append(ancilla_prob)
            
    print(f'Success probability: {success/98}')
    print(f'Fail probability: {1-(success/98)}\n')
    print(f'Predict: {predict}')