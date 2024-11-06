# https://github.com/firatk/QuantumKnapsack/blob/d4b5703ace7baa98714573a1c40f4866e4812898/knapsack_withSolvers.py
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 21:29:44 2020

@author: Grup Mavi
"""

import numpy as np

from random import randrange

from qiskit import BasicAer, IBMQ
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit.optimization.algorithms import GroverOptimizer, MinimumEigenOptimizer, RecursiveMinimumEigenOptimizer
from qiskit.optimization.problems import QuadraticProgram

def initializeCostMatrixMemoization(totWeight, itemCount):
    global costMatrix;
    for ii in range(itemCount + 1):
        costMatrix.append([]);
        for jj in range(totWeight + 1):
            costMatrix[ii].append(-1);
    return;
    

def generateItems(*args):
    weightValueList    = [[],[]];
    itemCount   = args[0];
    minWeight   = args[1];
    maxWeight   = args[2];
    minValue    = args[3];
    maxValue    = args[4];
    valueQuant  = 1;
    weightQuant = 1;
    if(len(args) > 5):
        weightQuant     = args[5];
        if(len(args) == 7):
            valueQuant  = args[6];
    for ii in range(itemCount):
        tempWeight  = randrange(minWeight, maxWeight+1, weightQuant);
        weightValueList[0].append(tempWeight);
        tempValue   = randrange(minValue, maxValue+1, valueQuant);
        weightValueList[1].append(tempValue);
    return weightValueList;

def recursiveKnapsack(itemCount, remainingWeight):
    global costMatrix, backPackCap, itemsWeight, itemsValue;
    if ((itemCount == 0) or (remainingWeight <= 0)):
        return 0;
    if (costMatrix[itemCount][remainingWeight] >= 0):
        return costMatrix[itemCount][remainingWeight];
    if(itemsWeight[itemCount] > remainingWeight):
        costMatrix[itemCount][remainingWeight]  = recursiveKnapsack(itemCount - 1, remainingWeight);
    else:
        costMatrix[itemCount][remainingWeight]  = max(recursiveKnapsack(itemCount - 1, remainingWeight), recursiveKnapsack(itemCount - 1, remainingWeight - itemsWeight[itemCount]) + itemsValue[itemCount]);
    return costMatrix[itemCount][remainingWeight];

def backtraceItems(itemCount, totalCap):
    global costMatrix, itemsWeight, itemsValue;
    chosenItems = [];
    for ii in range(itemCount):
        chosenItems.append(0);
    itemIndex   = itemCount;
    weightIndex = totalCap;
    while(itemIndex > 0):
        if (costMatrix[itemIndex][weightIndex] == costMatrix[itemIndex-1][weightIndex]):
            itemIndex   -= 1;
        else:
            weightIndex -= itemsWeight[itemIndex];
            itemIndex   -= 1;
            chosenItems[itemIndex]  = 1;    
    return chosenItems;

provider = IBMQ.load_account();

costMatrix  = [];
backPackCap = 8;
numOfItems  = 6;                # Number of objects

rangeMinWeight  = 1;
rangeMaxWeight  = 4;
rangeMinValue   = 10;
rangeMaxValue   = 50;

# For choosing A & B: 0 < B*max(itemValues) < A as described in Lucas article
A   = 1000;                      # For quadratic modeling chosen at start
B   = 10;                       # For quadratic modeling chosen at start

initializeCostMatrixMemoization(backPackCap, numOfItems);

valuables   = generateItems(numOfItems, rangeMinWeight, rangeMaxWeight, rangeMinValue, rangeMaxValue);
itemsWeight = valuables[0];     # Weight of the objects
itemsWeight.insert(0, None);
itemsValue  = valuables[1];     # Value of the objects
itemsValue.insert(0, None);
recursiveKnapsack(numOfItems, backPackCap);
chosenItems = backtraceItems(numOfItems, backPackCap);

del itemsValue[0];
del itemsWeight[0];
print("Backpack Capacity = ", backPackCap);
print("Items Weight = ", itemsWeight);
print("Items Value = ", itemsValue);
"""print("Cost Matrix")
for ii in range(numOfItems+1):
    print(costMatrix[ii]);"""

print('Max Value = ', costMatrix[numOfItems][backPackCap]);
print("Chosen Items = ", chosenItems);


# Weight values matrix as described in Lucas article
W_matrix    = [];
# Weight values Transpose of  matrix
W_matrix_T  = [];
for ii in range(numOfItems):
    W_matrix_T.append(-itemsWeight[ii]);
for ii in range(backPackCap):
    W_matrix_T.append(ii+1);

W_matrix    =  np.array([W_matrix_T]);
W_matrix    = W_matrix.T;
AWWT        = A * W_matrix * W_matrix_T;

L_T   = [];                             # Transpose of the lambda vector
for ii in range(numOfItems):
    L_T.append(0);
for i in range(backPackCap):
    L_T.append(1);

L   = np.array([L_T]).T;                # Lambda vector
ALLT = A * L * L_T;

ALT = np.dot(2 * A, L_T);

# Items value vector
BV_vector   = [];

for ii in range(numOfItems):
    BV_vector.append(-itemsValue[ii] * B);
for ii in range(backPackCap):
    BV_vector.append(0);

# Variables vector: 'x' variables --> items, 'y' variables --> weights
z   = [];
for ii in range(numOfItems):
    z.append('x' + str(ii));
for ii in range(backPackCap):
    z.append('y' + str(ii));


linearCoeff     = ALT + BV_vector;
#linearCoeff     = linearCoeff.tolist();
quadraticCoeffs = ALLT + AWWT;

tempMax = 1000;
for ii in range(len(z)):
    if (tempMax < max(abs(quadraticCoeffs[ii]))):
        tempMax = max(abs(quadraticCoeffs[ii]));
        
penalty = tempMax**3;

quadraticCoeffDict  = {};
for ii in range(len(z)):
    for jj in range(len(z)):
        if ((ii >= numOfItems) and (jj >= numOfItems)):
            if (ii != jj):
                quadraticCoeffDict[(z[ii],z[jj])] = penalty;
            else:
                quadraticCoeffDict[(z[ii],z[jj])] = quadraticCoeffs[ii][jj];
        else:
            quadraticCoeffDict[(z[ii],z[jj])] = quadraticCoeffs[ii][jj];

#backend = BasicAer.get_backend('statevector_simulator')

qubo    = QuadraticProgram();
for ii in range(numOfItems + backPackCap):
    qubo.binary_var(z[ii]);

qubo.minimize(linear=linearCoeff, quadratic=quadraticCoeffDict);

print(qubo.export_as_lp_string());

"""qaoa_mes = QAOA(quantum_instance=BasicAer.get_backend('statevector_simulator'));

qaoa = MinimumEigenOptimizer(qaoa_mes)   # using QAOA
qaoa_result = qaoa.solve(qubo)
print(qaoa_result)"""

quantumBackend  = QuantumInstance(provider.get_backend('ibmq_qasm_simulator'), shots = 2048, skip_qobj_validation=False);

qaoa_mes = QAOA(quantum_instance = quantumBackend);
qaoa = MinimumEigenOptimizer(qaoa_mes);     # using QAOA
qaoa_result = qaoa.solve(qubo);
print(qaoa_result);

exact_mes   = NumPyMinimumEigensolver();
exact       = MinimumEigenOptimizer(exact_mes);   # using the exact classical numpy minimum eigen solver
exact_result= exact.solve(qubo);
print(exact_result);

rqaoa = RecursiveMinimumEigenOptimizer(min_eigen_optimizer=qaoa, min_num_vars=14, min_num_vars_optimizer=MinimumEigenOptimizer(NumPyMinimumEigensolver()));
rqaoa_result = rqaoa.solve(qubo);
print(rqaoa_result);

grover_mes  = GroverOptimizer(numOfItems + backPackCap, num_iterations = 15, quantum_instance = quantumBackend);
grover_result   = grover_mes.solve(qubo);
print(grover_result);

  
print("x={}".format(grover_result.x))
print("fval={}".format(grover_result.fval))