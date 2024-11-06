# https://github.com/ayush0624/QIGA2-Tuned/blob/924928d6d4532d9456cb883903e25b4e05ea70a4/Code/QIGA-2/QIGA%20Program.py
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute, BasicAer, IBMQ
from qiskit.tools.visualization import circuit_drawer
from qiskit.tools.visualization import iplot_histogram
from qiskit.tools.visualization import plot_state_qsphere
from qiskit.mapper import CouplingMap, Layout
import random
import numpy as np

# set up quantum registers
qubit_number = 2
register_number = 4
weight_count = register_number//2
register_count = 0

# Choose a real device 
APItoken = '5b2f71479eae0159258df0ece626df4f137a6fa7126058500c086b17aa23333b244c003475c8b7f1c2c162c52fc81f2d272d300881e872cf1ba28a3060afe090'
url = 'https://quantumexperience.ng.bluemix.net/api'

#Choose Device
IBMQ.enable_account(APItoken, url=url)
IBMQ.load_accounts()
print(IBMQ.backends(name='ibmq_qasm_simulator', operational=True))
print(IBMQ.backends())
realBackend = IBMQ.backends(name='ibmq_qasm_simulator', operational=True)[0]
print(realBackend)

#Set up Probability Amplitude Simulator
simulator = BasicAer.backends(name='statevector_simulator')[0]

#Create quantum population and quantum chromosomes
registers = [QuantumRegister(qubit_number) for i in range(register_number)]
classicalregisters = [ClassicalRegister(qubit_number) for i in range(register_number)]
#print(registers)
#print(classicalregisters)

job_results = []
weightVals = np.array([])
profitVals = np.array([])
resultVisual = np.array([])
totalWeight = 0
w_o = 0
p_o = 0
cap = 0

for x in range(weight_count):
        w_o = int((random.random() * 9) + 1)
        p_o = w_o + 5

        weightVals = np.append(weightVals, w_o)
        profitVals = np.append(profitVals, p_o)


cap = sum(weightVals)
index = 0
#print(weightVals)

def knapsack(data, weight, p, chromoCount):
    currentWeight = knapsackWeight[chromoCount]
    currentProfit = knapsackWeight[chromoCount]

    print('currentProfit', currentProfit)

    profit = p * int(data, 2)

    print('profit',profit)

    knapsackWeight[chromoCount] = currentWeight + weight
    knapsackProfit[chromoCount] = currentProfit + profit
        

    print('weight',weight)
    print('cap',cap)


    if knapsackWeight[chromoCount] > cap:
        knapsackWeight[chromoCount] = currentWeight
        knapsackProfit[chromoCount] = currentProfit
    

    return knapsackProfit.item(chromoCount)
#superposition and measurement
for qr in registers:
        cr = classicalregisters[register_count]
        qc = QuantumCircuit(qr, cr)
        qc.h(qr)
        
        # execute the quantum circuit
        
        meas= QuantumCircuit(qr, cr)
        meas.measure(qr, cr)
        
        
        register_count = register_count + 1 
        
        realBackend = IBMQ.backends(name='ibmq_qasm_simulator')[0]
        circ = qc+meas
        result = execute(circ, realBackend, shots=1000).result()
        counts  = result.get_counts(circ)
        job_results.append(counts)
        np.append(resultVisual, result)
        print(counts)
        

knapsackWeight = np.array([])
knapsackProfit = np.array([])
currentWeight = 0
currentProfit = 0
chromCount = 0

print(job_results)

for wX in range(weight_count):
        knapsackProfit = np.append(knapsackProfit, 0)
        knapsackWeight = np.append(knapsackWeight, 0)
        #currentWeight.append(0)
        #currentProfit.append(0)

register_count = 0
profit = []
binaryString = ""
chromosomeCount = 0
weights = 0
p_i = 0
w_i = 0
randInt = 0

for c in range(len(job_results)):
        current_result = job_results[c]
        stringVals = []
        highestVal = 0
          
        DoubleZeroVal = current_result.get('00')
        stringVals.append(DoubleZeroVal)
        ZeroOneVal = current_result.get('01')
        stringVals.append(ZeroOneVal)
        OneZeroVal = current_result.get('10')
        stringVals.append(OneZeroVal)
        DoubleOneVal = current_result.get('11')
        stringVals.append(DoubleOneVal)
        stringVals.sort(reverse=True)
        highestVal = stringVals[0]
        
        print('highestVal', highestVal)

        if highestVal == DoubleOneVal:
                binaryString = binaryString + '11'
        elif highestVal == OneZeroVal:
                binaryString = binaryString + '10'
        elif highestVal == ZeroOneVal:
                binaryString = binaryString + '01'
        elif highestVal == DoubleZeroVal:
                binaryString = binaryString + '00'
                
        
        
        chromosomeCount = chromosomeCount + 1

        print('count',chromosomeCount)

        #evaluate knapsack 
        if chromosomeCount % 2 == 0:
                print(binaryString)
                randInt = random.randint(0, weight_count - 1)

                w_i = weightVals[randInt]
                p_i = profitVals[randInt]

                binaryVal = binaryString
                #binaryVal = format(binaryVal, 'b')

                print('binary_val',binaryVal)
                value = knapsack(binaryVal, w_i, p_i, chromCount)
                profit.append(value)

                binaryString = ""
                chromCount = chromCount + 1

chromCount = 0
chromosomeCount = 0

originalProfit = profit
print(originalProfit)


profit.sort(reverse=True)
print(profit)

b = profit[0]
register_count = 0
j = 2
jString = "" 
#superposition and measurement
for qr in registers:
        cr = classicalregisters[register_count]
        qc = QuantumCircuit(qr, cr)
        qc.h(qr)
        
        # execute the quantum circuit
        
        meas= QuantumCircuit(qr, cr)
        meas.measure(qr, cr)
        
        
        register_count = register_count + 1 
        
        realBackend = IBMQ.backends(name='ibmq_qasm_simulator')[0]
        circ = qc+meas
        result = execute(circ, realBackend, shots=1000).result()
        counts  = result.get_counts(circ)
        job_results.append(counts)
        np.append(resultVisual, result)
        print(counts)
        

knapsackWeight = np.array([])
knapsackProfit = np.array([])
currentWeight = 0
currentProfit = 0
chromCount = 0

print(job_results)

for wX in range(weight_count):
        knapsackProfit = np.append(knapsackProfit, 0)
        knapsackWeight = np.append(knapsackWeight, 0)
        #currentWeight.append(0)
        #currentProfit.append(0)

register_count = 0
profit = []
binaryString = ""
chromosomeCount = 0
weights = 0
p_i = 0
w_i = 0
randInt = 0

for c in range(len(job_results)):
        current_result = job_results[c]
        stringVals = []
        highestVal = 0
          
        DoubleZeroVal = current_result.get('00')
        stringVals.append(DoubleZeroVal)
        ZeroOneVal = current_result.get('01')
        stringVals.append(ZeroOneVal)
        OneZeroVal = current_result.get('10')
        stringVals.append(OneZeroVal)
        DoubleOneVal = current_result.get('11')
        stringVals.append(DoubleOneVal)
        stringVals.sort(reverse=True)
        highestVal = stringVals[0]
        
        print('highestVal', highestVal)

        if highestVal == DoubleOneVal:
                binaryString = binaryString + '11'
        elif highestVal == OneZeroVal:
                binaryString = binaryString + '10'
        elif highestVal == ZeroOneVal:
                binaryString = binaryString + '01'
        elif highestVal == DoubleZeroVal:
                binaryString = binaryString + '00'
                
        
        
        chromosomeCount = chromosomeCount + 1

        print('count',chromosomeCount)

        #evaluate knapsack 
        if chromosomeCount % 2 == 0:
                print(binaryString)
                randInt = random.randint(0, weight_count - 1)

                w_i = weightVals[randInt]
                p_i = profitVals[randInt]

                binaryVal = binaryString
                #binaryVal = format(binaryVal, 'b')

                print('binary_val',binaryVal)
                value = knapsack(binaryVal, w_i, p_i, chromCount)
                profit.append(value)

                binaryString = ""
                chromCount = chromCount + 1

chromCount = 0
chromosomeCount = 0

originalProfit = profit
print(originalProfit)


profit.sort(reverse=True)
print(profit)

b = profit[0]
register_count = 0
j = 2
jString = "" 
#superposition and measurement
for qr in registers:
        cr = classicalregisters[register_count]
        qc = QuantumCircuit(qr, cr)
        qc.h(qr)
        
        # execute the quantum circuit
        
        meas= QuantumCircuit(qr, cr)
        meas.measure(qr, cr)
        
        
        register_count = register_count + 1 
        
        realBackend = IBMQ.backends(name='ibmq_qasm_simulator')[0]
        circ = qc+meas
        result = execute(circ, realBackend, shots=1000).result()
        counts  = result.get_counts(circ)
        job_results.append(counts)
        np.append(resultVisual, result)
        print(counts)
        

knapsackWeight = np.array([])
knapsackProfit = np.array([])
currentWeight = 0
currentProfit = 0
chromCount = 0

print(job_results)

for wX in range(weight_count):
        knapsackProfit = np.append(knapsackProfit, 0)
        knapsackWeight = np.append(knapsackWeight, 0)
        #currentWeight.append(0)
        #currentProfit.append(0)

register_count = 0
profit = []
binaryString = ""
chromosomeCount = 0
weights = 0
p_i = 0
w_i = 0
randInt = 0

for c in range(len(job_results)):
        current_result = job_results[c]
        stringVals = []
        highestVal = 0
          
        DoubleZeroVal = current_result.get('00')
        stringVals.append(DoubleZeroVal)
        ZeroOneVal = current_result.get('01')
        stringVals.append(ZeroOneVal)
        OneZeroVal = current_result.get('10')
        stringVals.append(OneZeroVal)
        DoubleOneVal = current_result.get('11')
        stringVals.append(DoubleOneVal)
        stringVals.sort(reverse=True)
        highestVal = stringVals[0]
        
        print('highestVal', highestVal)

        if highestVal == DoubleOneVal:
                binaryString = binaryString + '11'
        elif highestVal == OneZeroVal:
                binaryString = binaryString + '10'
        elif highestVal == ZeroOneVal:
                binaryString = binaryString + '01'
        elif highestVal == DoubleZeroVal:
                binaryString = binaryString + '00'
                
        
        
        chromosomeCount = chromosomeCount + 1

        print('count',chromosomeCount)

        #evaluate knapsack 
        if chromosomeCount % 2 == 0:
                print(binaryString)
                randInt = random.randint(0, weight_count - 1)

                w_i = weightVals[randInt]
                p_i = profitVals[randInt]

                binaryVal = binaryString
                #binaryVal = format(binaryVal, 'b')

                print('binary_val',binaryVal)
                value = knapsack(binaryVal, w_i, p_i, chromCount)
                profit.append(value)

                binaryString = ""
                chromCount = chromCount + 1

chromCount = 0
chromosomeCount = 0

originalProfit = profit
print(originalProfit)


profit.sort(reverse=True)
print(profit)

b = profit[0]
register_count = 0
j = 2
jString = "" 
#superposition and measurement
for qr in registers:
        cr = classicalregisters[register_count]
        qc = QuantumCircuit(qr, cr)
        qc.h(qr)
        
        # execute the quantum circuit
        
        meas= QuantumCircuit(qr, cr)
        meas.measure(qr, cr)
        
        
        register_count = register_count + 1 
        
        realBackend = IBMQ.backends(name='ibmq_qasm_simulator')[0]
        circ = qc+meas
        result = execute(circ, realBackend, shots=1000).result()
        counts  = result.get_counts(circ)
        job_results.append(counts)
        np.append(resultVisual, result)
        print(counts)
        

knapsackWeight = np.array([])
knapsackProfit = np.array([])
currentWeight = 0
currentProfit = 0
chromCount = 0

print(job_results)

for wX in range(weight_count):
        knapsackProfit = np.append(knapsackProfit, 0)
        knapsackWeight = np.append(knapsackWeight, 0)
        #currentWeight.append(0)
        #currentProfit.append(0)

register_count = 0
profit = []
binaryString = ""
chromosomeCount = 0
weights = 0
p_i = 0
w_i = 0
randInt = 0

for c in range(len(job_results)):
        current_result = job_results[c]
        stringVals = []
        highestVal = 0
          
        DoubleZeroVal = current_result.get('00')
        stringVals.append(DoubleZeroVal)
        ZeroOneVal = current_result.get('01')
        stringVals.append(ZeroOneVal)
        OneZeroVal = current_result.get('10')
        stringVals.append(OneZeroVal)
        DoubleOneVal = current_result.get('11')
        stringVals.append(DoubleOneVal)
        stringVals.sort(reverse=True)
        highestVal = stringVals[0]
        
        print('highestVal', highestVal)

        if highestVal == DoubleOneVal:
                binaryString = binaryString + '11'
        elif highestVal == OneZeroVal:
                binaryString = binaryString + '10'
        elif highestVal == ZeroOneVal:
                binaryString = binaryString + '01'
        elif highestVal == DoubleZeroVal:
                binaryString = binaryString + '00'
                
        
        
        chromosomeCount = chromosomeCount + 1

        print('count',chromosomeCount)

        #evaluate knapsack 
        if chromosomeCount % 2 == 0:
                print(binaryString)
                randInt = random.randint(0, weight_count - 1)

                w_i = weightVals[randInt]
                p_i = profitVals[randInt]

                binaryVal = binaryString
                #binaryVal = format(binaryVal, 'b')

                print('binary_val',binaryVal)
                value = knapsack(binaryVal, w_i, p_i, chromCount)
                profit.append(value)

                binaryString = ""
                chromCount = chromCount + 1

chromCount = 0
chromosomeCount = 0

originalProfit = profit
print(originalProfit)


profit.sort(reverse=True)
print(profit)

b = profit[0]
register_count = 0
j = 2
jString = "" 
#superposition and measurement
for qr in registers:
        cr = classicalregisters[register_count]
        qc = QuantumCircuit(qr, cr)
        qc.h(qr)
        
        # execute the quantum circuit
        
        meas= QuantumCircuit(qr, cr)
        meas.measure(qr, cr)
        
        
        register_count = register_count + 1 
        
        realBackend = IBMQ.backends(name='ibmq_qasm_simulator')[0]
        circ = qc+meas
        result = execute(circ, realBackend, shots=1000).result()
        counts  = result.get_counts(circ)
        job_results.append(counts)
        np.append(resultVisual, result)
        print(counts)
        

knapsackWeight = np.array([])
knapsackProfit = np.array([])
currentWeight = 0
currentProfit = 0
chromCount = 0

print(job_results)

for wX in range(weight_count):
        knapsackProfit = np.append(knapsackProfit, 0)
        knapsackWeight = np.append(knapsackWeight, 0)
        #currentWeight.append(0)
        #currentProfit.append(0)

register_count = 0
profit = []
binaryString = ""
chromosomeCount = 0
weights = 0
p_i = 0
w_i = 0
randInt = 0

for c in range(len(job_results)):
        current_result = job_results[c]
        stringVals = []
        highestVal = 0
          
        DoubleZeroVal = current_result.get('00')
        stringVals.append(DoubleZeroVal)
        ZeroOneVal = current_result.get('01')
        stringVals.append(ZeroOneVal)
        OneZeroVal = current_result.get('10')
        stringVals.append(OneZeroVal)
        DoubleOneVal = current_result.get('11')
        stringVals.append(DoubleOneVal)
        stringVals.sort(reverse=True)
        highestVal = stringVals[0]
        
        print('highestVal', highestVal)

        if highestVal == DoubleOneVal:
                binaryString = binaryString + '11'
        elif highestVal == OneZeroVal:
                binaryString = binaryString + '10'
        elif highestVal == ZeroOneVal:
                binaryString = binaryString + '01'
        elif highestVal == DoubleZeroVal:
                binaryString = binaryString + '00'
                
        
        
        chromosomeCount = chromosomeCount + 1

        print('count',chromosomeCount)

        #evaluate knapsack 
        if chromosomeCount % 2 == 0:
                print(binaryString)
                randInt = random.randint(0, weight_count - 1)

                w_i = weightVals[randInt]
                p_i = profitVals[randInt]

                binaryVal = binaryString
                #binaryVal = format(binaryVal, 'b')

                print('binary_val',binaryVal)
                value = knapsack(binaryVal, w_i, p_i, chromCount)
                profit.append(value)

                binaryString = ""
                chromCount = chromCount + 1

chromCount = 0
chromosomeCount = 0

originalProfit = profit
print(originalProfit)


profit.sort(reverse=True)
print(profit)

b = profit[0]
register_count = 0
j = 2
jString = "" 
#superposition and measurement
for qr in registers:
        cr = classicalregisters[register_count]
        qc = QuantumCircuit(qr, cr)
        qc.h(qr)
        
        # execute the quantum circuit
        
        meas= QuantumCircuit(qr, cr)
        meas.measure(qr, cr)
        
        
        register_count = register_count + 1 
        
        realBackend = IBMQ.backends(name='ibmq_qasm_simulator')[0]
        circ = qc+meas
        result = execute(circ, realBackend, shots=1000).result()
        counts  = result.get_counts(circ)
        job_results.append(counts)
        np.append(resultVisual, result)
        print(counts)
        

knapsackWeight = np.array([])
knapsackProfit = np.array([])
currentWeight = 0
currentProfit = 0
chromCount = 0

print(job_results)

for wX in range(weight_count):
        knapsackProfit = np.append(knapsackProfit, 0)
        knapsackWeight = np.append(knapsackWeight, 0)
        #currentWeight.append(0)
        #currentProfit.append(0)

register_count = 0
profit = []
binaryString = ""
chromosomeCount = 0
weights = 0
p_i = 0
w_i = 0
randInt = 0

for c in range(len(job_results)):
        current_result = job_results[c]
        stringVals = []
        highestVal = 0
          
        DoubleZeroVal = current_result.get('00')
        stringVals.append(DoubleZeroVal)
        ZeroOneVal = current_result.get('01')
        stringVals.append(ZeroOneVal)
        OneZeroVal = current_result.get('10')
        stringVals.append(OneZeroVal)
        DoubleOneVal = current_result.get('11')
        stringVals.append(DoubleOneVal)
        stringVals.sort(reverse=True)
        highestVal = stringVals[0]
        
        print('highestVal', highestVal)

        if highestVal == DoubleOneVal:
                binaryString = binaryString + '11'
        elif highestVal == OneZeroVal:
                binaryString = binaryString + '10'
        elif highestVal == ZeroOneVal:
                binaryString = binaryString + '01'
        elif highestVal == DoubleZeroVal:
                binaryString = binaryString + '00'
                
        
        
        chromosomeCount = chromosomeCount + 1

        print('count',chromosomeCount)

        #evaluate knapsack 
        if chromosomeCount % 2 == 0:
                print(binaryString)
                randInt = random.randint(0, weight_count - 1)

                w_i = weightVals[randInt]
                p_i = profitVals[randInt]

                binaryVal = binaryString
                #binaryVal = format(binaryVal, 'b')

                print('binary_val',binaryVal)
                value = knapsack(binaryVal, w_i, p_i, chromCount)
                profit.append(value)

                binaryString = ""
                chromCount = chromCount + 1

chromCount = 0
chromosomeCount = 0

originalProfit = profit
print(originalProfit)


profit.sort(reverse=True)
print(profit)

b = profit[0]
register_count = 0
j = 2
jString = "" 
#superposition and measurement
for qr in registers:
        cr = classicalregisters[register_count]
        qc = QuantumCircuit(qr, cr)
        qc.h(qr)
        
        # execute the quantum circuit
        
        meas= QuantumCircuit(qr, cr)
        meas.measure(qr, cr)
        
        
        register_count = register_count + 1 
        
        realBackend = IBMQ.backends(name='ibmq_qasm_simulator')[0]
        circ = qc+meas
        result = execute(circ, realBackend, shots=1000).result()
        counts  = result.get_counts(circ)
        job_results.append(counts)
        np.append(resultVisual, result)
        print(counts)
        

knapsackWeight = np.array([])
knapsackProfit = np.array([])
currentWeight = 0
currentProfit = 0
chromCount = 0

print(job_results)

for wX in range(weight_count):
        knapsackProfit = np.append(knapsackProfit, 0)
        knapsackWeight = np.append(knapsackWeight, 0)
        #currentWeight.append(0)
        #currentProfit.append(0)

register_count = 0
profit = []
binaryString = ""
chromosomeCount = 0
weights = 0
p_i = 0
w_i = 0
randInt = 0

for c in range(len(job_results)):
        current_result = job_results[c]
        stringVals = []
        highestVal = 0
          
        DoubleZeroVal = current_result.get('00')
        stringVals.append(DoubleZeroVal)
        ZeroOneVal = current_result.get('01')
        stringVals.append(ZeroOneVal)
        OneZeroVal = current_result.get('10')
        stringVals.append(OneZeroVal)
        DoubleOneVal = current_result.get('11')
        stringVals.append(DoubleOneVal)
        stringVals.sort(reverse=True)
        highestVal = stringVals[0]
        
        print('highestVal', highestVal)

        if highestVal == DoubleOneVal:
                binaryString = binaryString + '11'
        elif highestVal == OneZeroVal:
                binaryString = binaryString + '10'
        elif highestVal == ZeroOneVal:
                binaryString = binaryString + '01'
        elif highestVal == DoubleZeroVal:
                binaryString = binaryString + '00'
                
        
        
        chromosomeCount = chromosomeCount + 1

        print('count',chromosomeCount)

        #evaluate knapsack 
        if chromosomeCount % 2 == 0:
                print(binaryString)
                randInt = random.randint(0, weight_count - 1)

                w_i = weightVals[randInt]
                p_i = profitVals[randInt]

                binaryVal = binaryString
                #binaryVal = format(binaryVal, 'b')

                print('binary_val',binaryVal)
                value = knapsack(binaryVal, w_i, p_i, chromCount)
                profit.append(value)

                binaryString = ""
                chromCount = chromCount + 1

chromCount = 0
chromosomeCount = 0

originalProfit = profit
print(originalProfit)


profit.sort(reverse=True)
print(profit)

b = profit[0]
register_count = 0
j = 2
jString = "" 
#superposition and measurement
for qr in registers:
        cr = classicalregisters[register_count]
        qc = QuantumCircuit(qr, cr)
        qc.h(qr)
        
        # execute the quantum circuit
        
        meas= QuantumCircuit(qr, cr)
        meas.measure(qr, cr)
        
        
        register_count = register_count + 1 
        
        realBackend = IBMQ.backends(name='ibmq_qasm_simulator')[0]
        circ = qc+meas
        result = execute(circ, realBackend, shots=1000).result()
        counts  = result.get_counts(circ)
        job_results.append(counts)
        np.append(resultVisual, result)
        print(counts)
        

knapsackWeight = np.array([])
knapsackProfit = np.array([])
currentWeight = 0
currentProfit = 0
chromCount = 0

print(job_results)

for wX in range(weight_count):
        knapsackProfit = np.append(knapsackProfit, 0)
        knapsackWeight = np.append(knapsackWeight, 0)
        #currentWeight.append(0)
        #currentProfit.append(0)

register_count = 0
profit = []
binaryString = ""
chromosomeCount = 0
weights = 0
p_i = 0
w_i = 0
randInt = 0

for c in range(len(job_results)):
        current_result = job_results[c]
        stringVals = []
        highestVal = 0
          
        DoubleZeroVal = current_result.get('00')
        stringVals.append(DoubleZeroVal)
        ZeroOneVal = current_result.get('01')
        stringVals.append(ZeroOneVal)
        OneZeroVal = current_result.get('10')
        stringVals.append(OneZeroVal)
        DoubleOneVal = current_result.get('11')
        stringVals.append(DoubleOneVal)
        stringVals.sort(reverse=True)
        highestVal = stringVals[0]
        
        print('highestVal', highestVal)

        if highestVal == DoubleOneVal:
                binaryString = binaryString + '11'
        elif highestVal == OneZeroVal:
                binaryString = binaryString + '10'
        elif highestVal == ZeroOneVal:
                binaryString = binaryString + '01'
        elif highestVal == DoubleZeroVal:
                binaryString = binaryString + '00'
                
        
        
        chromosomeCount = chromosomeCount + 1

        print('count',chromosomeCount)

        #evaluate knapsack 
        if chromosomeCount % 2 == 0:
                print(binaryString)
                randInt = random.randint(0, weight_count - 1)

                w_i = weightVals[randInt]
                p_i = profitVals[randInt]

                binaryVal = binaryString
                #binaryVal = format(binaryVal, 'b')

                print('binary_val',binaryVal)
                value = knapsack(binaryVal, w_i, p_i, chromCount)
                profit.append(value)

                binaryString = ""
                chromCount = chromCount + 1

chromCount = 0
chromosomeCount = 0

originalProfit = profit
print(originalProfit)


profit.sort(reverse=True)
print(profit)

b = profit[0]
register_count = 0
j = 2
jString = "" 
#superposition and measurement
for qr in registers:
        cr = classicalregisters[register_count]
        qc = QuantumCircuit(qr, cr)
        qc.h(qr)
        
        # execute the quantum circuit
        
        meas= QuantumCircuit(qr, cr)
        meas.measure(qr, cr)
        
        
        register_count = register_count + 1 
        
        realBackend = IBMQ.backends(name='ibmq_qasm_simulator')[0]
        circ = qc+meas
        result = execute(circ, realBackend, shots=1000).result()
        counts  = result.get_counts(circ)
        job_results.append(counts)
        np.append(resultVisual, result)
        print(counts)
        

knapsackWeight = np.array([])
knapsackProfit = np.array([])
currentWeight = 0
currentProfit = 0
chromCount = 0

print(job_results)

for wX in range(weight_count):
        knapsackProfit = np.append(knapsackProfit, 0)
        knapsackWeight = np.append(knapsackWeight, 0)
        #currentWeight.append(0)
        #currentProfit.append(0)

register_count = 0
profit = []
binaryString = ""
chromosomeCount = 0
weights = 0
p_i = 0
w_i = 0
randInt = 0

for c in range(len(job_results)):
        current_result = job_results[c]
        stringVals = []
        highestVal = 0
          
        DoubleZeroVal = current_result.get('00')
        stringVals.append(DoubleZeroVal)
        ZeroOneVal = current_result.get('01')
        stringVals.append(ZeroOneVal)
        OneZeroVal = current_result.get('10')
        stringVals.append(OneZeroVal)
        DoubleOneVal = current_result.get('11')
        stringVals.append(DoubleOneVal)
        stringVals.sort(reverse=True)
        highestVal = stringVals[0]
        
        print('highestVal', highestVal)

        if highestVal == DoubleOneVal:
                binaryString = binaryString + '11'
        elif highestVal == OneZeroVal:
                binaryString = binaryString + '10'
        elif highestVal == ZeroOneVal:
                binaryString = binaryString + '01'
        elif highestVal == DoubleZeroVal:
                binaryString = binaryString + '00'
                
        
        
        chromosomeCount = chromosomeCount + 1

        print('count',chromosomeCount)

        #evaluate knapsack 
        if chromosomeCount % 2 == 0:
                print(binaryString)
                randInt = random.randint(0, weight_count - 1)

                w_i = weightVals[randInt]
                p_i = profitVals[randInt]

                binaryVal = binaryString
                #binaryVal = format(binaryVal, 'b')

                print('binary_val',binaryVal)
                value = knapsack(binaryVal, w_i, p_i, chromCount)
                profit.append(value)

                binaryString = ""
                chromCount = chromCount + 1

chromCount = 0
chromosomeCount = 0

originalProfit = profit
print(originalProfit)


profit.sort(reverse=True)
print(profit)

b = profit[0]
register_count = 0
j = 2
jString = "" 
#superposition and measurement
for qr in registers:
        cr = classicalregisters[register_count]
        qc = QuantumCircuit(qr, cr)
        qc.h(qr)
        
        # execute the quantum circuit
        
        meas= QuantumCircuit(qr, cr)
        meas.measure(qr, cr)
        
        
        register_count = register_count + 1 
        
        realBackend = IBMQ.backends(name='ibmq_qasm_simulator')[0]
        circ = qc+meas
        result = execute(circ, realBackend, shots=1000).result()
        counts  = result.get_counts(circ)
        job_results.append(counts)
        np.append(resultVisual, result)
        print(counts)
        

knapsackWeight = np.array([])
knapsackProfit = np.array([])
currentWeight = 0
currentProfit = 0
chromCount = 0

print(job_results)

for wX in range(weight_count):
        knapsackProfit = np.append(knapsackProfit, 0)
        knapsackWeight = np.append(knapsackWeight, 0)
        #currentWeight.append(0)
        #currentProfit.append(0)

register_count = 0
profit = []
binaryString = ""
chromosomeCount = 0
weights = 0
p_i = 0
w_i = 0
randInt = 0

for c in range(len(job_results)):
        current_result = job_results[c]
        stringVals = []
        highestVal = 0
          
        DoubleZeroVal = current_result.get('00')
        stringVals.append(DoubleZeroVal)
        ZeroOneVal = current_result.get('01')
        stringVals.append(ZeroOneVal)
        OneZeroVal = current_result.get('10')
        stringVals.append(OneZeroVal)
        DoubleOneVal = current_result.get('11')
        stringVals.append(DoubleOneVal)
        stringVals.sort(reverse=True)
        highestVal = stringVals[0]
        
        print('highestVal', highestVal)

        if highestVal == DoubleOneVal:
                binaryString = binaryString + '11'
        elif highestVal == OneZeroVal:
                binaryString = binaryString + '10'
        elif highestVal == ZeroOneVal:
                binaryString = binaryString + '01'
        elif highestVal == DoubleZeroVal:
                binaryString = binaryString + '00'
                
        
        
        chromosomeCount = chromosomeCount + 1

        print('count',chromosomeCount)

        #evaluate knapsack 
        if chromosomeCount % 2 == 0:
                print(binaryString)
                randInt = random.randint(0, weight_count - 1)

                w_i = weightVals[randInt]
                p_i = profitVals[randInt]

                binaryVal = binaryString
                #binaryVal = format(binaryVal, 'b')

                print('binary_val',binaryVal)
                value = knapsack(binaryVal, w_i, p_i, chromCount)
                profit.append(value)

                binaryString = ""
                chromCount = chromCount + 1

chromCount = 0
chromosomeCount = 0

originalProfit = profit
print(originalProfit)


profit.sort(reverse=True)
print(profit)

b = profit[0]
register_count = 0
j = 2
jString = "" 

# for qr in registers:
#         #determining probability amplitudes
#         cr = classicalregisters[register_count]
#         qc = QuantumCircuit(qr, cr)
#         job_sim = execute(qc, simulator)
#         sim_result = job_sim.result()
#         probability_amplitude = sim_result.get_statevector()

#         DoubleZeroRegister = probability_amplitude[0]
#         ZeroOneRegister = probability_amplitude[1]
#         OneZeroRegister = probability_amplitude[2]
#         DoubleOneRegister = probability_amplitude[3]

#         if register_count%2 == 0:
#                 jString = b[:j]
#         else:
#                 jString = b[j:]
