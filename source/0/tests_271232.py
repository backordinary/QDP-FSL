# https://github.com/claudioalvesmonteiro/quantumML-health/blob/5e5feca1236041219c607e4a87b4117b811403aa/code/quantum_neural_network/tests.py
from quantum_neural_network.QuantumNeuralNetwork import HybridSequentialQuantumNeuralNetwork
from qiskit import  Aer



#=============================
# SF Quantum Network
#=============================

# data 
input_info = [1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1]


###### build model ######

model = HybridSequentialQuantumNeuralNetwork()

model.add_input_layer(4, input_info, 'sf')

model.add_hidden_layer(2, 'sf')


###### feed forward execution ######

simulator = Aer.get_backend('qasm_simulator')

model.execute_FeedForward(simulator, 0.5, 'sf')

#=============================
# HSGS Quantum Network
#=============================

# data 
input_info = [1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1]


###### build model ######

model = HybridSequentialQuantumNeuralNetwork()

model.add_input_layer(4, input_info, 'hsgs')

model.add_hidden_layer(2, 'hsgs')


###### feed forward execution ######

simulator = Aer.get_backend('qasm_simulator')

model.execute_FeedForward(simulator, 0.5, 'hsgs')


#=============================
# Encoding Quantum Network 
#=============================

# data 
input_info = [1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1]


###### build model ######

model = HybridSequentialQuantumNeuralNetwork()

model.add_input_layer(4, input_info, 'encoding-weight')

model.add_hidden_layer(2, 'encoding-weight')


###### feed forward execution ######

simulator = Aer.get_backend('qasm_simulator')

model.execute_FeedForward(simulator, 0.5, 'encoding-weight')
