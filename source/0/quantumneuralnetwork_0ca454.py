# https://github.com/claudioalvesmonteiro/quantumML-health/blob/5e5feca1236041219c607e4a87b4117b811403aa/code/quantum_neural_network/QuantumNeuralNetwork.py
from neuron import createNeuron, executeNeuron
from qiskit import execute, Aer, QuantumRegister, QuantumCircuit
import random

simulator = Aer.get_backend('qasm_simulator')

def init_random_weights(n, weights_generator_operator):
    ''' generates random list of n numbers,
        based on weights_generator_operator method
    '''

    weights =[]

    if weights_generator_operator == 'hsgs' or  weights_generator_operator == 'sf':
        for i in range(n):
            weight = random.randint(-1,1)
            while weight == 0:
                weight = random.randint(-1,1)
            weights.append(weight)

    elif weights_generator_operator == 'encoding-weight': 
        for i in range(n):
            weight = random.random()
            weights.append(weight)

    else: 
        print('weights_generator_operator '+weights_generator_operator+' not available')
    return weights


class Neuron():

    def __init__(self, input_data, weights_generator_operator):
        
        self.WEIGHTS = init_random_weights(len(input_data), weights_generator_operator)
        self.CIRCUIT = createNeuron(input_data, self.WEIGHTS, weights_generator_operator)

    def execute_neuron(self, simulator, threshold, mode='nshots'):
        ''' execute neuron circuit and returns output
        '''

        if mode == 'nshots':
            nshots = 1024
            job = execute(self.CIRCUIT , backend=simulator, shots=nshots)
            result = job.result()


            try:
                count = result.get_counts()['1']; print(result.get_counts())
            except:
                count = 0; print(result.get_counts())
                

            if count/nshots >= threshold:
                output = -1 ## neuronio ativado
            else:
                output = 1  ## neuronio nao ativado
        return output


    def insert_input(self, input_data, weights_generator_operator):
        self.CIRCUIT = createNeuron(input_data, self.WEIGHTS, weights_generator_operator) 


    def update_weights(self, weights):
        self.WEIGHTS = weights



class HybridSequentialQuantumNeuralNetwork():

    def __init__(self):
    
        self.INPUT_LAYER = []
        self.HIDDEN_LAYERS = []
        self.HYBRID_OUTPUT_OF_LAYER = []


    def add_input_layer(self, n_neurons, input_data, weights_generator_operator):
        ''' generate input layer of neurons
        '''

        for i in range(n_neurons):
            print('add input neuron')
            self.INPUT_LAYER.append(Neuron(input_data, weights_generator_operator))     


    def add_hidden_layer(self, n_neurons, weights_generator_operator):
        ''' define architecture of hidden layer to be added
            if hidden layer exists, capture number of neurons in the last one to
            generate initial inputs and weights of neurons,
            otherwise use output config from input layer
        ''' 

        hidden_layer = []

        if self.HIDDEN_LAYERS:
            previous_layer_n_neurons = len(self.HIDDEN_LAYERS[len(self.HIDDEN_LAYERS)])
        else:
            previous_layer_n_neurons = len(self.INPUT_LAYER)

        for i in range(n_neurons):
            init_random_input = init_random_weights(previous_layer_n_neurons, weights_generator_operator)
            hidden_layer.append(Neuron(init_random_input, weights_generator_operator)) 
        
        self.HIDDEN_LAYERS.append(hidden_layer)


    def execute_FeedForward(self, simulator, threshold, weights_generator_operator):
        ''' capture results from each neuron
            starting from input layer and passing to next layer (feedforward)
            return output of last layer
        '''

        for neuron in self.INPUT_LAYER:
            self.HYBRID_OUTPUT_OF_LAYER.append(neuron.execute_neuron(simulator, threshold))

        for layer_id in range(len(self.HIDDEN_LAYERS)):
            layer_output = []

            for neuron in self.HIDDEN_LAYERS[layer_id]:

                neuron.insert_input(self.HYBRID_OUTPUT_OF_LAYER, weights_generator_operator)
                output = neuron.execute_neuron(simulator, threshold)
                layer_output.append(output)

            self.HYBRID_OUTPUT_OF_LAYER = layer_output

        return self.HYBRID_OUTPUT_OF_LAYER

