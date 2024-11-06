# https://github.com/naomisolomons/qflow/blob/be3a1aa29d926d1a7e49aa546567b60df7dad286/Graph_Constructor.py
import numpy as np
import networkx as nx

#This module takes a level unitary and initial state and formats a networkx 
#graph object with the required data and then converts this object into a JSON string

class Graph_Wrapper:
    
    def __init__(self, level_operator, input_state):
        
        self.level_operator = level_operator
        self.input_StateVector = input_state
        self.state_vector = input_state.data
        self.unitary = level_operator.data
        self.support = self.support(self.unitary)    
        self.prob_flows = self.probability_flows()
        
        self.graph = nx.DiGraph(self.support) #This defines as Directed Graph object this is structred as a dict of dicts
        #self.mapping = {node:"{0:b}".format(node) for node in self.graph.nodes()} #This defines a maps the node lables to there representation in the computational basis
        self.values = nx.circular_layout(self.graph,220) #This defines an array of x,y coridinates that will give the graph a circualr structure when drawn
        
        for source, target in self.graph.edges: # assigns the weight of each edge the corresponding value of the flow matrix
            self.graph[source][target]['weight'] = self.prob_flows[source,target]       
        
        for nodes in self.graph.nodes: #Assigns each node a x position and y position 
            self.graph.nodes[nodes]['x_pos'] = self.values[nodes][0]
            self.graph.nodes[nodes]['y_pos'] = self.values[nodes][1]
            
        #self.graph = nx.relabel_nodes(self.graph,self.mapping,False) #This applies the earlier defined map to the nodes of the graph.
        self.data = nx.node_link_data(self.graph,{'link': "edges", 'source': "from", 'target': "to", 'nodes':"nodes", 'id':"id"}) #This assigns the graph data to the variable data
        
        
        
    def support(self,Matrix):   # This method constructs the support matrix for a given input matrix. This is then used as the adjacency matrix of the graph.
        support = np.zeros((Matrix.shape), dtype='int32')
        for i in range(Matrix.shape[0]):
            for j in range(Matrix.shape[1]):
                if Matrix[i,j] != 0:
                    support[i,j] = 1
                else:
                        support[i,j] = support[i,j]
        return support
    
    def probability_flows(self):#This method calculates the probability flows for each edge of the graph.
        flow_matrix = np.zeros((self.unitary.shape))
        
        for i in range(self.unitary.shape[0]):
            for j in range(self.unitary.shape[1]):
                A = 0
                for k in range(self.unitary.shape[0]):
                    A += self.state_vector[k]*self.unitary[i,k]
                flow_matrix[i,j] = (((self.state_vector[j]*self.unitary[i,j]).conjugate())*A).real
        
        return flow_matrix
    
    

if __name__== '__main__': #Test Code
    from Qflow_Level_Builder import stage
    from qiskit import QuantumCircuit
    
    
    qs = QuantumCircuit(2)
    qs.h((0,1))
    qs.cx(1,0)
    qs.x(0)
    qs.y(0)

    Stage_One = stage(qs)
    G_wrapper = Graph_Wrapper(Stage_One.levelOperator(1),Stage_One.levelState(1))
    a = G_wrapper.output_JSON()
    
    

