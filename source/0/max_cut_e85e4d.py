# https://github.com/Arkonaire/QAOA-Combinatorics/blob/84ea91ef71076b1ff22cf6d0ec1484619c2b5ac6/max_cut.py
import networkx as nx
import matplotlib.pyplot as plt

from qaoa_engine import QAOA
from networkx.drawing.layout import spring_layout
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


class MaxCut(QAOA):

    """QAOA implementation for Max Cut."""

    def __init__(self, V, E, backend=None):

        """
        Build input graph and begin QAOA
        Args:
            V: Vertices of input graph as a list.
            E: Edges of input graph as a dictionary with weights or a list if unweighted.
            backend: Custom backend. Can be used for noisy simulations
        """

        # Set up vertices and edges
        self.vertices = V
        if isinstance(E, dict):
            self.edges = list(E.keys())
            self.weights = E
        else:
            self.edges = E
            self.weights = {edge: 1 for edge in self.edges}
        self.weights = {key: value/sum(self.weights.values()) for key, value in self.weights.items()}

        # Build input graph
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.vertices)
        self.graph.add_edges_from(self.edges)

        # Begin QAOA
        super().__init__(len(V), p=6, backend=backend)

    def cost_function(self, z):

        """
        Max Cut cost function.
        Args:
            z: An integer or bitstring whose cost is to be determined.
        Return:
            Cost function as integer.
        """

        # Convert to bitstr
        if not isinstance(z, str):
            z = format(z, '0' + str(self.n) + 'b')
        z: str

        # Evaluate C(z)
        cost = 0
        for edge in self.edges:
            if z[edge[0]] != z[edge[1]]:
                cost -= self.weights[edge]
        return cost

    def build_cost_ckt(self):

        """
        Max Cut cost circuit. Override in child class.
        Return:
            QuantumCircuit. Parameterized cost circuit layer.
        """

        # Build cost circuit
        circ = QuantumCircuit(self.n, name='$U(H_C,\\gamma)$')
        param = Parameter('param_c')
        for edge in self.edges:
            circ.cp(-2*param*self.weights[edge], edge[0], edge[1])
            circ.p(param*self.weights[edge], edge[0])
            circ.p(param*self.weights[edge], edge[1])
        return circ

    def visualize_output(self):

        """
        Visualize Max Cut output post QAOA optimization.
        """

        # Sample output
        z, avg_cost = self.sample(vis=True)
        print('Sampled Output: ' + str(z))
        print('Minimum Cost: ' + str(self.cost_function(z)))
        print('Expectation Value: ' + str(avg_cost))

        # Extract colormap
        color_map = []
        for i in range(len(self.graph.nodes)):
            if z[i] == '0':
                color_map.append('blue')
            else:
                color_map.append('red')

        # Extract cuts
        cuts = []
        for e in self.graph.edges:
            if z[e[0]] == z[e[1]]:
                cuts.append('solid')
            else:
                cuts.append('dashed')

        # Draw input graph
        fig = plt.figure(figsize=(10, 5))
        fig.suptitle('Max Cut')
        plt.subplot(121)
        ax = plt.gca()
        ax.set_title('Input Graph')
        pos = spring_layout(self.graph)
        nx.draw(self.graph, with_labels=True, node_color='lightgreen', edge_color='lightblue',
                style='solid', width=2, ax=ax, pos=pos, font_size=8, font_weight='bold')

        # Draw output graph
        plt.subplot(122)
        ax = plt.gca()
        ax.set_title('Output Graph')
        nx.draw(self.graph, with_labels=True, node_color=color_map, edge_color='green',
                style=cuts, width=2, ax=ax, pos=pos, font_size=8, font_weight='bold')
        plt.show()


if __name__ == '__main__':

    # Test code
    V = list(range(7))
    E = [(0, 5), (0, 2), (1, 2), (1, 3), (2, 5), (2, 6), (3, 5), (3, 4), (4, 5), (4, 6)]
    W = [2, 3, 5, 1, 2, 3, 6, 2, 5, 3]
    obj = MaxCut(V, {E[i]: W[i] for i in range(len(E))})
    obj.visualize_output()
