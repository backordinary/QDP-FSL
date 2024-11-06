# https://github.com/zommiommy/quantum_research/blob/6a345e02894140a08f8c2b1d87bab89ff48b5605/Minimum_Spanning_Tree/qiskit_implementation/edges_encoding/graph_test.py

import qiskit
import qiskit_aqua
import qiskit as q

from edge_finder import EdgeFinder
from edge_finder import encode_graph
from edge_finder import select_value
from edge_finder import get_logger

logger = get_logger(__name__)


class GraphTest(EdgeFinder):

    def setup_oracle(self):
        self.oracle  = q.QuantumCircuit(*self.qregisters)

        self.oracle.barrier()

        self.oracle = encode_graph(self.oracle, self.graph, self.start, self.end, self.flags[0], self.ancillas)

        self.oracle.barrier()

        self.oracle.z(self.flags[0])


if __name__ == "__main__":
    t = GraphTest()
    edge = t.run()
    logger.info("Result: {}".format(edge))