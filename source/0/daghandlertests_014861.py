# https://github.com/KernalPanik/QC_Optimizer/blob/e49775ec39526568da6f543d4e1f31d24afcbcd8/Scheduler/DagHandlerTests.py
import unittest
from DagHandler import dag_to_list, check_if_interchangeable, divide_into_subdags, sort_subdag
from qiskit.dagcircuit import DAGCircuit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.converters import circuit_to_dag

class DagHandlerTests(unittest.TestCase):
    def test_dag_to_list_given_dag_returns_adj_list(self):
        q = QuantumRegister(2, 'q')
        c = ClassicalRegister(2, 'c')
        circuit = QuantumCircuit(q, c)
        circuit.x(q[0])
        circuit.h(q[1])
        circuit.x(q[1])

        dag = circuit_to_dag(circuit)
        adj_list = dag_to_list(dag)
        success = True
        if(adj_list[0] != "x_q0_" or adj_list[1] != "h_q1_" or adj_list[2] != "x_q1_"):
            success = False

        self.assertTrue(success)


    def test_interchange_test_h0_h0_should_interchange(self):
        self.assertTrue(check_if_interchangeable("h_q0_", "h_q0_"))

    def test_interchange_test_x1_cx10_should_not_interchange(self):
        self.assertFalse(check_if_interchangeable("x_q1_", "cx_q1_q0_"))

    def test_interchage_test_x1_cx01_should_interchange(self):
        self.assertTrue(check_if_interchangeable("x_q1_", "cx_q0_q1_"))

    def test_interchange_test_x1_h1_should_not_interchage(self):
        self.assertFalse(check_if_interchangeable("x_q1_", "h_q1_"))

    def test_interchange_test_x1_x0_should_interchange(self):
        self.assertTrue(check_if_interchangeable("x_q1_", "h_q0_"))

    def test_divide_into_subdags_test_returns_correct_amount_of_subdags(self):
        q = QuantumRegister(2, 'q')
        c = ClassicalRegister(2, 'c')
        circ = QuantumCircuit(q, c)
        circ.x(q[0])
        circ.x(q[0])
        circ.cx(q[1], q[0])
        circ.x(q[0])
        circ.cx(q[0], q[1])
        circ.cx(q[1], q[0])
        circ.x(q[0])
        circ.cx(q[0], q[1])
        circ.x(q[1])
        circ.x(q[1])
        dag = circuit_to_dag(circ)
        adj_list = dag_to_list(dag)
        subdags, cxdir = divide_into_subdags(adj_list)
        self.assertEqual(len(subdags), 3)

    def test_divide_into_subdags_test_hadamard_subdag_generated(self):
        q = QuantumRegister(3, 'q')
        c = ClassicalRegister(3, 'c')
        circ = QuantumCircuit(q, c)
        circ.h(q[0])
        circ.x(q[1])
        circ.x(q[2])
        circ.cx(q[0], q[1])
        circ.h(q[0])

        dag = circuit_to_dag(circ)
        adj_list = dag_to_list(dag)
        subdags, cxdir = divide_into_subdags(adj_list)
        success = ((subdags[0][0] == "h_q0_") and (subdags[0][-1] == "h_q0_"))
        hm = subdags[0][0]
        hmm = subdags[0][-1]
        self.assertTrue(success)

    def test_sort_subdag_test_given_unsorted_returns_sorted(self):
        q = QuantumRegister(2, 'q')
        c = ClassicalRegister(2, 'c')
        circuit = QuantumCircuit(q, c)
        circuit.x(q[0])
        circuit.cx(q[1], q[0])
        circuit.x(q[0])
        circuit.cx(q[1], [0])

        dag = circuit_to_dag(circuit)
        adj_list = dag_to_list(dag)
        sorted_list = sort_subdag(adj_list)
        success = True
        expected_list = ["x_q0_", "x_q0_", "cx_q1_q0_", "cx_q1_q0_"]
        self.assertEqual(sorted_list, expected_list)

if __name__ == '__main__':
    unittest.main()