# https://github.com/akotil/qcircuit-optimization/blob/fa62344a80741aec9a94f001b15c57f5d8423dcf/tests.py
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit

import optimizations


class TestEquivalences(unittest.TestCase):

    # H ⊗ H - CNOT - H ⊗ H
    def test_cnot_transformation(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        qc.cx(0, 1)
        qc.h(0)
        qc.h(1)

        qc.rz(np.pi, 0)
        qc.rz(np.pi, 1)

        qc.h(0)
        qc.h(1)
        qc.cx(0, 1)
        qc.h(0)
        qc.h(1)

        dag = circuit_to_dag(qc)

        # Reference optimized quantum circuit
        qc_ref = QuantumCircuit(2)
        qc_ref.cx(1, 0)
        qc_ref.rz(np.pi, 0)
        qc_ref.rz(np.pi, 1)
        qc_ref.cx(1, 0)

        # Apply the optimization procedure
        reduction = optimizations.HGateReduction(dag)
        dag = reduction.apply()
        qc_optimized = dag_to_circuit(dag)

        # The circuit should not change
        assert qc_ref == qc_optimized

    # S gate transformation
    def test_p_transformation_1(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.s(0)
        qc.h(0)
        qc.rz(np.pi, 0)
        qc.h(0)
        qc.s(0)
        qc.h(0)
        dag = circuit_to_dag(qc)

        # Reference optimized quantum circuit
        qc_ref = QuantumCircuit(1)
        qc_ref.sdg(0)
        qc_ref.h(0)
        qc_ref.sdg(0)
        qc_ref.rz(np.pi, 0)
        qc_ref.sdg(0)
        qc_ref.h(0)
        qc_ref.sdg(0)

        # Apply the optimization procedure
        reduction = optimizations.HGateReduction(dag)
        dag = reduction.apply()
        qc_optimized = dag_to_circuit(dag)

        # The circuit should not change
        assert qc_ref == qc_optimized

    # S dagger gate transformation
    def test_p_transformation_2(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.sdg(0)
        qc.h(0)
        dag = circuit_to_dag(qc)

        # Reference optimized quantum circuit
        qc_ref = QuantumCircuit(1)
        qc_ref.s(0)
        qc_ref.h(0)
        qc_ref.s(0)

        # Apply the optimization procedure
        reduction = optimizations.HGateReduction(dag)
        dag = reduction.apply()
        qc_optimized = dag_to_circuit(dag)

        # The circuit should not change
        assert qc_ref == qc_optimized

    # H - P - CNOT - P_dagger - H = P_dagger - CNOT - P
    def test_p_transformation_3(self):
        qc = QuantumCircuit(2)
        qc.h(1)
        qc.s(1)
        qc.cx(0, 1)
        qc.sdg(1)
        qc.h(1)
        dag = circuit_to_dag(qc)

        # Reference optimized quantum circuit
        qc_ref = QuantumCircuit(2)
        qc_ref.sdg(1)
        qc_ref.cx(0, 1)
        qc_ref.s(1)

        # Apply the optimization procedure
        reduction = optimizations.HGateReduction(dag)
        dag = reduction.apply()
        qc_optimized = dag_to_circuit(dag)

        # The circuit should not change
        assert qc_ref == qc_optimized

    # H - P_dagger - CNOT - P - H = P - CNOT - P_dagger
    def test_p_transformation_4(self):
        qc = QuantumCircuit(2)
        qc.h(1)
        qc.sdg(1)
        qc.cx(0, 1)
        qc.s(1)
        qc.h(1)
        dag = circuit_to_dag(qc)

        # Reference optimized quantum circuit
        qc_ref = QuantumCircuit(2)
        qc_ref.s(1)
        qc_ref.cx(0, 1)
        qc_ref.sdg(1)

        # Apply the optimization procedure
        reduction = optimizations.HGateReduction(dag)
        dag = reduction.apply()
        qc_optimized = dag_to_circuit(dag)

        # The circuit should not change
        assert qc_ref == qc_optimized

    def test_rz_commutation_1(self):
        qc = QuantumCircuit(3)
        qc.rz(np.pi, 0)
        qc.h(0)
        qc.cnot(2, 0)
        qc.h(0)
        qc.x(0)
        qc.rz(np.pi, 0)
        qc.h(0)
        qc.cnot(2, 0)
        qc.h(0)
        qc.rz(np.pi, 0)
        qc.cnot(0, 2)
        qc.rz(np.pi, 0)
        print(qc.draw())
        dag = circuit_to_dag(qc)

        # Reference optimized quantum circuit
        qc_ref = QuantumCircuit(3)
        qc_ref.rz(np.pi, 0)
        qc_ref.h(0)
        qc_ref.cnot(2, 0)
        qc_ref.h(0)
        qc_ref.x(0)
        qc_ref.h(0)
        qc_ref.cnot(2, 0)
        qc_ref.h(0)
        qc_ref.rz(2 * np.pi, 0)
        qc_ref.cnot(0, 2)
        qc_ref.rz(np.pi, 0)

        # Apply the optimization procedure
        reduction = optimizations.RzReduction(dag)
        dag = reduction.apply()
        qc_optimized = dag_to_circuit(dag)

        assert qc_ref == qc_optimized

    def test_rz_commutation_2(self):
        qc = QuantumCircuit(2)
        qc.rz(np.pi, 1)
        qc.cnot(0, 1)
        qc.rz(-np.pi, 1)
        qc.cnot(0, 1)
        qc.rz(np.pi, 1)
        dag = circuit_to_dag(qc)

        # Reference optimized quantum circuit
        qc_ref = QuantumCircuit(2)
        qc_ref.cnot(0, 1)
        qc_ref.rz(-np.pi, 1)
        qc_ref.cnot(0, 1)
        qc_ref.rz(2 * np.pi, 1)

        # Apply the optimization procedure
        reduction = optimizations.RzReduction(dag)
        dag = reduction.apply()
        qc_optimized = dag_to_circuit(dag)

        assert qc_ref == qc_optimized

    def test_cx_commutation_1(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 2)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.cx(0, 2)
        dag = circuit_to_dag(qc)

        # Reference optimized quantum circuit
        qc_ref = QuantumCircuit(3)
        qc_ref.cx(1, 2)
        qc_ref.cx(0, 1)

        # Apply the optimization procedure
        reduction = optimizations.CxReduction(dag)
        dag = reduction.apply()
        qc_optimized = dag_to_circuit(dag)

        assert qc_ref == qc_optimized

    def test_cx_commutation_2(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 2)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.h(2)
        qc.cx(0, 2)
        dag = circuit_to_dag(qc)

        # Apply the optimization procedure
        reduction = optimizations.CxReduction(dag)
        dag = reduction.apply()
        qc_optimized = dag_to_circuit(dag)

        # The circuit should not change
        assert qc == qc_optimized

    def test_cx_commutation_3(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.h(1)
        qc.cx(1, 2)
        qc.h(1)
        qc.cx(0, 2)
        qc.cx(0, 1)
        qc.h(0)
        dag = circuit_to_dag(qc)

        # Reference optimized quantum circuit
        qc_ref = QuantumCircuit(3)
        qc_ref.h(1)
        qc_ref.cx(1, 2)
        qc_ref.h(1)
        qc_ref.cx(0, 2)
        qc_ref.h(0)

        # Apply the optimization procedure
        reduction = optimizations.CxReduction(dag)
        dag = reduction.apply()
        qc_optimized = dag_to_circuit(dag)

        assert qc_ref == qc_optimized
