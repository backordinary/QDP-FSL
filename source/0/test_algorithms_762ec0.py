# https://github.com/UST-QuAntiL/quantum-circuit-generator/blob/c04b73046df35bacc54b1b810124a0f1b331b202/tests/test_algorithms.py
import unittest
import os, sys
import json
import contextlib
import re
import numpy as np
from qiskit import QuantumCircuit
from builtins import isinstance

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from app import create_app


class FlaskClientTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app("testing")
        self.app_context = self.app.app_context()
        self.app_context.push()
        self.client = self.app.test_client(use_cookies=True)

    def tearDown(self):
        self.app_context.pop()

    def test_hhl_algorithm(self):
        # Test errors
        response = self.client.post(
            "/algorithms/hhl",
            data=json.dumps(
                {"matrix": [[1.5, 0.5, 1], [0.5, 1.5, 1]], "vector": [0, 1]}
            ),
            content_type="application/json",
        )
        self.assertEqual(
            "Invalid matrix input! Matrix must be square.",
            response.get_json().get("message"),
        )
        self.assertEqual(response.status_code, 400)

        response = self.client.post(
            "/algorithms/hhl",
            data=json.dumps({"matrix": [[1.5, 0.8], [0.5, 1.5]], "vector": [0, 1]}),
            content_type="application/json",
        )
        self.assertEquals(
            "Invalid matrix input! Matrix must be hermitian.",
            response.get_json().get("message"),
        )
        self.assertEqual(response.status_code, 400)

        response = self.client.post(
            "/algorithms/hhl",
            data=json.dumps({"matrix": [[1.5, 0.5], [0.5, 1.5]], "vector": [0, 1, 2]}),
            content_type="application/json",
        )
        self.assertTrue(
            "Invalid matrix, vector input! Matrix and vector must be of the same dimension."
            in response.get_json().get("message")
        )
        self.assertEqual(response.status_code, 400)

        response = self.client.post(
            "/algorithms/hhl",
            data=json.dumps(
                {
                    "matrix": [[1.5, 0.5, 1.0], [0.5, 1.5, 1.0], [1.0, 1.0, 0.5]],
                    "vector": [0, 1, 2],
                }
            ),
            content_type="application/json",
        )
        self.assertTrue(
            "Invalid matrix input! Input matrix dimension must be 2^n."
            in response.get_json().get("message")
        )
        self.assertEqual(response.status_code, 400)

        # Test different matrix sizes
        response = self.client.post(
            "/algorithms/hhl",
            data=json.dumps({"matrix": [[1.5, 0.5], [0.5, 1.5]], "vector": [0, 1]}),
            content_type="application/json",
        )
        self.assertEqual(5, response.get_json().get("n_qubits"))
        self.assertEqual(4, response.get_json().get("depth"))
        match = re.search(
            "amplitude_enc q.*;\nqpe q.*,q.*,q.*;\ninvx q.*,q.*,q.*;\nqpe_dg q.*,q.*,q.*;\n",
            response.get_json().get("circuit"),
        )
        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

        response = self.client.post(
            "/algorithms/hhl",
            data=json.dumps(
                {
                    "matrix": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "vector": [0, 1, 0, 0],
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(7, response.get_json().get("n_qubits"))
        self.assertEqual(4, response.get_json().get("depth"))
        match = re.search(
            "amplitude_enc q.*,q.*;\nqpe q.*,q.*,q.*,q.*,q.*;\ninvx q.*,q.*,q.*,q.*;\nqpe_dg q.*,q.*,q.*,q.*,q.*;\n",
            response.get_json().get("circuit"),
        )
        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

    def test_qaoa_algorithm(self):
        # Test errors
        # invalid initial state
        # suppress message: Error near line ...
        with contextlib.redirect_stdout(None):
            response = self.client.post(
                "/algorithms/qaoa",
                data=json.dumps(
                    {
                        "initial_state": 'OPENQASM 2.0; +++++ \ninclude "qelib1.inc";\nqreg q[3];\nx q[0];\nx q[1];\nx q[2];',
                        "pauli_op_string": "0.5 * ((I^Z^Z) + (Z^I^Z) + (Z^Z^I))",
                        "reps": 2,
                        "gammas": [1.0, 1.2],
                        "betas": [0.4, 0.7],
                    }
                ),
                content_type="application/json",
            )
        self.assertTrue(
            "Invalid initial_state: \"Expected an ID, received '+'\""
            in response.get_json().get("message")
        )
        self.assertEqual(response.status_code, 400)

        # invalid Pauli string
        response = self.client.post(
            "/algorithms/qaoa",
            data=json.dumps(
                {
                    "pauli_op_string": "0.5 * ((I^Z^Z) + (Z^I^Z) + (Z^Z))",
                    "reps": 2,
                    "gammas": [1.0, 1.2],
                    "betas": [0.4, 0.7],
                }
            ),
            content_type="application/json",
        )
        self.assertTrue(
            "Invalid pauli_op_string: Sum of operators with different numbers of qubits, 3 and 2, is not well defined"
            in response.get_json().get("message")
        )
        self.assertEqual(response.status_code, 400)

        # invalid mixer (pauli operator String)
        response = self.client.post(
            "/algorithms/qaoa",
            data=json.dumps(
                {
                    "pauli_op_string": "0.5 * ((I^Z^Z) + (Z^I^Z) + (Z^Z^I))",
                    "mixer": "(Y^I^I) + (I^Y^I) + (I^Y)",
                    "reps": 2,
                    "gammas": [1.0, 1.2],
                    "betas": [0.4, 0.7],
                }
            ),
            content_type="application/json",
        )
        self.assertTrue(
            "Invalid mixer: Sum of operators with different numbers of qubits, 3 and 2, is not well defined"
            in response.get_json().get("message")
        )
        self.assertEqual(response.status_code, 400)

        # invalid mixer (qasm String)
        # suppress message: Error near line ...
        with contextlib.redirect_stdout(None):
            response = self.client.post(
                "/algorithms/qaoa",
                data=json.dumps(
                    {
                        "pauli_op_string": "0.5 * ((I^Z^Z) + (Z^I^Z) + (Z^Z^I))",
                        "mixer": 'OPENQASM 2.0; +++++ \ninclude "qelib1.inc";\nqreg q[3];\nrx(pi/2) q[0];\nry(pi/2) q[1];\nrz(pi/2) q[2];',
                        "reps": 2,
                        "gammas": [1.0, 1.2],
                    }
                ),
                content_type="application/json",
            )
        self.assertTrue(
            "Invalid mixer: \"Expected an ID, received '+'\""
            in response.get_json().get("message")
        )
        self.assertEqual(response.status_code, 400)

        # invalid reps
        response = self.client.post(
            "/algorithms/qaoa",
            data=json.dumps(
                {
                    "pauli_op_string": "0.5 * ((I^Z^Z) + (Z^I^Z) + (Z^Z^I))",
                    "reps": 3,
                    "gammas": [1.0, 1.2],
                    "betas": [0.4, 0.7],
                }
            ),
            content_type="application/json",
        )
        self.assertTrue(
            "Number of angles and repetitions don't match. You specified 2 gamma(s) and 2 beta(s) for 3 repetition(s)."
            in response.get_json().get("message")
        )
        self.assertEqual(response.status_code, 400)

        # invalid gammas length
        response = self.client.post(
            "/algorithms/qaoa",
            data=json.dumps(
                {
                    "pauli_op_string": "0.5 * ((I^Z^Z) + (Z^I^Z) + (Z^Z^I))",
                    "reps": 2,
                    "gammas": [1.0, 1.2, 1.1],
                    "betas": [0.4, 0.7],
                }
            ),
            content_type="application/json",
        )
        self.assertTrue(
            "Number of angles and repetitions don't match. You specified 3 gamma(s) and 2 beta(s) for 2 repetition(s)."
            in response.get_json().get("message")
        )
        self.assertEqual(response.status_code, 400)

        # invalid betas length
        response = self.client.post(
            "/algorithms/qaoa",
            data=json.dumps(
                {
                    "pauli_op_string": "0.5 * ((I^Z^Z) + (Z^I^Z) + (Z^Z^I))",
                    "reps": 2,
                    "gammas": [1.0, 1.2],
                    "betas": [0.4, 0.7, 0.8],
                }
            ),
            content_type="application/json",
        )
        self.assertTrue(
            "Number of angles and repetitions don't match. You specified 2 gamma(s) and 3 beta(s) for 2 repetition(s)."
            in response.get_json().get("message")
        )
        self.assertEqual(response.status_code, 400)

        # good response
        # test 4 qbit QAOA with reps = 1
        response = self.client.post(
            "/algorithms/qaoa",
            data=json.dumps(
                {
                    "pauli_op_string": "0.5 * ((I^I^Z^Z) + (I^Z^I^Z) + (I^Z^Z^I) + (Z^I^Z^I) + (Z^Z^I^I))",
                    "reps": 1,
                    "gammas": [1.0],
                    "betas": [1.0],
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(4, response.get_json().get("n_qubits"))
        match = re.search(
            "\nqreg q.*;\nh q.*;\nh q.*;\nrzz\(1.0\) q.*,q.*;\nh q.*;\nrzz\(1.0\) q.*,q.*;\nrx\(2.0\) q.*;\nrzz\(1.0\) q.*,q.*;\nh q.*;\nrzz\(1.0\) q.*,q.*;\nrx\(2.0\) q.*;\nrzz\(1.0\) q.*,q.*;\nrx\(2.0\) q.*;\nrx\(2.0\) q.*;\n",
            response.get_json().get("circuit"),
        )
        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

        # test 3 qbit QAOA with reps = 2
        response = self.client.post(
            "/algorithms/qaoa",
            data=json.dumps(
                {
                    "pauli_op_string": "0.5 * ((I^Z^Z) + (Z^I^Z) + (Z^Z^I))",
                    "reps": 2,
                    "gammas": [1.0, 1.2],
                    "betas": [0.4, 0.7],
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(3, response.get_json().get("n_qubits"))
        match = re.search(
            "\nqreg q.*;\nh q.*;\nh q.*;\nrzz\(1.0\) q.*,q.*;\nh q.*;\nrzz\(1.0\) q.*,q.*;\nrx\(0.8\) q.*;\nrzz\(1.0\) q.*,q.*;\nrx\(0.8\) q.*;\nrzz\(1.2\) q.*,q.*;\nrx\(0.8\) q.*;\nrzz\(1.2\) q.*,q.*;\nrx\(1.4\) q.*;\nrzz\(1.2\) q.*,q.*;\nrx\(1.4\) q.*;\nrx\(1.4\) q.*;\n",
            response.get_json().get("circuit"),
        )
        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

        # test 3 qbit QAOA with initial_state
        response = self.client.post(
            "/algorithms/qaoa",
            data=json.dumps(
                {
                    "initial_state": 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\nx q[0];\nx q[1];\nx q[2];',
                    "pauli_op_string": "0.5 * ((I^Z^Z) + (Z^I^Z) + (Z^Z^I))",
                    "reps": 2,
                    "gammas": [1.0, 1.2],
                    "betas": [0.4, 0.7],
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(3, response.get_json().get("n_qubits"))
        match = re.search(
            "\nqreg q.*;\nx q.*;\nx q.*;\nrzz\(1.0\) q.*,q.*;\nx q.*;\nrzz\(1.0\) q.*,q.*;\nrx\(0.8\) q.*;\nrzz\(1.0\) q.*,q.*;\nrx\(0.8\) q.*;\nrzz\(1.2\) q.*,q.*;\nrx\(0.8\) q.*;\nrzz\(1.2\) q.*,q.*;\nrx\(1.4\) q.*;\nrzz\(1.2\) q.*,q.*;\nrx\(1.4\) q.*;\nrx\(1.4\) q.*;\n",
            response.get_json().get("circuit"),
        )
        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

        # test 3 qbit QAOA with custom mixer (pauli operator String)
        response = self.client.post(
            "/algorithms/qaoa",
            data=json.dumps(
                {
                    "pauli_op_string": "0.5 * ((I^Z^Z) + (Z^I^Z) + (Z^Z^I))",
                    "mixer": "(Y^I^I) + (I^Y^I) + (I^I^Y)",
                    "reps": 2,
                    "gammas": [1.0, 1.2],
                    "betas": [0.4, 0.7],
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(3, response.get_json().get("n_qubits"))
        match = re.search(
            "qreg q.*;\nh q.*;\nh q.*;\nrzz\(1.0\) q.*,q.*;\nh q.*;\nrzz\(1.0\) q.*,q.*;\nry\(0.8\) q.*;\nrzz\(1.0\) q.*,q.*;\nry\(0.8\) q.*;\nrzz\(1.2\) q.*,q.*;\nry\(0.8\) q.*;\nrzz\(1.2\) q.*,q.*;\nry\(1.4\) q.*;\nrzz\(1.2\) q.*,q.*;\nry\(1.4\) q.*;\nry\(1.4\) q.*;\n",
            response.get_json().get("circuit"),
        )
        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

        # test 3 qbit QAOA with custom mixer (qasm String)
        response = self.client.post(
            "/algorithms/qaoa",
            data=json.dumps(
                {
                    "pauli_op_string": "0.5 * ((I^Z^Z) + (Z^I^Z) + (Z^Z^I))",
                    "mixer": 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\nrx(pi/2) q[0];\nry(pi/2) q[1];\nrz(pi/2) q[2];',
                    "reps": 2,
                    "gammas": [1.0, 1.2],
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(3, response.get_json().get("n_qubits"))
        match = re.search(
            "\nqreg q.*;\nh q.*;\nh q.*;\nrzz\(1.0\) q.*,q.*;\nh q.*;\nrzz\(1.0\) q.*,q.*;\nrx\(pi/2\) q.*;\nrzz\(1.0\) q.*,q.*;\nry\(pi/2\) q.*;\nrzz\(1.2\) q.*,q.*;\nrz\(pi/2\) q.*;\nrzz\(1.2\) q.*,q.*;\nrx\(pi/2\) q.*;\nrzz\(1.2\) q.*,q.*;\nry\(pi/2\) q.*;\nrz\(pi/2\) q.*;\n",
            response.get_json().get("circuit"),
        )
        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

    def test_maxcutqaoa_algorithm(self):
        # test simple 4 node graph
        response = self.client.post(
            "/algorithms/maxcutqaoa",
            data=json.dumps(
                {
                    "adj_matrix": [
                        [0, 1, 1, 0],
                        [1, 0, 1, 1],
                        [1, 1, 0, 1],
                        [0, 1, 1, 0],
                    ],
                    "betas": [0.7],
                    "gammas": [1.2],
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(4, response.get_json().get("n_qubits"))
        match = "rzz(1.2) q[0],q[1];\nh q[2];\nrzz(1.2)" in response.get_json().get(
            "circuit"
        )

        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

    def test_maxcutqaoa_algorithm_parameterized(self):
        # test simple 4 node graph
        response = self.client.post(
            "/algorithms/maxcutqaoa",
            data=json.dumps(
                {
                    "adj_matrix": [
                        [0, 1, 1, 0],
                        [1, 0, 1, 1],
                        [1, 1, 0, 1],
                        [0, 1, 1, 0],
                    ],
                    "p": 2,
                    "parameterized": "true",
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(4, response.get_json().get("n_qubits"))
        self.assertIsNotNone(response.get_json().get("circuit"))
        self.assertEqual(response.status_code, 200)

    def test_maxcutqaoa_algorithm_initialState(self):
        # test simple 4 node graph
        response = self.client.post(
            "/algorithms/maxcutqaoa",
            data=json.dumps(
                {
                    "adj_matrix": [
                        [0, 1, 1, 0],
                        [1, 0, 1, 1],
                        [1, 1, 0, 1],
                        [0, 1, 1, 0],
                    ],
                    "betas": [0.7],
                    "gammas": [1.2],
                    "initial_state": "1001",
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(4, response.get_json().get("n_qubits"))
        match = "rzz(1.2) q[0],q[1];\nh q[2];\nrzz(1.2)" in response.get_json().get(
            "circuit"
        )

        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

    def test_maxcutqaoa_algorithm_initialState_parameterized(self):
        # test simple 4 node graph
        response = self.client.post(
            "/algorithms/maxcutqaoa",
            data=json.dumps(
                {
                    "adj_matrix": [
                        [0, 1, 1, 0],
                        [1, 0, 1, 1],
                        [1, 1, 0, 1],
                        [0, 1, 1, 0],
                    ],
                    "parameterized": True,
                    "initial_state": "1001",
                    "p": 2,
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(4, response.get_json().get("n_qubits"))
        match = "rzz(1.2) q[0],q[1];\nh q[2];\nrzz(1.2)" in response.get_json().get(
            "circuit"
        )

        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

    def test_qft_algorithm(self):
        # Test 4 qubit QFT
        response = self.client.post(
            "/algorithms/qft",
            data=json.dumps({"n_qubits": 4, "inverse": False, "barriers": True}),
            content_type="application/json",
        )
        self.assertEqual(4, response.get_json().get("n_qubits"))
        match = re.search(
            "\nqreg q.*;\nbarrier q.*,q.*,q.*,q.*;\nh q.*;\ncp\(pi/2\) q.*,q.*;\ncp\(pi/4\) q.*,q.*;\ncp\(pi/8\) q.*,q.*;\nbarrier q.*,q.*,q.*,q.*;\nh q.*;\ncp\(pi/2\) q.*,q.*;\ncp\(pi/4\) q.*,q.*;\nbarrier q.*,q.*,q.*,q.*;\nh q.*;\ncp\(pi/2\) q.*,q.*;\nbarrier q.*,q.*,q.*,q.*;\nh q.*;\nbarrier q.*,q.*,q.*,q.*;\nswap q.*,q.*;\nswap q.*,q.*;\n",
            response.get_json().get("circuit"),
        )
        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

        # Test 4 qubit inverse QFT
        response = self.client.post(
            "/algorithms/qft",
            data=json.dumps({"n_qubits": 4, "inverse": True, "barriers": True}),
            content_type="application/json",
        )
        self.assertEqual(4, response.get_json().get("n_qubits"))
        match = re.search(
            "\nqreg q.*;\nswap q.*,q.*;\nswap q.*,q.*;\nbarrier q.*,q.*,q.*,q.*;\nh q.*;\nbarrier q.*,q.*,q.*,q.*;\ncp\(-pi/2\) q.*,q.*;\nh q.*;\nbarrier q.*,q.*,q.*,q.*;\ncp\(-pi/4\) q.*,q.*;\ncp\(-pi/2\) q.*,q.*;\nh q.*;\nbarrier q.*,q.*,q.*,q.*;\ncp\(-pi/8\) q.*,q.*;\ncp\(-pi/4\) q.*,q.*;\ncp\(-pi/2\) q.*,q.*;\nh q.*;\nbarrier q.*,q.*,q.*,q.*;\n",
            response.get_json().get("circuit"),
        )
        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

        # Test 4 qubit QFT without barriers
        response = self.client.post(
            "/algorithms/qft",
            data=json.dumps({"n_qubits": 4, "inverse": False, "barriers": False}),
            content_type="application/json",
        )
        self.assertEqual(4, response.get_json().get("n_qubits"))
        match = re.search(
            "\nqreg q.*;\nh q.*;\ncp\(pi/2\) q.*,q.*;\ncp\(pi/4\) q.*,q.*;\ncp\(pi/8\) q.*,q.*;\nh q.*;\ncp\(pi/2\) q.*,q.*;\ncp\(pi/4\) q.*,q.*;\nh q.*;\ncp\(pi/2\) q.*,q.*;\nh q.*;\nswap q.*,q.*;\nswap q.*,q.*;\n",
            response.get_json().get("circuit"),
        )
        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

        # Test 4 qubit inverse QFT without barriers
        response = self.client.post(
            "/algorithms/qft",
            data=json.dumps({"n_qubits": 4, "inverse": True, "barriers": False}),
            content_type="application/json",
        )
        self.assertEqual(4, response.get_json().get("n_qubits"))
        match = re.search(
            "\nqreg q.*;\nswap q.*,q.*;\nswap q.*,q.*;\nh q.*;\ncp\(-pi/2\) q.*,q.*;\nh q.*;\ncp\(-pi/4\) q.*,q.*;\ncp\(-pi/2\) q.*,q.*;\nh q.*;\ncp\(-pi/8\) q.*,q.*;\ncp\(-pi/4\) q.*,q.*;\ncp\(-pi/2\) q.*,q.*;\nh q.*;\n",
            response.get_json().get("circuit"),
        )
        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

    def test_qpe_algorithm(self):
        # Test invalid qasm string
        # suppress message: Error near line ...
        with contextlib.redirect_stdout(None):
            response = self.client.post(
                "/algorithms/qpe",
                data=json.dumps(
                    {
                        "n_eval_qubits": 3,
                        "unitary": 'OPENQASM 2.0; +++++ \ninclude "qelib1.inc";\nqreg q[1];\np(pi/2) q[0];\n',
                    }
                ),
                content_type="application/json",
            )
        self.assertTrue(
            "Invalid unitary (qasm string): \"Expected an ID, received '+'\""
            in response.get_json().get("message")
        )
        self.assertEqual(response.status_code, 400)

        # Test 3 qubit QPE
        response = self.client.post(
            "/algorithms/qpe",
            data=json.dumps(
                {
                    "n_eval_qubits": 3,
                    "unitary": 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\np(pi/2) q[0];\n',
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(4, response.get_json().get("n_qubits"))
        match = re.search(
            "\ngate mcphase\(param0\) q0,q1 { cp\(pi/2\) q0,q1; }\nqreg eval.*;\nqreg q.*;\nh eval.*;\nh eval.*;\nh eval.*;\nmcphase\(pi/2\) eval.*,q.*;\nmcphase\(pi/2\) eval.*,q.*;\nmcphase\(pi/2\) eval.*,q.*;\nmcphase\(pi/2\) eval.*,q.*;\nmcphase\(pi/2\) eval.*,q.*;\nmcphase\(pi/2\) eval.*,q.*;\nmcphase\(pi/2\) eval.*,q.*;\nh eval.*;\ncp\(-pi/2\) eval.*,eval.*;\ncp\(-pi/4\) eval.*,eval.*;\nh eval.*;\ncp\(-pi/2\) eval.*,eval.*;\nh eval.*;\n",
            response.get_json().get("circuit"),
        )
        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

        # Test 2 qubit QPE with 2 qubit operation
        response = self.client.post(
            "/algorithms/qpe",
            data=json.dumps(
                {
                    "n_eval_qubits": 2,
                    "unitary": 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncx q[0], q[1];\n',
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(4, response.get_json().get("n_qubits"))
        match = re.search(
            "\nqreg eval.*;\nqreg q.*;\nh eval.*;\nh eval.*;\nccx eval.*,q.*,q.*;\nccx eval.*,q.*,q.*;\nccx eval.*,q.*,q.*;\nh eval.*;\ncp\(-pi/2\) eval.*,eval.*;\nh eval.*;\n",
            response.get_json().get("circuit"),
        )
        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

    def test_vqe_algorithm(self):
        # Test ansatz & parameters given
        response = self.client.post(
            "/algorithms/vqe",
            data=json.dumps(
                {
                    "ansatz": 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nry(0.1) q[0];\nry(0.2) q[1];\n',
                    "parameters": [0.1, 0.2],
                    "observable": "Z^X",
                }
            ),
            content_type="application/json",
        )
        self.assertTrue(
            'Custom ansatz and parameters not supported. Remove "parameters" field!'
            in response.get_json().get("message")
        )
        self.assertEqual(response.status_code, 400)

        # Test invalid qasm string
        # suppress message: Error near line ...
        with contextlib.redirect_stdout(None):
            response = self.client.post(
                "/algorithms/vqe",
                data=json.dumps(
                    {
                        "ansatz": 'OPENQASM 2.0; +++++ \ninclude "qelib1.inc";\nqreg q[2];\nry(0.1) q[0];\nry(0.2) q[1];\n',
                        "observable": "Z^X",
                    }
                ),
                content_type="application/json",
            )
        self.assertTrue(
            "Invalid ansatz (qasm string): \"Expected an ID, received '+'\""
            in response.get_json().get("message")
        )
        self.assertEqual(response.status_code, 400)

        # Test 2 qubit VQE with custom ansatz
        response = self.client.post(
            "/algorithms/vqe",
            data=json.dumps(
                {
                    "ansatz": 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nry(0.1) q[0];\nry(0.2) q[1];\n',
                    "observable": "Z^X",
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(2, response.get_json().get("n_qubits"))
        match = re.search(
            "\nqreg q.*;\nry\(0.1\) q.*;\nry\(0.2\) q.*;\nh q.*;\n",
            response.get_json().get("circuit"),
        )
        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

        # Test 2 qubit VQE with RealAmplitudes ansatz
        response = self.client.post(
            "/algorithms/vqe",
            data=json.dumps(
                {
                    "parameters": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    "observable": "Z^X",
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(2, response.get_json().get("n_qubits"))
        match = re.search(
            "\nqreg q.*;\nry\(0.1\) q.*;\nry\(0.2\) q.*;\ncx q.*,q.*;\nry\(0.3\) q.*;\nry\(0.4\) q.*;\ncx q.*,q.*;\nry\(0.5\) q.*;\nry\(0.6\) q.*;\ncx q.*,q.*;\nry\(0.7\) q.*;\nh q.*;\nry\(0.8\) q.*;\n",
            response.get_json().get("circuit"),
        )
        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

    def test_grover_algorithm(self):
        # Test errors
        # invalid oracle
        with contextlib.redirect_stdout(None):
            response = self.client.post(
                "/algorithms/grover",
                data=json.dumps(
                    {
                        "oracle": 'OPENQASM 2.0; +++++ \ninclude "qelib1.inc";\nqreg q[3];\nccx q[0],q[1],q[2];\n',
                        "initial_state": 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\nx q[0];\ny q[1];\nz q[2];\n',
                        "iterations": 2,
                        "reflection_qubits": [0, 1],
                        "barriers": True,
                    }
                ),
                content_type="application/json",
            )
        self.assertTrue(
            "Invalid oracle (qasm string): \"Expected an ID, received '+'\""
            in response.get_json().get("message")
        )
        self.assertEqual(response.status_code, 400)

        # Test invalid initial state
        with contextlib.redirect_stdout(None):
            response = self.client.post(
                "/algorithms/grover",
                data=json.dumps(
                    {
                        "oracle": 'OPENQASM 2.0; \ninclude "qelib1.inc";\nqreg q[3];\nccx q[0],q[1],q[2];\n',
                        "initial_state": 'OPENQASM 2.0; +++++ \ninclude "qelib1.inc";\nqreg q[3];\nx q[0];\ny q[1];\nz q[2];\n',
                        "iterations": 2,
                        "reflection_qubits": [0, 1],
                        "barriers": True,
                    }
                ),
                content_type="application/json",
            )
        self.assertTrue(
            "Invalid initial_state (qasm string): \"Expected an ID, received '+'\""
            in response.get_json().get("message")
        )
        self.assertEqual(response.status_code, 400)

        # Test good cases
        # simple oracle (basic case)
        response = self.client.post(
            "/algorithms/grover",
            data=json.dumps(
                {
                    "oracle": 'OPENQASM 2.0; \ninclude "qelib1.inc";\nqreg q[2];\ncz q[0],q[1];\n',
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(2, response.get_json().get("n_qubits"))
        match = re.search(
            "\nqreg q.*;\nh q.*;\nh q.*;\ncz q.*,q.*;\nh q.*;\nx q.*;\nh q.*;\nx q.*;\nh q.*;\ncx q.*,q.*;\nx q.*;\nh q.*;\nh q.*;\nx q.*;\nh q.*;\n",
            response.get_json().get("circuit"),
        )
        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

        # Test initial state
        response = self.client.post(
            "/algorithms/grover",
            data=json.dumps(
                {
                    "oracle": 'OPENQASM 2.0; \ninclude "qelib1.inc";\nqreg q[2];\ncz q[0],q[1];\n',
                    "initial_state": 'OPENQASM 2.0; \ninclude "qelib1.inc";\nqreg q[2];\nx q[0];\ny q[1];\n',
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(2, response.get_json().get("n_qubits"))
        match = re.search(
            "\nqreg q.*;\nx q.*;\ny q.*;\ncz q.*,q.*;\nh q.*;\nx q.*;\nh q.*;\nx q.*;\nh q.*;\ncx q.*,q.*;\nx q.*;\nh q.*;\nh q.*;\nx q.*;\nh q.*;\n",
            response.get_json().get("circuit"),
        )
        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

        # Test iterations
        response = self.client.post(
            "/algorithms/grover",
            data=json.dumps(
                {
                    "oracle": 'OPENQASM 2.0; \ninclude "qelib1.inc";\nqreg q[2];\ncz q[0],q[1];\n',
                    "iterations": 2,
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(2, response.get_json().get("n_qubits"))
        match = re.search(
            "\nqreg q.*;\nh q.*;\nh q.*;\ncz q.*,q.*;\nh q.*;\nx q.*;\nh q.*;\nx q.*;\nh q.*;\ncx q.*,q.*;\nx q.*;\nh q.*;\nh q.*;\nx q.*;\nh q.*;\ncz q.*,q.*;\nh q.*;\nx q.*;\nh q.*;\nx q.*;\nh q.*;\ncx q.*,q.*;\nx q.*;\nh q.*;\nh q.*;\nx q.*;\nh q.*;\n",
            response.get_json().get("circuit"),
        )
        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

        # Test reflection qubits
        response = self.client.post(
            "/algorithms/grover",
            data=json.dumps(
                {
                    "oracle": 'OPENQASM 2.0; \ninclude "qelib1.inc";\nqreg q[3];\nccx q[0],q[1],q[2];\n',
                    "reflection_qubits": [0, 1],
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(3, response.get_json().get("n_qubits"))
        match = re.search(
            "\nqreg q.*;\nh q.*;\nh q.*;\nx q.*;\nh q.*;\nccx q.*,q.*,q.*;\nh q.*;\nx q.*;\nh q.*;\nx q.*;\nh q.*;\ncx q.*,q.*;\nx q.*;\nh q.*;\nh q.*;\nx q.*;\nh q.*;\n",
            response.get_json().get("circuit"),
        )
        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

        # Test barriers
        response = self.client.post(
            "/algorithms/grover",
            data=json.dumps(
                {
                    "oracle": 'OPENQASM 2.0; \ninclude "qelib1.inc";\nqreg q[2];\nbarrier q[0],q[1];\ncz q[0],q[1];\n',
                    "barriers": True,
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(2, response.get_json().get("n_qubits"))
        match = re.search(
            "\nqreg q.*;\nh q.*;\nh q.*;\nbarrier q.*,q.*;\ncz q.*,q.*;\nbarrier q.*,q.*;\nh q.*;\nh q.*;\nbarrier q.*,q.*;\nx q.*;\nx q.*;\nh q.*;\ncx q.*,q.*;\nx q.*;\nh q.*;\nx q.*;\nbarrier q.*,q.*;\nh q.*;\nh q.*;\n",
            response.get_json().get("circuit"),
        )
        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

        # Test all features
        response = self.client.post(
            "/algorithms/grover",
            data=json.dumps(
                {
                    "oracle": 'OPENQASM 2.0; \ninclude "qelib1.inc";\nqreg q[3];\nccx q[0],q[1],q[2];\n',
                    "initial_state": 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\nx q[0];\ny q[1];\nz q[2];\n',
                    "iterations": 2,
                    "reflection_qubits": [0, 1],
                    "barriers": True,
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(3, response.get_json().get("n_qubits"))
        match = re.search(
            "\nqreg q.*;\nx q.*;\ny q.*;\nz q.*;\nccx q.*,q.*,q.*;\nbarrier q.*,q.*,q.*;\nh q.*;\nh q.*;\nbarrier q.*,q.*,q.*;\nx q.*;\nx q.*;\nh q.*;\ncx q.*,q.*;\nx q.*;\nh q.*;\nx q.*;\nbarrier q.*,q.*,q.*;\nh q.*;\nh q.*;\nccx q.*,q.*,q.*;\nbarrier q.*,q.*,q.*;\nh q.*;\nh q.*;\nbarrier q.*,q.*,q.*;\nx q.*;\nx q.*;\nh q.*;\ncx q.*,q.*;\nx q.*;\nh q.*;\nx q.*;\nbarrier q.*,q.*,q.*;\nh q.*;\nh q.*;\n",
            response.get_json().get("circuit"),
        )
        self.assertTrue(match is not None)
        self.assertEqual(response.status_code, 200)

    def test_tsp_qaoa(self):
        # test tsp qaoa and openqasm
        for i in range(25):
            n = np.random.randint(3, 5)
            matrix = np.random.rand(n, n)
            p = np.random.randint(1, 4)
            betas, gammas = np.random.rand(p), np.random.rand(p)
            request = {
                "adj_matrix": matrix.tolist(),
                "p": p,
                "betas": betas.tolist(),
                "gammas": gammas.tolist(),
            }

            response = self.client.post(
                "algorithms/tspqaoa",
                data=json.dumps(request),
                content_type="application/json",
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(n ** 2, response.get_json().get("n_qubits"))

            openqasm = response.get_json().get("circuit")
            qc = QuantumCircuit.from_qasm_str(openqasm)
            self.assertTrue(isinstance(qc, QuantumCircuit))
