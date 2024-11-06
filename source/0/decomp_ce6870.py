# https://github.com/maximkha/qnet/blob/827778fe0a5837f4c2695d5c5897204b3afd199c/decomp.py
import numpy as np
from qiskit.circuit import gate
import quantum_decomp
from pathlib import Path
from qiskit import QuantumCircuit, transpile
from qiskit.test.mock import FakeBoeblingen, FakeAthens, FakeYorktown
import qiskit.quantum_info as qi

backend = FakeYorktown()

RTOL = 1e-3

PASSES = 4
OPT_LEVEL= 4

for file in (Path(".") / "intermediate").glob("mat*.npy"):
    print(f"Processing {file.name}")
    matrix = np.load(file)

    print(f"{matrix=}")

    qiskit_circuit = quantum_decomp.matrix_to_qiskit_circuit(matrix)
    op = qi.Operator(qiskit_circuit)
    print(f"{qiskit_circuit=}")
    print(f"{op.data=}")
    print(f"{np.allclose(op.data, matrix)}")
    
    gate_op_min = qiskit_circuit.depth()
    gate_min = qiskit_circuit
    
    for _ in range(PASSES):
        for kk in range(OPT_LEVEL):
            qiskit_circuit_OPT = transpile(qiskit_circuit, backend, optimization_level=kk)
            depth = qiskit_circuit_OPT.depth()
            print('Optimization Level {}'.format(kk))
            print('Depth:', depth)
            print('Gate counts:', qiskit_circuit_OPT.count_ops())
            if gate_op_min > depth and depth != 1:
                print("Found better!")
                op = qi.Operator(qiskit_circuit_OPT)
                print(f"{np.allclose(op.data[:matrix.shape[0], :matrix.shape[1]], matrix, rtol=RTOL)=}")
                print(f"{op.data[:matrix.shape[0], :matrix.shape[1]] - matrix =}")
                if not np.allclose(op.data[:matrix.shape[0], :matrix.shape[1]], matrix, rtol=RTOL):
                    print("Invalid matrix!")
                    continue
                
                gate_min = qiskit_circuit_OPT
                gate_op_min = depth
            print()

        print("Mat check")
        op = qi.Operator(gate_min)
        print(f"{np.allclose(op.data[:matrix.shape[0], :matrix.shape[1]], matrix, rtol=RTOL)=}")
        print(f"{op.data[:matrix.shape[0], :matrix.shape[1]] - matrix =}")
        if not np.allclose(op.data[:matrix.shape[0], :matrix.shape[1]], matrix, rtol=RTOL):
            print("Invalid matrix!!")
            exit()

    gate_min.qasm(filename=f"intermediate/{file.stem}.qasm")
    file.unlink()
    print("")