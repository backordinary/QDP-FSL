# https://github.com/hmy98213/Fault-Simulation/blob/e96bcde84f27f0470c94a6438761063c7e9bc1aa/draw_qasm.py
from qiskit import QuantumCircuit

def draw_qaoa():
    with open("pic.qasm", 'r') as f:
        qasm_str = f.read()
    qc = QuantumCircuit.from_qasm_str(qasm_str)
    return qc.draw('latex')

if __name__ == "__main__":
    draw_qaoa()