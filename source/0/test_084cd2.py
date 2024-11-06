# https://github.com/QPoland/walcz-o-superpozycje-2021/blob/2cb7c7866aca53e67c3e369d1dc31e83ea4640f1/przygotowanie_srodowiska/test.py
from qiskit import QuantumCircuit, QuantumRegister
from hashlib import sha512
from qiskit.providers.aer import QasmSimulator

def _counts_to_str(counts: dict):
    return str(sorted(list(zip(counts.keys(), counts.values()))))

def _get_hash(counts: dict):
    return sha512(str.encode(_counts_to_str(counts))).hexdigest()

n = 8
qreg = QuantumRegister(n)
qcirc = QuantumCircuit(qreg)
qcirc.h(qreg)
qcirc.measure_all()

qasm = QasmSimulator(seed_simulator=42) # nie zmieniaj wartości seed_simulator!
result = qasm.run(qcirc).result().get_counts(qcirc)
the_hash = _get_hash(result)
if the_hash == "de66f0cee3e8e43b41b4f7d4a61be6cd68313c3c26e28057d244a65fb262a3de755c1df1c6de043aa22a12a318dd906ee9190dbdaea7d1e67949f5512c661599":
    print("Wszystko OK!")
else:
    print("Twoje srodowisko nie jest poprawne - sprobuj skontaktowac sie z organizatorami przez dedykowany kanal Discord")
