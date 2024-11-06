# https://github.com/QPoland/walcz-o-superpozycje-2021/blob/a43e9cd4f2458b1f216ec974f8b4a8e46720212b/najlepsze_zgloszenia/bnp_challenge_verifier/graders.py
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller
    
def grade_circuit(qc, quiet=False): 
    if not isinstance(qc, QuantumCircuit):
        print("Proszę prześlij obiekt typu QuantumCircuit!")
        return None

    pass_ = Unroller(['u3', 'cx'])
    pm = PassManager(pass_) 
    try:
        qc = transpile(qc, basis_gates=['u3', 'cx']) 
        qc = pm.run(qc)
    except:
        print("Błąd upraszczania obwodu - prawdopodobnie użyłeś niedozwolonych operacji.")
        return None

    ops_dict = qc.count_ops()
    cnot_counter = ops_dict['cx'] if 'cx' in ops_dict.keys() else 0
    u3_counter = ops_dict['u3'] if 'u3' in ops_dict.keys() else 0

    cost = 10*cnot_counter + u3_counter
    if not quiet:
        print("Koszt twojego obwodu wynosi {}.".format(cost))
    return cost