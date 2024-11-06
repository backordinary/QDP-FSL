# https://github.com/Trufelek124/projekt_ppp/blob/b28123e980af925bca46dd8bdbabad67a75a8163/main.py
import quantum_settings_service as qss
import quantum_circuit_creator as qcc
import num_to_binary_converter as nbc
import ibmq_quantum_service as iqs
from qiskit import *


def main():
    print("elo!")
    setup_quantum_connection()

    backend_name="qasm_simulator"
    num_of_shots=1024

    backend = qss.get_backend(backend_name)
    iqs.run_program_on_backend(backend, num_of_shots)

    print("elo2")

def setup_quantum_connection():
    IBMQ.load_account()

# zrobić program
# program powinien rysować kwantowego sinusa

# będzie się podawało backend -> run
# moze liczbę powtórzeń
# będzie zwracało tablicę z wynikami
# moze po 5 obliczeń na raz, zeby było szybciej


# na dockerze zeby stało
# po stronie frontu - rusować normalną sinusoidę
# nie musi być logowania
main()