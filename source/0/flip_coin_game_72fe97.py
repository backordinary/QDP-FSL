# https://github.com/susumige/questforconsciousness.github.io-src/blob/b9e570c44064c2992c9aa0e81fe5557500b3ef58/content/code/flip_coin_game.py
from qiskit import QuantumCircuit, Aer, execute
import numpy as np

tosses = 15


def input_guess():
    guess = str(input("What's your guess heads/tails? [h/t]: ")).lower().strip()
    try:
        if guess[0] in ('h', 't'):
            return guess[0]
        else:
            print('Invalid Input')
            return input_guess()
    except Exception as error:
        print("Please enter valid inputs")
        print(error)
        return input_guess()


def main():
    """Asks for guess"""
    guess = input_guess()
    heads = 0
    tails = 0

    qc = QuantumCircuit(1)  # quantum circuit with one qubit
    qc.h(0)  # applying Hadamard gate
    qc.measure_all()

    for i in range(tosses):
        backend = Aer.get_backend('statevector_simulator')
        result = execute(qc, backend).result()

        coin = result.get_statevector()

        if np.all(coin == np.array([0., 1.])):
            heads = heads + 1
        else:
            tails = tails + 1

    if (guess == 'h' and heads > tails) or (guess == 't' and tails > heads):
        msg = 'You WON!'
    else:
        msg = 'You LOST!'

    print('{}\nThe experiment resulted in: {} heads and {} tails.'
          .format(msg, heads, tails))


main()
