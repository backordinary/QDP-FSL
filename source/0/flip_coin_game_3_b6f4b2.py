# https://github.com/susumige/questforconsciousness.github.io-src/blob/b9e570c44064c2992c9aa0e81fe5557500b3ef58/content/code/flip_coin_game_3.py
from qiskit import QuantumCircuit
from qiskit import IBMQ
from qiskit.compiler import transpile, assemble

tosses = 15 # The number of experiments in the Qobj (101) is higher than the number of experiments supported by the device (75)
qc_list = []


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
    # Authenticate an account and add for use during this session. Replace string
    # argument with your private token.
    IBMQ.load_account()

    """Asks for guess"""
    guess = input_guess()
    heads = 0
    tails = 0

    for i in range(tosses):
        qc = QuantumCircuit(1)  # quantum circuit with one qubit
        qc.h(0)  # applying Hadamard gate
        qc.measure_all()
        qc_list.append(qc)

    # Execute a list of circuits
    provider = IBMQ.get_provider(group='open')
    backend = provider.get_backend('ibmq_london')
    transpiled_circs = transpile(qc_list, backend=backend)
    qobjs = assemble(transpiled_circs, backend=backend)
    job_info = backend.run(qobjs)

    # Measure the results
    for circ_index in range(len(transpiled_circs)):
        result = job_info.result().get_counts(transpiled_circs[circ_index])
        print(result)
        heads_count = result['0']
        tails_count = result['1']
        if heads_count > tails_count:
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
