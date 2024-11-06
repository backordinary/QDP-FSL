# https://github.com/mariuskriukovas/Projektinis/blob/ed4de5f13a6053a9ef1f18e133b6fc70fca3bc60/tools/simulators.py
import matplotlib.pyplot as plt
from qiskit import execute, Aer
from qiskit.visualization import plot_bloch_multivector, circuit_drawer
from qiskit.visualization import plot_histogram

number_of_shots = 1024

def simulate(qc):
    job = execute(qc, Aer.get_backend('qasm_simulator'), shots=number_of_shots)
    return job.result()

def simulate_and_return_result(qc):
    result = simulate(qc).get_counts()
    return result

def simulate_and_show_result(qc, title = "", circle_title = ""):
    result = simulate(qc).get_counts()
    circuit_drawer(qc, output='mpl')
    plt.title(circle_title)
    # qc.draw(output='mpl', title = "labas")
    # print(result)
    plot_histogram(result, title=title)
    plt.show()
    return result

def simulate_unitary(qc):
    # qc.draw(output='mpl')
    backend = Aer.get_backend('unitary_simulator')
    job = execute(qc, backend)
    return job.result()

def simulate_unitary_matrix_df(qc):
    result = simulate_unitary(qc)
    unitary = result.get_unitary(qc, decimals=3)

    # (a,_) = unitary.shape
    # index = list(range(0,a))
    # df = pd.DataFrame(data=unitary, index=index, columns=index)
    # pretty_print_df(df)
    # print(df)

    return unitary


def add_point(plt, x, y, idx, color = "red"):
    plt.scatter(x,y, c=color)
    plt.text(x,y, f' $T_{str(idx)}$')

def simulate_bloch_sphere(qc, title):
    qc.draw(output='mpl')
    backend = Aer.get_backend('statevector_simulator')
    out_state = execute(qc, backend).result().get_statevector()
    print(out_state)
    plot_bloch_multivector(out_state)
    # plot_bloch_multivector(out_state, title=title)
    # plt.title(title)
    plt.show()



def all_binary_combs(size = 3):

    def give_me_zeros(str, max_len):
        n = len(str)
        n = max_len - n
        n_str = str
        for i in range(0, n):
            n_str = "0" + n_str
        return n_str

    n = 2 ** size
    res = []
    for i in range(0, n):
        bytes =  "{0:b}".format(i)
        res.append(give_me_zeros(bytes, size))
    return res