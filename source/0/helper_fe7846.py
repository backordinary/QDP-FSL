# https://github.com/mariuskriukovas/Bakalaurinis/blob/f877821cb75d534b28c1d3d03ae2ac0df5cc68b6/tools/helper.py
from qiskit import execute, Aer
import numpy as np
import array_to_latex as a2l
from qiskit import execute, Aer
from qiskit.visualization import plot_bloch_multivector
from qiskit.visualization import plot_histogram

number_of_shots = 1024 * 1

def simulate(qc):
    job = execute(qc, Aer.get_backend('qasm_simulator'), shots=number_of_shots)
    return job.result()

def convert_result_to_np_arr(counts):
    keys = list(counts.keys())
    arr_len = 2 ** len(keys[0])
    arr = np.zeros(arr_len)
    for k in keys:
        idx = int(k, 2)
        print(idx)
        arr[idx] = counts[k]
    return arr


def simulate_and_return_result(qc):
    result = simulate(qc).get_counts()
    result = convert_result_to_np_arr(result)
    result = result/1024
    return result

def simulate_and_show_result(qc, title = "", circle_title = ""):
    result = simulate(qc).get_counts()
    print(result)
    # circuit_drawer(qc, output='mpl')
    # plt.title(circle_title)
    # qc.draw(output='mpl', title = "labas")
    # print(result)
    plot_histogram(result, title=title)
    # plt.show()
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

def simulate_state_value(qc):
    # qc.draw(output='mpl')
    backend = Aer.get_backend('statevector_simulator')
    return execute(qc, backend).result().get_statevector()

def simulate_bloch_sphere(qc, title):
    out_state = simulate_state_value(qc)
    print(out_state)
    plot_bloch_multivector(out_state, title=title)
    # plt.title(title)
    # plt.show()



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

def to_latex(a):
    return a2l.to_ltx(a, frmt='{:6.2f}', arraytype='bmatrix')

def to_wolf_matrix(A):
    print("{", end="")
    last = len(A) - 1

    if isinstance(A[0], list):
        for i, x in enumerate(A):
            print("{", end="")
            last_x = len(x) - 1

            for j, y in enumerate(x):
                print(y, end=", ") if j != last_x else print(y, end="")

            print("}", end=", ") if i != last else print("}", end="")
    else:
        last_x = len(A) - 1
        print("{", end="")
        for i, x in enumerate(A):
            print(x, end=", ") if i != last_x else print(x, end="")
        print("}" , end="")
    print("}")

from git.Bakalaurinis.simuliator.math import find_prob

def printProb(v):
    prob = find_prob(v)
    for i in range(0, len(v)):
        b = bin(i).replace("0b", "")
        print("( " + str(b) + " ) " + str(v[i]) + "---->" + str(prob[i]))
