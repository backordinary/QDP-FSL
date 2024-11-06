# https://github.com/melmatt7/quantum_algos/blob/3e1821d0ab7dcc5cdd59d6b049e81d3c5ba15a9b/qiskit/__base.py
import numpy as np
import math
import matplotlib.pyplot as plt
from random import shuffle
#from pylatexenc.latex2text import LatexNodes2Text

from qiskit import Aer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.visualization import plot_bloch_vector
from qiskit.result.counts import Counts


def bloch_vec(qc):
    backend = Aer.get_backend('statevector_simulator')
    ket = execute(qc, backend).result().get_statevector()
    if ket[0] !=0:
        theta = 2*np.arctan(np.abs(ket[1]/ket[0]))
        phi = np.angle(ket[1]/ket[0])
    else:
        theta = np.pi
        phi = 0
    bloch_vector = np.round([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], 5)
    return bloch_vector


def check_answer(output,sol):
    if (output == sol).all():
        return True
    else:
        if (round(output,5) == round(sol,5)).all():
            return True
        else:
            return False


def check(ex, sol, nr, hint=''):
    output = ex
    if(type(sol) == QuantumCircuit):
        if output == sol:
            print("\033[1;30;42m Correct! Well done! \n")
            score = 1
        else:
            print("\033[1;30;41m Incorrect, try again! \n")
            score = 0
        return score
    if (type(sol) == Counts):
        compare_dict = np.zeros(len(sol))
        m = 0
        for key in sol:
            if key in ex:
                if abs(sol[key]-ex[key])<= 100:
                    compare_dict[m] = True
            m+=1
        if compare_dict.all() == True:
            print("\033[1;30;42m Correct! Well done! \n")
            score = 1
        else:
            print("\033[1;30;41m Incorrect, try again! \n")
            score = 0
        return score
    """if (type(sol) == np.ndarray):
        if output.all() == sol.all():
            print("\033[1;30;42m Correct! Well done! \n")
            score = 1
        else:
            if output.all() == -1*sol.all():
                print("\033[1;30;42m Correct! Well done! \n")
                score = 1
            else:
                print("\033[1;30;41m Incorrect, try again! \n")
                score = 0
        return score
    else:
        print("different solution type")"""
    check = np.zeros(len(sol))
    for i in range(len(sol)):
        check[i] = check_answer(output[i], sol[i])
        print(check[i])
    if check.all() == True:
        print("\033[1;30;42m Correct! Well done! \n")
        score = 1
    else:
        if nr>=4:
            sol*=-1
            for i in range(len(sol)):
                check[i] = check_answer(output[i], sol[i])
                print(check[i])
        if check.all() == True:
            print("\033[1;30;42m Correct! Well done! (-1* statevector)\n")
            score = 1
        else:
            print("\033[1;30;41m Incorrect, try again! \n")
            score = 0
            print(hint)
    return score

def run_and_check(ex, run_sol):
    sol, nr = run_sol
    score[nr-1] = check(ex,sol,nr)
    totalscore = int(sum(score))
    print('\033[1;30;47m Your score: ', totalscore, "/8")


def qc_check(circ, sol):
    if type(sol) is QuantumCircuit:
        if circ == sol:
            print("state preparation correct")
        else:
            print("state preparation wrong")

def intro_sol1():
    qc = QuantumCircuit(1)
    qc.x(0)
    bloch_vector = bloch_vec(qc)
    return bloch_vector, 1


def intro_sol2():
    qc = QuantumCircuit(1)
    qc.h(0)
    bloch_vector = bloch_vec(qc)
    return bloch_vector, 2


def intro_sol3():
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.h(0)
    bloch_vector = bloch_vec(qc)
    return bloch_vector, 3

def intro_sol4():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.sdg(0)
    bloch_vector = bloch_vec(qc)
    return bloch_vector, 4

def intro_sol5():
    qc = QuantumCircuit(2,2)
    qc.h(0)
    qc.cx(0,1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    backend = Aer.get_backend('qasm_simulator')
    counts = execute(qc, backend, shots = 1000).result().get_counts()
    return counts, 5

def intro_sol6():
    qc = QuantumCircuit(2,2)
    qc.h(0)
    qc.cx(0,1)
    qc.z(1)
    qc.x(1)
    backend = Aer.get_backend('statevector_simulator')
    vec = execute(qc, backend).result().get_statevector()
    print(vec)
    return vec, 6

def intro_sol7():
    qc = QuantumCircuit(2,2)
    qc.rx(np.pi/3,1)
    qc.x(0)
    qc.measure(0, 0)
    qc.measure(1, 1)
    backend = Aer.get_backend('qasm_simulator')
    result = execute(qc, backend, shots = 1000).result()
    counts = result.get_counts()
    return counts, 7

def intro_sol8():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0,1)
    qc.cx(0,2)
    backend = Aer.get_backend('statevector_simulator')
    vec = execute(qc, backend).result().get_statevector()
    return vec, 8


def check_mcanswer(answer, answers):
    if answers[answer_dict[answer]][1] == 0:
        print("\033[1;30;42m Correct! Well done!")
        score = 1
    else:
        print("\033[1;30;41m Incorrect, try again!")
        score = 0
    return score


def ltx(latexstring):
    return LatexNodes2Text().latex_to_text(latexstring)


def create_question(answernr):
    answers = [[ltx(answernr[0]), 0],
               [ltx(answernr[1]), 1],
               [ltx(answernr[2]), 2],
               [ltx(answernr[3]), 3],
               [ltx(answernr[4]), 4]]
    shuffle(answers)
    print(' a:', answers[0][0], '\n b:', answers[1][0],
          '\n c:', answers[2][0], '\n d:', answers[3][0], '\n e:', answers[4][0])
    return answers


def question_and_check(answernr):
    all_answers = create_question(answernr)
    answer = input()
    nr = answernr[5]
    score[nr-1] = check_mcanswer(answer, all_answers)
    totalscore = int(sum(score))
    print('\033[1;30;47m Your score: ', totalscore, "/10")

def combine_alg_and_appl(answernr, algos):
    checks = np.zeros(5)
    all_answers = create_question(answernr)
    print(algos[0])
    answer = input()
    if all_answers[answer_dict[answer]][1] == 0:
        print("Correct!")
        checks[0] = 1
    else:
        print("Incorrect!")
        checks[0] = 0
    print(algos[1])
    answer = input()
    if all_answers[answer_dict[answer]][1] == 1:
        print("Correct!")
        checks[1] = 1
    else:
        print("Incorrect!")
        checks[1] = 0
    print(algos[2])
    answer = input()
    if all_answers[answer_dict[answer]][1] == 2:
        print("Correct!")
        checks[2] = 1
    else:
        print("Incorrect!")
        checks[2] = 0
    print(algos[3])
    answer = input()
    if all_answers[answer_dict[answer]][1] == 3:
        print("Correct!")
        checks[3] = 1
    else:
        print("Incorrect!")
        checks[3] = 0
    print(algos[4])
    answer = input()
    if all_answers[answer_dict[answer]][1] == 4:
        print("Correct!")
        checks[4] = 1
    else:
        print("Incorrect!")
        checks[4] = 0
    nr = answernr[5]
    score[nr - 1] = checks.all()
    if checks.all()== True:
        print("\033[1;30;42m All answers matched correctly! Well done!")
    else:
        print("\033[1;30;41m Not all answers matched correctly, try again!")
    totalscore = int(sum(score))
    print('\033[1;30;47m Your score: ', totalscore, "/10")


global score
score = np.zeros(10)

answer_dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}

answers1 = [r'$2^10$',
            r'$20$',
            r'$10$',
            r'$10^2$',
            r'$10^10$',
            9]

app = [r'integer factorization',
            r'database search',
            r'approximation of measurement probabilities',
            r'find the ground state energy of a molecule',
            r'combinatorial optimization',
            10]

algos = [r"Shor's algorithm",
              r"Grover's algorithm",
              r'amplitude estimation',
              r'Variational Quantum Eigensolver',
              r'QAOA']
