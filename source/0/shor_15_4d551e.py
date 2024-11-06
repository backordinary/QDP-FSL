# https://github.com/tokyotech-toyota/quantum_computing/blob/f60b1c7197a1f15d91d07c9b435a0ec44847f10b/shor_15/shor_15.py
#!/usr/bin/env -S python3 -W ignore
# coding: utf-8
################################################################################
from __future__ import print_function
import tbvaccine as tb; tb.add_hook(isolate=False, show_vars=False)
import os
import random
import numpy as np
import qiskit
from qiskit import (
    IBMQ,
    QuantumRegister,
    ClassicalRegister,
    QuantumCircuit,
    Aer,
    execute,
    transpile,
)
################################################################################


##################################################
################### Token auth ###################
##################################################
token_path = "../IBMQ_token"
with open(token_path, "r") as file:
    token = file.read()
IBMQ.save_account(token, overwrite=True)


##################################################
################## Set problem ###################
##################################################
M = 15
N = 4  # Satisfy 2^N > M
x = 0  # Initialization
r = 0  # Initialization


################################################################################
##################################### Main #####################################
################################################################################
def main():
    ########## Global variables ##########
    global M, N, x, r
    
    ##################################################
    ##################### Step 1 #####################
    ##################################################
    print("########## Step 1 ##########")
    check_even_odd()

    ##################################################
    ##################### Step 2 #####################
    ##################################################
    print("########## Step 2 ##########")
    check_exp()

    ##################################################
    ##################### Step 3 #####################
    ##################################################
    print("########## Step 3 ##########")
    pick_random_x()

    ##################################################
    ##################### Step 4 #####################
    ##################################################
    print("########## Step 4 ##########")
    
    ########## Make circuit ##########
    x = 7; print("DEBUG FIX: x = {} !!!!!!!!!!".format(x))
    circ = make_circuit()
    
    ########## Print circuit ##########
    circ_trans = transpile(circ)
    style = {"dpi":200, "showindex":True, "cregbundle":True, "margin":[1.5,1,0.5,1]}
    circ_trans.draw(output="mpl", filename="./circuit.png",
                    style=style, initial_state=True, plot_barriers=True, fold=20,)

    ########## Run job ##########
    backend = Aer.get_backend("qasm_simulator")
    shots = 4096
    job = execute(circ_trans, backend=backend, shots=shots)

    ########## Print result ##########
    result = job.result()
    result_dict = result.get_counts(circ_trans)
    
    ########## Export result ##########
    with open("result.ssv", "w") as file:
        file.write("# qubit probability\n")
        for i in result_dict:
            file.write("{0} {1:.8E}\n".format(i, result_dict[i]/shots))
            
    ##################################################
    ##################### Step 5 #####################
    ##################################################
    print("########## Step 5 ##########")
    find_factor()


################################################################################
################################################################################
################################################################################
def find_factor():
    if r % 2 == 0:
        print("r = {} is even number, continue...".format(r))
    else:
        print("r = {} is odd number, exit.".format(r))
        quit(0)
    f1 = gcd(x1=x**(r//2)-1, x2=M)
    f2 = gcd(x1=x**(r//2)+1, x2=M)
    print("At least one of followings will be the answer:")
    print("Answer1: {} = {} * {}".format(M, f1, M//f1))
    print("Answer2: {} = {} * {}".format(M, f2, M//f2))
    

################################################################################
################################################################################
################################################################################
def gcd(x1=[], x2=[]):
    """
    Euclidean algorithm
    """
    while x1*x2 != 0:
        [a, b] = sorted([x1, x2], reverse=True)  # a > b
        r = a % b  # a = b*q + r
        x1 = b
        x2 = r
    return abs(x1 - x2)


################################################################################
################################################################################
################################################################################
def pick_random_x():
    """
    Pick up x (1 <= x <= M-1) randomly, then:
    * gcd(x,M) == 1 -> return x, continue...
    * gcd(x,M) != 1 -> gcd(x,M) is factor of M, exit.
    """
    x = random.randint(1, M-1)
    while (g := gcd(x, M)) != 1:
        print("M contains {} as a factor, exit.".format(g))
        print("Answer: {} = {} * {}".format(M, g, M//g))
        quit(0)
        x = random.randint(1, M-1)
    print("x = {} is picked up, continue...".format(x))
    # return x


################################################################################
################################################################################
################################################################################
def check_exp():
    """
    Answer Yes / No:
    M = a^b (a>=1, b>=2)
    """
    for b in range(2, M):
        a = M**(1/b)
        if a.is_integer():
            print("M contains {} as a factor, exit.".format(int(a)))
            print("Answer: {} = {} * {}".format(M, int(a), b))
            quit(0)
        elif a < 2.0:
            print("M does NOT consist of a^b, continue...")
            break


################################################################################
################################################################################
################################################################################
def check_even_odd():
    if M % 2 == 0:
        print("M contains 2 as a factor, exit.")
        print("Answer: {} = {} * {}".format(M, 2, M//2))
        quit(0)
    else:
        print("M does NOT contain 2 as a factor, continue...")


################################################################################
################################################################################
################################################################################
def make_circuit():
    ########## Initialization ##########
    qr = QuantumRegister(5)
    cr = ClassicalRegister(4)
    circ = QuantumCircuit(qr, cr)
    circ.h(0)
    circ.h(1)
    circ.h(2)
    circ.h(3)
    circ.x(4)
    ########## Order finding ########## <- Under development
    circ.barrier()
    # angle = 2*np.pi/4
    # repetitions = 1
    # for counting_qubit in range(4):
    #     for i in range(repetitions):
    #         circ.cu1(angle, counting_qubit, 4)
    #     repetitions *= 2
        
    ########## QFT^(-1) ##########
    circ.barrier()
    circ.swap(0, 3)
    circ.swap(1, 2)
    circ.barrier()
    circ.h(3)
    circ.barrier()
    circ.cu1(-np.pi/2, 3, 2)
    circ.h(2)
    circ.barrier()
    circ.cu1(-np.pi/4, 3, 1)
    circ.cu1(-np.pi/2, 2, 1)
    circ.h(1)
    circ.barrier()
    circ.cu1(-np.pi/8, 3, 0)
    circ.cu1(-np.pi/4, 2, 0)
    circ.cu1(-np.pi/2, 1, 0)
    circ.h(0)
    ########## Measurement ##########
    circ.barrier()
    circ.measure([0,1,2,3], [3,2,1,0])
    ########## Return circuit ##########
    return circ

            
################################################################################
################################################################################
################################################################################
if __name__ == "__main__":
    main()
