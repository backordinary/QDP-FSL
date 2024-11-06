# https://github.com/JohnSu0713/QSSO_func/blob/eb65d1206401ba6e84deb4b37e2c58dc9d7f36c4/QSSO_v1/QSSO_qiskit.py
import qiskit
import time
import matplotlib.pyplot as plt
import numpy as np
import qrng
import seaborn as sns
import pandas as pd
from math import pi
from qiskit import IBMQ, Aer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute


#################### Initialize HyperParameter ####################
MaxNsol = 300           # define MaxNsol
MaxNvar = 150           # define MaxNvar

# Nrun = 10             # define Nrun
Ngen = 10               # define Ngen
Nsol = 5                # define Nsol
Nvar = 10               # define Nvar
Nbit = 10               # define the variable bits
Qbest_sol = Nsol + 1    # solution Number include Qbest_sol
Qbest_idx = Nsol        # Index of Qbest_sol
theta_1 = 0.08 * pi
theta_2 = 0.001 * pi

Cg = 0.7                # define Cg
Cp = 0.9                # define Cp
# Cw = 0.9              # define Cw
apitoken = 'e26cab383f69b81aaf36106b0620f3e994afd5ef57bb861bc714b41a91dabb3c08dcc5f5f45fa395a490d1bbf0490a8b1e7caeeaa560546833ca8149d4375ef3'
#################### Initialize HyperParameter ####################


def Qsol_init():
    '''
    return a qc_list represent the solution value [x1, x2, x3, ...]
    '''
    qc_list = []
    for var in range(Nvar):
        qc = QuantumCircuit(Nbit, Nbit)
        qr = QuantumRegister(Nbit, 'q')
        cr = ClassicalRegister(Nbit, 'c')
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0:Nbit])
        qc_list.append(qc)
    return qc_list


def God_init():
    '''
    Initialize one generation with Nsol + 1.
    Last one qc_list is Qgbest.
    '''
    God_list = []
    for sol in range(Qbest_sol):
        God_list.append(Qsol_init())
    return God_list


def sol_measure(Qsol_list):
    '''
    Measure a Quantum state solution to Classical presentation.
    '''
    Csol_list = []
    Qstate_list = []
    simulator = Aer.get_backend('qasm_simulator')

    for qc in Qsol_list:
        qc_copy = qc
        for bit in range(Nbit):
            qc_copy.measure(bit, bit)
        result = execute(qc_copy, simulator).result()  # defaul shot number
        result_dict = result.to_dict()
        Qstate_list.append(result_dict['results'][0]['data']['counts'])
    for var in Qstate_list:
        Keymax = max(var, key=var.get)
        Csol_list.append(int(Keymax, 16))
    return Csol_list


def var_measure(qc):
    '''
    Measure a Q state variable to classical bits.
    '''
    simulator = Aer.get_backend('qasm_simulator')
    qc_copy = qc.copy()
    for bit in range(Nbit):
        qc_copy.measure(bit, bit)
    result = execute(qc_copy, simulator).result()  # defaul shot number
    result_dict = result.to_dict()
    var_dict = result_dict['results'][0]['data']['counts']
    Keymax = max(var_dict, key=var_dict.get)
    return int(Keymax, 16)


def FIT_cal(XX):
    '''
    This function define the OBJ function and return the fitness value.
    In this example we use the square sum of each variable.
    '''
    SUM = 0
    for var in range(Nvar):
        SUM += XX[var] * XX[var]
    return SUM


def best_dict_init():
    '''
    Use best_dict to maintain the gbest_fitness, gbest_list, pbest_fitness, pbest_list
    '''
    best_dict = {
        "gbest_fitness": 0,
        "gbest_sol": [0 for i in range(Nvar)],
        "pbest_fitness": [0 for i in range(Nsol)],
        "pbest_list": [[0 for i in range(Nvar)] for j in range(Nsol)]
    }
    return best_dict


def int_to_bin(var):
    '''
    Define a function to convert integer to binary_string.
    '''
    return format(var, f'0{Nbit}b')


def get_Qrnd():
    '''
    Function to get a random floating number from a Qubit in Quantum Computer.
    '''
    qrng.set_backend()
    rnd = qrng.get_random_float(0, 1)
    return rnd


def sol_update(sol):
    '''
    Get a solution with Nvar from Qsol_list.
    '''
    Random_Qlist = Qsol_init()
    sol_list = []
    for var in range(Nvar):
        rnd = get_Qrnd()
        if (rnd < Cg):
            variable = (var_measure(God_list[Qbest_idx][var]))
            sol_list.append(variable)
        elif (rnd < Cp):
            variable = (var_measure(God_list[sol][var]))
            sol_list.append(variable)
        else:
            variable = (var_measure(Random_Qlist[var]))
            sol_list.append(variable)
    return sol_list


def get_bin_sol(sol_list):
    '''
    Convert a decimal solution list into a binary solution list.
    '''
    bin_list = []
    for var in sol_list:
        bin_list.append(int_to_bin(var))
    return bin_list


def get_ab_list(Qsol):
    '''
    Return a list with ai bi state in Qsol_list.
    '''
    backend = Aer.get_backend('statevector_simulator')
    ab_list = []
    for var in range(Nvar):
        ab_var = []
        for bit in range(Nbit):
            qc_tmp = Qsol[var].copy()
            collapse_bit = [i for i in range(Nbit) if i != bit]
            qc_tmp.measure(collapse_bit, collapse_bit)
            out_state = execute(qc_tmp, backend).result().get_statevector()
            out_state_complex = out_state.tolist()
            out_state_real = [
                out_state_complex[i].real for i in range(len(out_state_complex))]
            ab_bit = [state for state in out_state_real if state != float(0)]
            # print(ab_bit) # if need to check the amplitude
            # if there is a the extreme condition!
            if len(ab_bit) < 2:
                print("\n +-1 in the state_list!!")
                for i in range(len(out_state_real)):
                    if (out_state_real[i] != float(0)):
                        if ((int(int_to_bin(i)[Nbit - 1 - bit]) == 1)):
                            ab_bit = [0, 1]
                        elif (int(int_to_bin(i)[Nbit - 1 - bit]) == 0):
                            ab_bit = [1, 0]
            ab_var.append(ab_bit)
        ab_list.append(ab_var)
    return ab_list


def God_update(sol, sol_list, sol_fitness):
    '''
    Update the gbest and pbest for "solution_i" in best_dict.
    '''
    global best_dict
    global God_list

    curr_bin_sol = get_bin_sol(sol_list)
    # if curr_sol is better(in this case: smaller)

    if sol_fitness >= best_dict["gbest_fitness"]:
        # get the ai, bi of God_list[Qbest_idx]
        Qgbest_ab = get_ab_list(God_list[Qbest_idx])
        print(f'Get better Gbest: {sol_fitness}!\n')

        # get the binary sol_list to compare
        gbest_bin_sol = get_bin_sol(best_dict["gbest_sol"])

        for var in range(Nvar):
            for bit in range(Nbit):
                # a, b要從尾巴取
                a = Qgbest_ab[var][Nbit - 1 - bit][0]
                b = Qgbest_ab[var][Nbit - 1 - bit][1]

                if (int(gbest_bin_sol[var][bit]) == 0 and int(curr_bin_sol[var][bit]) == 0):
                    # Let delta_theta = 0.01 * pi
                    delta_theta = theta_2
                    if ((a*b).real > 0):
                        God_list[Qbest_idx][var].ry(
                            (-2)*delta_theta, Nbit - 1 - bit)

                    elif (a*b).real < 0:
                        God_list[Qbest_idx][var].ry(
                            2*delta_theta, Nbit - 1 - bit)
                    elif (a.real == 0 or b.real == 0):
                        God_list[Qbest_idx][var].ry(
                            2*delta_theta, Nbit - 1 - bit)

                if (int(gbest_bin_sol[var][bit] == 1) and int(curr_bin_sol[var][bit] == 0)):
                    # Let delta_theta = 0.01 * pi
                    delta_theta = theta_2
                    if (a*b) > 0:
                        God_list[Qbest_idx][var].ry(-2 *
                                                    delta_theta, Nbit - 1 - bit)
                    elif (a*b) < 0:
                        God_list[Qbest_idx][var].ry(
                            2*delta_theta, Nbit - 1 - bit)
                    elif (a == 0 or b == 0):
                        God_list[Qbest_idx][var].ry(
                            2*delta_theta, Nbit - 1 - bit)

                if (int(gbest_bin_sol[var][bit] == 0) and int(curr_bin_sol[var][bit] == 1)):
                    # Let delta_theta = 0.01 * pi
                    delta_theta = theta_2
                    if (a*b).real > 0:
                        God_list[Qbest_idx][var].ry(
                            2*delta_theta, Nbit - 1 - bit)
                    elif (a*b).real < 0:
                        God_list[Qbest_idx][var].ry(-2 *
                                                    delta_theta, Nbit - 1 - bit)
                    elif (a.real == 0 or b.real == 0):
                        God_list[Qbest_idx][var].ry(
                            2*delta_theta, Nbit - 1 - bit)

                if (int(gbest_bin_sol[var][bit] == 1) and int(curr_bin_sol[var][bit] == 1)):
                    # Let delta_theta = 0.01 * pi
                    delta_theta = theta_2
                    if (a*b).real > 0:
                        God_list[Qbest_idx][var].ry(
                            2*delta_theta, Nbit - 1 - bit)
                    elif (a*b).real < 0:
                        God_list[Qbest_idx][var].ry(-2 *
                                                    delta_theta, Nbit - 1 - bit)
                    elif (a.real == 0 or b.real == 0):
                        God_list[Qbest_idx][var].ry(
                            2*delta_theta, Nbit - 1 - bit)
        # update the best_dict[gbest]
        best_dict["gbest_fitness"] = sol_fitness
        best_dict["gbest_sol"] = sol_list

    # if sol_fitness worse than best_dict["gbest_fitness"]
    elif sol_fitness < best_dict["gbest_fitness"]:
        # get the ai, bi of God_list[Qbest_idx]
        Qgbest_ab = get_ab_list(God_list[Qbest_idx])

        # get the binary sol_list to compare
        gbest_bin_sol = get_bin_sol(best_dict["gbest_sol"])
        for var in range(Nvar):
            for bit in range(Nbit):
                a = Qgbest_ab[var][Nbit - 1 - bit][0]
                b = Qgbest_ab[var][Nbit - 1 - bit][1]
                if (int(gbest_bin_sol[var][bit] == 0) and int(curr_bin_sol[var][bit] == 0)):
                    # Let delta_theta = 0.01 * pi
                    delta_theta = theta_2
                    if (a*b).real > 0:
                        God_list[Qbest_idx][var].ry(-2 *
                                                    delta_theta, Nbit - 1 - bit)
                    elif (a*b).real < 0:
                        God_list[Qbest_idx][var].ry(
                            2*delta_theta, Nbit - 1 - bit)
                    elif (a.real == 0 or b.real == 0):
                        God_list[Qbest_idx][var].ry(
                            2*delta_theta, Nbit - 1 - bit)

                if int(gbest_bin_sol[var][bit] == 1) and int(curr_bin_sol[var][bit] == 0):
                    # Let delta_theta = 0.08 * pi
                    delta_theta = theta_1
                    if (a*b).real > 0:
                        God_list[Qbest_idx][var].ry(-2 *
                                                    delta_theta, Nbit - 1 - bit)
                    elif (a*b).real < 0:
                        God_list[Qbest_idx][var].ry(
                            2*delta_theta, Nbit - 1 - bit)
                    elif (a.real == 0 or b.real == 0):
                        God_list[Qbest_idx][var].ry(
                            2*delta_theta, Nbit - 1 - bit)

                if (int(gbest_bin_sol[var][bit]) == 0 and int(curr_bin_sol[var][bit]) == 1):
                    # Let delta_theta = 0.01 * pi
                    delta_theta = theta_1
                    if (a*b).real > 0:
                        God_list[Qbest_idx][var].ry(
                            2*delta_theta, Nbit - 1 - bit)
                    elif (a*b).real < 0:
                        God_list[Qbest_idx][var].ry(-2 *
                                                    delta_theta, Nbit - 1 - bit)
                    elif (a == 0 or b == 0):
                        God_list[Qbest_idx][var].ry(
                            2*delta_theta, Nbit - 1 - bit)

                if (int(gbest_bin_sol[var][bit]) == 1 and int(curr_bin_sol[var][bit]) == 1):
                    # Let delta_theta = 0.01 * pi
                    delta_theta = theta_2
                    if (a*b).real > 0:
                        God_list[Qbest_idx][var].ry(
                            2*delta_theta, Nbit - 1 - bit)
                    elif (a*b).real < 0:
                        God_list[Qbest_idx][var].ry(-2 *
                                                    delta_theta, Nbit - 1 - bit)
                    elif (a.real == 0 or b.real == 0):
                        God_list[Qbest_idx][var].ry(
                            2*delta_theta, Nbit - 1 - bit)
    # Compare for pbest_solution
    if sol_fitness >= best_dict["pbest_fitness"][sol]:
        print(f'Get Better pbest sol_{sol}: {sol_fitness}')
        Qpbest_ab = get_ab_list(God_list[sol])
        pbest_bin_sol = get_bin_sol(best_dict["pbest_list"][sol])

        for var in range(Nvar):
            for bit in range(Nbit):
                a = Qpbest_ab[var][Nbit - 1 - bit][0]
                b = Qpbest_ab[var][Nbit - 1 - bit][1]
                if int(pbest_bin_sol[var][bit] == 0) and int(curr_bin_sol[var][bit] == 0):
                    # Let delta_theta = 0.01 * pi
                    delta_theta = theta_2
                    if (a*b).real > 0:
                        God_list[sol][var].ry(-2*delta_theta, Nbit - 1 - bit)
                    elif (a*b).real < 0:
                        God_list[sol][var].ry(2*delta_theta, Nbit - 1 - bit)
                    elif (a.real == 0 or b.real == 0):
                        God_list[sol][var].ry(2*delta_theta, Nbit - 1 - bit)

                if (int(pbest_bin_sol[var][bit] == 1) and int(curr_bin_sol[var][bit] == 0)):
                    # Let delta_theta = 0.01 * pi
                    delta_theta = theta_2
                    if (a*b).real > 0:
                        God_list[sol][var].ry(-2*delta_theta, Nbit - 1 - bit)
                    elif (a*b).real < 0:
                        God_list[sol][var].ry(2*delta_theta, Nbit - 1 - bit)
                    elif (a.real == 0 or b.real == 0):
                        God_list[sol][var].ry(2*delta_theta, Nbit - 1 - bit)

                if (int(pbest_bin_sol[var][bit] == 0) and int(curr_bin_sol[var][bit]) == 1):
                    # Let delta_theta = 0.01 * pi
                    delta_theta = theta_2
                    if (a*b).real > 0:
                        God_list[sol][var].ry(2*delta_theta, Nbit - 1 - bit)
                    elif (a*b).real < 0:
                        God_list[sol][var].ry(-2*delta_theta, Nbit - 1 - bit)
                    elif (a.real == 0 or b.real == 0):
                        God_list[sol][var].ry(2*delta_theta, Nbit - 1 - bit)

                if (int(pbest_bin_sol[var][bit] == 1) and int(curr_bin_sol[var][bit] == 1)):
                    # Let delta_theta = 0.01 * pi
                    delta_theta = theta_2
                    if ((a*b).real > 0):
                        God_list[sol][var].ry(2*delta_theta, Nbit - 1 - bit)
                    elif ((a*b).real < 0):
                        God_list[sol][var].ry(-2*delta_theta, Nbit - 1 - bit)
                    elif ((a*b).real == 0):
                        God_list[sol][var].ry(2*delta_theta, Nbit - 1 - bit)

        best_dict["pbest_fitness"][sol] = sol_fitness
        best_dict["pbest_list"][sol] = sol_list

    elif sol_fitness < best_dict["pbest_fitness"][sol]:
        # get the ai, bi of God_list[Qsol]
        Qpbest_ab = get_ab_list(God_list[sol])
        # get the binary sol_list to compare
        pbest_bin_sol = get_bin_sol(best_dict["pbest_list"][sol])
        for var in range(Nvar):
            for bit in range(Nbit):
                a = Qpbest_ab[var][Nbit - 1 - bit][0]
                b = Qpbest_ab[var][Nbit - 1 - bit][1]
                if (int(pbest_bin_sol[var][bit] == 0) and int(curr_bin_sol[var][bit] == 0)):
                    # Let delta_theta = 0.01 * pi
                    delta_theta = theta_2
                    if (a*b).real > 0:
                        God_list[sol][var].ry(-2*delta_theta, Nbit - 1 - bit)
                    elif (a*b).real < 0:
                        God_list[sol][var].ry(2*delta_theta, Nbit - 1 - bit)
                    elif (a == 0 or b == 0):
                        God_list[sol][var].ry(2*delta_theta, Nbit - 1 - bit)

                if int(pbest_bin_sol[var][bit] == 1) and int(curr_bin_sol[var][bit] == 0):
                    # Let delta_theta = 0.01 * pi
                    delta_theta = theta_1
                    if (a*b).real > 0:
                        God_list[sol][var].ry(-2*delta_theta, Nbit - 1 - bit)
                    elif (a*b).real < 0:
                        God_list[sol][var].ry(2*delta_theta, Nbit - 1 - bit)
                    elif (a.real == 0 or b.real == 0):
                        God_list[sol][var].ry(2*delta_theta, Nbit - 1 - bit)

                if (int(pbest_bin_sol[var][bit] == 0) and int(curr_bin_sol[var][bit]) == 1):
                    # Let delta_theta = 0.01 * pi
                    delta_theta = theta_1
                    if (a*b).real > 0:
                        God_list[sol][var].ry(2*delta_theta, Nbit - 1 - bit)
                    elif (a*b).real < 0:
                        God_list[sol][var].ry(-2*delta_theta, Nbit - 1 - bit)
                    elif (a.real == 0 or b.real == 0):
                        God_list[sol][var].ry(2*delta_theta, Nbit - 1 - bit)

                if (int(pbest_bin_sol[var][bit]) == 1 and int(curr_bin_sol[var][bit]) == 1):
                    # Let delta_theta = 0.01 * pi
                    delta_theta = theta_2
                    if ((a*b).real > 0):
                        God_list[sol][var].ry(2*delta_theta, Nbit - 1 - bit)
                    elif ((a*b).real < 0):
                        God_list[sol][var].ry(-2*delta_theta, Nbit - 1 - bit)
                    elif ((a*b).real == 0):
                        God_list[sol][var].ry(2*delta_theta, Nbit - 1 - bit)

    return God_list


#################### main function ####################
if __name__ == '__main__':
    time_start = time.time()  # count the time
    God_list = God_init()
    best_dict = best_dict_init()
    for sol in range(Nsol):
        sol_list = sol_update(sol)
        sol_fitness = FIT_cal(sol_list)
        print(f"Initialize sol_{sol}:", sol_list)
        print("sol_fitness: ", sol_fitness)
        God_update(sol, sol_list, sol_fitness)
        gen_info = ["Init"]            # Draw the plot
        fitness_info = [sol_fitness]   # Draw the plot

    for gen in range(Ngen):
        print(f'=============== Generation_{gen} ===============')
        for sol in range(Nsol):
            print(f'----- sol_{sol} -----')
            sol_list = sol_update(sol)
            sol_fitness = FIT_cal(sol_list)
            print('sol_fitness: {}'.format(sol_fitness))
            print('sol_list: ', sol_list)
            God_update(sol, sol_list, sol_fitness)

        print('Global best fitness: {}'.format(best_dict['gbest_fitness']))
        print('Global best solution: {}'.format(best_dict['gbest_sol']))

    time_end = time.time()  # 結束計時
    time_c = time_end - time_start  # 執行所花時間
    print('time cost', time_c, 's')
    print('Global best fitness: {}'.format(best_dict['gbest_fitness']))
    print('Global best solution: {}'.format(best_dict['gbest_sol']))

#################### main function ####################
