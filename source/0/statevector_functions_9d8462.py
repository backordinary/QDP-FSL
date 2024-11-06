# https://github.com/smitchaudhary/QAOA-MaxCut/blob/9a91a85cc16f49fa986ce1761e44b5edc132c676/statevector_functions.py
from qiskit.providers.aer import StatevectorSimulator
from qiskit import execute, QuantumCircuit
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt


def circuit_ansatz(G, gamma, beta, p=1): #gamma and beta are p-lists
    n = len(G.nodes())
    E = G.edges()

    QAOA = QuantumCircuit(n, n)
    QAOA.h(range(n))
    QAOA.barrier()
    for i in range(p):
        for edge in E:
            k = edge[0]
            l = edge[1]
            QAOA.cu1(-2*gamma[i], k, l) #Controlled-Z gate with a -2*gamma phase
            QAOA.u1(gamma[i], k) #Rotation of gamma around the z axis
            QAOA.u1(gamma[i], l)
        QAOA.barrier()
        QAOA.rx(2*beta[i], range(n)) #X rotation
        QAOA.barrier()

    return QAOA

def execute_circuit(G, gamma, beta, p = 1): #Returns same as earlier, the counts
    n = len(G.nodes())
    QAOA = circuit_ansatz(G, gamma, beta, p = p)
    result = execute(QAOA, backend=StatevectorSimulator()).result()
    statevector = result.get_statevector(QAOA)
    probabilities = ([abs(i)**2 for i in statevector])
    state_dictionary = {bin(i)[2:].zfill(n) : probabilities[i] for i in range(len(probabilities))}
    return state_dictionary

def cost_function_C(x,G): #x is a list
    E = G.edges()
    C = 0
    for vertice in E:
        e1 = vertice[0]
        e2 = vertice[1]
        w = G[e1][e2]['weight']
        C = C + w*x[e1]*(1-x[e2]) + w*x[e2]*(1-x[e1])
    return C

def get_expectval(state_dictionary, G): #state_dict = dictionary holding 'state' and probabilities
    avr_cost = 0
    for state in list(state_dictionary.keys()):
        x = [int(bit_num) for bit_num in list(state)] #the bit string is saved as a list
        cost_x = cost_function_C(x,G)
        probablity = state_dictionary[state]
        avr_cost += probablity*cost_x
    return avr_cost

def get_solution(state_dict, G): #takes as the solution the state with the highest cost within all the measured states
    solution_cost = 0
    for state in list(state_dict.keys()):
        x = [int(bit_num) for bit_num in list(state)]
        cost_x = cost_function_C(x,G)
        if cost_x > solution_cost:
            solution = x
            solution_cost = cost_x
    print(f'The solution is the state {solution} with a cost value of {solution_cost}')
    return solution, solution_cost

def cost_function(parameters, G, p = 1):
    gamma = parameters[0:p]
    beta = parameters[p:]
    state_dictionary = execute_circuit(G, gamma, beta, p=p)
    avr_cost = get_expectval(state_dictionary, G)
    return -avr_cost

def plot_grid(G, p):
    if p==1:
        gamma_max, beta_max = 2*np.pi, 2*np.pi #gamma and beta grid between 2pi and 2pi
        steps = 20
        gamma_list, beta_list = np.linspace(0,beta_max,steps), np.linspace(0,gamma_max,steps)
        avr_cost_grid = np.zeros((len(gamma_list),len(beta_list)))

        for i in range(len(gamma_list)):
            gamma = gamma_list[i]
            for j in range(len(beta_list)):
                beta = beta_list[j]
                state_dictionary = execute_circuit(G,gamma,beta,p=p)
                avr_cost = get_expectval(state_dictionary, G)
                avr_cost_grid[i,j] = avr_cost

        f = pl.figure(facecolor='w', edgecolor='k')
        pl.imshow(avr_cost_grid, interpolation = 'bicubic', extent = [0,beta_max,0,gamma_max])
        pl.title('Ideal parameter grid', fontsize=8)
        pl.colorbar()
        pl.xlabel(r'$\\beta$')
        pl.ylabel(r'$\\gamma$')
        pl.show()
        #pl.savefig('')

    else:
        print(f'Can not plot for {2*p} parameters.')
        #print(f"0 - don't plot")
        #for i in range(p):
        #    print(f'{i+1} - Plot for $\gamma_{i+1}$ and $\\beta_{i+1}$ ')
        #plot_for_iteration = eval(input())
        #print("For the following values of other parameters:")
        #for i in range(p):
        #    if i != plot_for_iteration - 1:

    return


def show_amplitudes(G, gamma, beta, p=1):

    n = len(G.nodes())
    E = G.edges()

    QAOA = QuantumCircuit(n, n)
    QAOA.h(range(n))
    QAOA.barrier()

    fig = plt.figure()
    bar_graph = plt.bar(range(2**n), [1/(2**n) for i in range(2**n)], align = 'center')
    x_ticks = [bin(i)[2:].zfill(n) for i in range(2**n)]
    plt.xticks(ticks = range(2**n), labels = x_ticks, rotation = 60)
    plt.ylim(0,0.75)
    plt.title(f'Initial State')
    fig.canvas.draw()
    plt.pause(1)
    #plt.savefig(f'Initial State bad')

    for i in range(p):
        for edge in E:
            k = edge[0]
            l = edge[1]
            QAOA.cu1(-2*gamma[i], k, l) #Controlled-Z gate with a -2*gamma phase
            QAOA.u1(gamma[i], k) #Rotation of gamma around the z axis
            QAOA.u1(gamma[i], l)

        statevector = execute(QAOA, backend = StatevectorSimulator()).result().get_statevector(QAOA)
        probabilities = ([abs(i)**2 for i in statevector])
        state_dict = {bin(i)[2:].zfill(n) : probabilities[i] for i in range(len(probabilities))}

        for rectangle, probs in zip(bar_graph, probabilities):
            rectangle.set_height(probs)
        plt.title(f'Cost: {get_expectval(state_dict, G)}\n Iteration {i + 1}, after applying $U_C$\n $\gamma$ = {gamma[i]}')
        fig.canvas.draw()
        #plt.savefig(f'{i} gamma bad')
        plt.pause(0.5)

        QAOA.barrier()
        QAOA.rx(2*beta[i], range(n)) #X rotation
        statevector = execute(QAOA, backend = StatevectorSimulator()).result().get_statevector(QAOA)
        probabilities = ([abs(i)**2 for i in statevector])
        state_dict = {bin(i)[2:].zfill(n) : probabilities[i] for i in range(len(probabilities))}

        for rectangle, probs in zip(bar_graph, probabilities):
            rectangle.set_height(probs)
        plt.title(f'Cost: {get_expectval(state_dict, G)}\n Iteration {i + 1}, after applying $U_B$\n $\\beta$ = {beta[i]}')
        fig.canvas.draw()
        #plt.savefig(f'{i} beta bad')
        plt.pause(0.5)

        QAOA.barrier()
    plt.show()
