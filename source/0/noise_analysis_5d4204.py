# https://github.com/Danny-sc/QFT_multiplier/blob/057b8450b8e21e67eeb6a2e444eb6fe1e950b052/Noise_analysis.py
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import depolarizing_error
from qft_multiplier import qft_multiplier
from qiskit import(QuantumCircuit, execute, Aer)
import matplotlib.pyplot as plt
import random

"""
The code below 
1. runs `qft_multiplier` with a noise added, in the form of a 
depolarizing error channel, for random pairs of a and b with 2,3,4,5 bits. For
each pair, the noisy circuit is run 1000 times and for each time the proportion
of correct shots is stored.
2. makes a plot for each number of bits 2,3,4,5, with a boxplot for 
each value of the parameter p in the depolarizing channel. Each boxplot shows 
the distribution of the proportion of correct shots for a specific value of p.
3. makes a plot with the average proportion of correct shots for different 
values of the parameter p and for numbers a and b with 2,3,4,5 bits.
"""

# Values of interest for the parameter p in the depolarizing channel 
prob = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
# For each value of p, create a list of four lists. The first of the four lists
# will contain statistics regarding the multiplication of 2-bit numbers, the 
# second will contain statistics regarding the multiplication of 3-bit numbers, etc
prop_001 = [[],[],[],[]]
prop_0025 = [[],[],[],[]]
prop_005 = [[],[],[],[]]
prop_0075 = [[],[],[],[]]
prop_01 = [[],[],[],[]]
prop_015 = [[],[],[],[]]
prop_02 = [[],[],[],[]]
prop =[prop_001, prop_0025, prop_005, prop_0075, prop_01, prop_015, prop_02]
for p in range(len(prob)):
    # Create the depolarizing error channel,
    # epsilon(P) = (1-P)rho + (P/3)(XrhoX + YrhoY + ZrhoZ)
    depo_err_chan = depolarizing_error((4*prob[p])/3, 1)
    # Create the noise model to be used during execution
    noise_model = NoiseModel()
    # Measurement error is applied to measurements
    noise_model.add_all_qubit_quantum_error(depo_err_chan, "measure") 
    # Sample pairs of integers for a and b with 2,3,4,5 bits
    for i in range(0,4):
        lst = range(2**(i+1), 2**(i+2))
        output = []
        for k in range(0,len(lst)):
            for j in range(0,len(lst)):
                if (k<=j):
                    output.append((lst[k],lst[j]))
        if i==2: 
            output = random.sample(output, 16)
        if i==3:
            output = random.sample(output, 32)
        if i==4: 
            output = random.sample(output, 64) 
        # Run the noise model on each pair 1000 times and store the proportion
        # of shots with the correct value in prop[p][i]
        for s in output:
            qc = qft_multiplier(s[0],s[1])[0]
            counts = execute(qc, Aer.get_backend('qasm_simulator'), 
                  noise_model=noise_model,
                  shots=1000).result().get_counts()
            n_bit = max(s[0].bit_length(), s[1].bit_length())
            binary_format = '0'+str(2*n_bit)+'b'
            proportion = counts[format(s[0]*s[1], binary_format)]/1000
            prop[p][i].append(proportion)
plt.boxplot([prop[0][0],prop[1][0],prop[2][0],prop[3][0],prop[4][0],prop[5][0],prop[6][0]])
plt.xticks([1, 2, 3, 4, 5, 6, 7], ['0.01', '0.025', '0.05', '0.075', '0.1', '0.15', '0.2'])
plt.title("Proportion of correct shots when multiplying two %i-bit numbers" %(0+2), fontsize = 10)
plt.xlabel("Parameter p in the depolarizing channel", fontsize=8)
plt.show()
plt.boxplot([prop[0][1],prop[1][1],prop[2][1],prop[3][1],prop[4][1],prop[5][1],prop[6][1]])
plt.xticks([1, 2, 3, 4, 5, 6, 7], ['0.01', '0.025', '0.05', '0.075', '0.1', '0.15', '0.2'])
plt.title("Proportion of correct shots when multiplying two %i-bit numbers" %(1+2), fontsize = 10)
plt.xlabel("Parameter p in the depolarizing channel", fontsize=8)
plt.show()
plt.boxplot([prop[0][2],prop[1][2],prop[2][2],prop[3][2],prop[4][2],prop[5][2],prop[6][2]])
plt.xticks([1, 2, 3, 4, 5, 6, 7], ['0.01', '0.025', '0.05', '0.075', '0.1', '0.15', '0.2'])
plt.title("Proportion of correct shots when multiplying two %i-bit numbers" %(2+2), fontsize = 10)
plt.xlabel("Parameter p in the depolarizing channel", fontsize=8)
plt.show()
plt.boxplot([prop[0][3],prop[1][3],prop[2][3],prop[3][3],prop[4][3],prop[5][3],prop[6][3]])
plt.xticks([1, 2, 3, 4, 5, 6, 7], ['0.01', '0.025', '0.05', '0.075', '0.1', '0.15', '0.2'])
plt.title("Proportion of correct shots when multiplying two %i-bit numbers" %(3+2), fontsize = 10)
plt.xlabel("Parameter p in the depolarizing channel", fontsize=8)
plt.show()
# Make a plot with a line for each number of bits (2,3,4,5) needed for a and b.
# Each line represents the average proportion of correct shots when using
# `qft_multiplier` with noise, for different values of the parameter p.
plt.plot([0.01,0.025,0.05,0.075, 0.1, 0.15, 0.2], [sum(prop[0][0])/len(prop[0][0]), sum(prop[1][0])/len(prop[1][0]),sum(prop[2][0])/len(prop[2][0]),sum(prop[3][0])/len(prop[3][0]),sum(prop[4][0])/len(prop[4][0]),sum(prop[5][0])/len(prop[5][0]),sum(prop[6][0])/len(prop[6][0])], marker = '.', markersize = 8, label = '2 qubits')
plt.plot([0.01,0.025,0.05,0.075, 0.1, 0.15, 0.2], [sum(prop[0][1])/len(prop[0][1]), sum(prop[1][1])/len(prop[1][1]),sum(prop[2][1])/len(prop[2][1]),sum(prop[3][1])/len(prop[3][1]),sum(prop[4][1])/len(prop[4][1]),sum(prop[5][1])/len(prop[5][1]),sum(prop[6][1])/len(prop[6][1])], marker = '.', markersize = 8, label = '3 qubits')
plt.plot([0.01,0.025,0.05,0.075, 0.1, 0.15, 0.2], [sum(prop[0][2])/len(prop[0][2]), sum(prop[1][2])/len(prop[1][2]),sum(prop[2][2])/len(prop[2][2]),sum(prop[3][2])/len(prop[3][2]),sum(prop[4][2])/len(prop[4][2]),sum(prop[5][2])/len(prop[5][2]),sum(prop[6][2])/len(prop[6][2])], marker = '.', markersize = 8, label = '4 qubits')
plt.plot([0.01,0.025,0.05,0.075, 0.1, 0.15, 0.2], [sum(prop[0][3])/len(prop[0][3]), sum(prop[1][3])/len(prop[1][3]),sum(prop[2][3])/len(prop[2][3]),sum(prop[3][3])/len(prop[3][3]),sum(prop[4][3])/len(prop[4][3]),sum(prop[5][3])/len(prop[5][3]),sum(prop[6][3])/len(prop[6][3])], marker = '.', markersize = 8, label = '5 qubits')
plt.legend()
plt.title("Average proportion of correct shots when multiplying two numbers", fontsize=10)
plt.xlabel("Parameter p in the depolarizing channel")
plt.show()
