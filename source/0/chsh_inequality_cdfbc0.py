# https://github.com/knight9999/chsh_inequality/blob/5a275eb66bd986fd8b1809d2ee78ae15303f4188/chsh_inequality.py
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.providers.aer import QasmSimulator

def makeInitial(theta):
    circuit = QuantumCircuit(2, 2)
    circuit.ry(theta, 0)
    circuit.cx(0, 1)
    return circuit

def makeMeasure(initialState, angleAlice, angleBob):
    circuit = initialState
    if angleAlice != 0:
        circuit.rx(angleAlice, 0) # Alice
    if angleBob != 0:
        circuit.rx(angleBob, 1) # Bob
    circuit.measure([0,1], [0,1])
    return circuit

def makeCircuit(theta, angleAlice, angleBob):
    circuit = makeInitial(theta)
    circuit.barrier()
    makeMeasure(circuit, angleAlice, angleBob)
    return circuit

def experimentSigmaSigma(theta_val, shots, angleAlice, angleBob):
    theta = Parameter('θ')
    circuit = makeCircuit(theta, angleAlice, angleBob)
    circuit = circuit.bind_parameters({theta: theta_val})
    simulator = QasmSimulator()
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=shots)
    result = job.result()
    dict = result.get_counts(compiled_circuit)
    value = dict.get('00', 0) + dict.get('11', 0)  - dict.get('01', 0) - dict.get('10', 0)
    return value/shots

def experimentD(theta_val, shots):
    term1 = experimentSigmaSigma(theta_val, shots, -np.pi/2, -np.pi/4)
    term2 = experimentSigmaSigma(theta_val, shots, -np.pi/2, np.pi/4)
    term3 = experimentSigmaSigma(theta_val, shots, 0, -np.pi/4)
    term4 = experimentSigmaSigma(theta_val, shots, 0, np.pi/4)
    D = term1 - term2 + term3 + term4
    return {"term1": term1, "term2": term2, "term3": term3, "D": D}

def main():
    shots = 10000
    times = 40
    xmin = -np.pi
    xmax = np.pi
    ymin = 0.0
    ymax = 2.0 * np.sqrt(2.0)
    xs = np.linspace(xmin, xmax, times)
    ys = []
    for x in xs:
        result = experimentD(x, shots)
        ys.append(result["D"])

    for (x,y) in zip (xs,ys):
        print(x,y)

    fig = plt.figure()

    plt.plot(xs, ys, color='r', marker='.', ls='-', label='D')
    plt.hlines([2.0], xmin, xmax, "blue", linestyles='dashed') 
    plt.vlines([-1/2*np.pi], ymin, ymax, "blue", linestyles="-")
    plt.vlines([-7/8*np.pi, -np.pi/8], ymin, ymax, "gray", linestyles='dashed') 
    plt.legend()
    plt.xlabel('θ')
    plt.ylabel('D')
    plt.text( -1/2*np.pi+0.1, 0.6, 'θ=-π/2', ha='left', va='center')
    plt.text( -7/8*np.pi+0.1, 1.1, 'θ=-7π/8', ha='left', va='center')
    plt.text( -1/8*np.pi+0.1, 0.8, 'θ=-π/8', ha='left', va='center')
    plt.title('CHSH Inequality')

    plt.show()
    fig.savefig("output/chsh_inequality.png")

if __name__ == "__main__":
    main()
