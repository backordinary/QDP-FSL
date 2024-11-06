# https://github.com/QuantumCyberW0lf/Quantum-Algorithms/blob/b9384cdb4926cb4978d0319ecae9acd9f01aaccf/toffoli.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys, time
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.tools.visualization import circuit_drawer

def construct_circuit_from_toffofi_gate(n_qubit:int)-> object:
    control_qubits = QuantumRegister(3,'control qubits')
    target_qubits = QuantumRegister(1,'target qubits')
    hilfsqubits = QuantumRegister(n_qubit,'hilfsqubits')
    circ = QuantumCircuit(control_qubits,hilfsqubits,target_qubits)

    circ.ccx(control_qubits[0],control_qubits[1],hilfsqubits[0])
    for i in range(2,n_qubit):
        circ.ccx(control_qubits[2],hilfsqubits[i-2],hilfsqubits[i-1])
    circ.cx(hilfsqubits[n_qubit-2],target_qubits[0])
    for i in range(n_qubit,1,-1):
        circ.ccx(control_qubits[2],hilfsqubits[i-2],hilfsqubits[i-1])
    circ.ccx(control_qubits[0],control_qubits[1],hilfsqubits[0])
    return circ

def simulation(circ:object)->None:
    circuit_drawer(circ)

def save_fig(circ:object)->None:
    figure = circ.draw(output="mpl")
    figure.savefig(fname="blatt_3_aufgabe4b_altenschmidt")

def main()->None:
    des="Hausaufgabe 4b Quantenalgorithmen SoSe20"
    epi="Built by Qu@ntumCyberW01f/Qu@ntumH@ck3r = Thi Altenschmidt"
    parser=argparse.ArgumentParser(description=des,epilog=epi)
    parser.add_argument("--num","-n",action="store",type=int,dest="num",
            help="Specify a positiv integer number",required=True)
    given_args = parser.parse_args()
    num = given_args.num
    if (num == None):
        print("[!] Usage: {} -h/--help for more information.".format(sys.argv[0]))
        sys.exit(-1)
    try:
        num = int(num)
    except ValueError as val_err:
        print("[-] Illegal input: {}".format(given_args.num))
        sys.exit(-1)
    if (num < 2):
        print("[-] input number must greater equal 2.")
        sys.exit(-1)

    circ = construct_circuit_from_toffofi_gate(num)
    print("Visulizing the quantum circuit from Toffofi Quantum Gate...")
    time.sleep(0.5)
    simulation(circ)
    time.sleep(0.5)
    print("Saving figure...")
    time.sleep(0.5)
    save_fig(circ)

if __name__ == "__main__":
    main()


