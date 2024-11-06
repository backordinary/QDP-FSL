# https://github.com/QuantumQoder/qosf-mentorship/blob/60188c64b5a70f6288af55426c1cf79acb95743a/task_2.py
"""
The circuit uses the repition code analogy and implements entanglement syndrome measurement to detect, analyse and rectify the errors.
At first the circuit creates a bell's pair, and then encodes "error gates" on each of the two code qubits. Then, two rounds of syndrome measurement is applied on the circuit. The first round detects the possibility of error from the link qubit, and then applies a bit-flip to one of the code qubits to eradicate the error from the circuit. Based on the output produced by the measurement operator, one can indentiy and analuse the errors in one of the code qubits.
The code produces all the possible choices of the "error gates" on each of the code qubits.

Analysing the output of the syndrome measurement:-
The output produced is a python dictionary which contains bit strings in the form '00 00' as its keys, and the values of those keys represent the number of shots for which the key occured.
1. The 00 and 11 to the far right of the key are the logical readouts, representing the logical 0 and logical 1, respectively.
2. The 00, 01 or 11 to the far left of the dictionary key indicates the possible presence of errors and their successful correction.
	2.1. 00 indicates that the code was unable to detect any error in the two rounds of the syndrome measurement.
	2.2. 01 indicates that the code detected a possible error in one of the code qubits, and then was able to apply correction to the circuit.
	2.3. 10 indicates that the code didn't detect any error in the first round, but some kind of unknown error was introduced into the circuit before the second measurement.
	2.4. 11 inidcates that the code detected an error in one of the code qubits before the first round, but was unable to rectify the error in the circuit.
	
Note:- Please notice that the circuit is unable to detect errors if both the code qubits has been compromised. The circuit works best only when one of the code qubits has been affected.

Following is the list of outputs the code produces upon succesful compilation.

			========= circuit[ 0 ][ 0 ]=========

                  ┌───┐     ┌───┐ ░                      ░                ┌─┐»
    code_qubit_0: ┤ H ├──■──┤ I ├─░───■──────────────────░────────■───────┤M├»
                  └───┘┌─┴─┐├───┤ ░   │           ┌───┐  ░        │       └╥┘»
    code_qubit_1: ─────┤ X ├┤ I ├─░───┼────■──────┤ X ├──░────────┼────■───╫─»
                       └───┘└───┘ ░ ┌─┴─┐┌─┴─┐┌─┐ └─┬─┘  ░      ┌─┴─┐┌─┴─┐ ║ »
    link_qubit_0: ────────────────░─┤ X ├┤ X ├┤M├───┼────░──|0>─┤ X ├┤ X ├─╫─»
                                  ░ └───┘└───┘└╥┘   │    ░      └───┘└───┘ ║ »
      code_bit: 2/═════════════════════════════╬════╪══════════════════════╩═»
                                               ║ ┌──┴──┐                   0 »
round_link_bit: 2/═════════════════════════════╩═╡ = 1 ╞═════════════════════»
                                               0 └─────┘                     »
«                        
«    code_qubit_0: ──────
«                     ┌─┐
«    code_qubit_1: ───┤M├
«                  ┌─┐└╥┘
«    link_qubit_0: ┤M├─╫─
«                  └╥┘ ║ 
«      code_bit: 2/═╬══╩═
«                   ║  1 
«round_link_bit: 2/═╩════
«                   1    
{'00 00': 492, '00 11': 532}

			========= circuit[ 0 ][ 1 ]=========

                  ┌───┐     ┌───┐ ░                      ░                ┌─┐»
    code_qubit_0: ┤ H ├──■──┤ I ├─░───■──────────────────░────────■───────┤M├»
                  └───┘┌─┴─┐├───┤ ░   │           ┌───┐  ░        │       └╥┘»
    code_qubit_1: ─────┤ X ├┤ X ├─░───┼────■──────┤ X ├──░────────┼────■───╫─»
                       └───┘└───┘ ░ ┌─┴─┐┌─┴─┐┌─┐ └─┬─┘  ░      ┌─┴─┐┌─┴─┐ ║ »
    link_qubit_0: ────────────────░─┤ X ├┤ X ├┤M├───┼────░──|0>─┤ X ├┤ X ├─╫─»
                                  ░ └───┘└───┘└╥┘   │    ░      └───┘└───┘ ║ »
      code_bit: 2/═════════════════════════════╬════╪══════════════════════╩═»
                                               ║ ┌──┴──┐                   0 »
round_link_bit: 2/═════════════════════════════╩═╡ = 1 ╞═════════════════════»
                                               0 └─────┘                     »
«                        
«    code_qubit_0: ──────
«                     ┌─┐
«    code_qubit_1: ───┤M├
«                  ┌─┐└╥┘
«    link_qubit_0: ┤M├─╫─
«                  └╥┘ ║ 
«      code_bit: 2/═╬══╩═
«                   ║  1 
«round_link_bit: 2/═╩════
«                   1    
{'01 00': 504, '01 11': 520}

			========= circuit[ 0 ][ 2 ]=========

                  ┌───┐     ┌───┐ ░                      ░                ┌─┐»
    code_qubit_0: ┤ H ├──■──┤ I ├─░───■──────────────────░────────■───────┤M├»
                  └───┘┌─┴─┐├───┤ ░   │           ┌───┐  ░        │       └╥┘»
    code_qubit_1: ─────┤ X ├┤ Z ├─░───┼────■──────┤ X ├──░────────┼────■───╫─»
                       └───┘└───┘ ░ ┌─┴─┐┌─┴─┐┌─┐ └─┬─┘  ░      ┌─┴─┐┌─┴─┐ ║ »
    link_qubit_0: ────────────────░─┤ X ├┤ X ├┤M├───┼────░──|0>─┤ X ├┤ X ├─╫─»
                                  ░ └───┘└───┘└╥┘   │    ░      └───┘└───┘ ║ »
      code_bit: 2/═════════════════════════════╬════╪══════════════════════╩═»
                                               ║ ┌──┴──┐                   0 »
round_link_bit: 2/═════════════════════════════╩═╡ = 1 ╞═════════════════════»
                                               0 └─────┘                     »
«                        
«    code_qubit_0: ──────
«                     ┌─┐
«    code_qubit_1: ───┤M├
«                  ┌─┐└╥┘
«    link_qubit_0: ┤M├─╫─
«                  └╥┘ ║ 
«      code_bit: 2/═╬══╩═
«                   ║  1 
«round_link_bit: 2/═╩════
«                   1    
{'00 00': 538, '00 11': 486}

			========= circuit[ 1 ][ 0 ]=========

                  ┌───┐     ┌───┐ ░                      ░                ┌─┐»
    code_qubit_0: ┤ H ├──■──┤ X ├─░───■──────────────────░────────■───────┤M├»
                  └───┘┌─┴─┐├───┤ ░   │           ┌───┐  ░        │       └╥┘»
    code_qubit_1: ─────┤ X ├┤ I ├─░───┼────■──────┤ X ├──░────────┼────■───╫─»
                       └───┘└───┘ ░ ┌─┴─┐┌─┴─┐┌─┐ └─┬─┘  ░      ┌─┴─┐┌─┴─┐ ║ »
    link_qubit_0: ────────────────░─┤ X ├┤ X ├┤M├───┼────░──|0>─┤ X ├┤ X ├─╫─»
                                  ░ └───┘└───┘└╥┘   │    ░      └───┘└───┘ ║ »
      code_bit: 2/═════════════════════════════╬════╪══════════════════════╩═»
                                               ║ ┌──┴──┐                   0 »
round_link_bit: 2/═════════════════════════════╩═╡ = 1 ╞═════════════════════»
                                               0 └─────┘                     »
«                        
«    code_qubit_0: ──────
«                     ┌─┐
«    code_qubit_1: ───┤M├
«                  ┌─┐└╥┘
«    link_qubit_0: ┤M├─╫─
«                  └╥┘ ║ 
«      code_bit: 2/═╬══╩═
«                   ║  1 
«round_link_bit: 2/═╩════
«                   1    
{'01 00': 490, '01 11': 534}

			========= circuit[ 1 ][ 1 ]=========

                  ┌───┐     ┌───┐ ░                      ░                ┌─┐»
    code_qubit_0: ┤ H ├──■──┤ X ├─░───■──────────────────░────────■───────┤M├»
                  └───┘┌─┴─┐├───┤ ░   │           ┌───┐  ░        │       └╥┘»
    code_qubit_1: ─────┤ X ├┤ X ├─░───┼────■──────┤ X ├──░────────┼────■───╫─»
                       └───┘└───┘ ░ ┌─┴─┐┌─┴─┐┌─┐ └─┬─┘  ░      ┌─┴─┐┌─┴─┐ ║ »
    link_qubit_0: ────────────────░─┤ X ├┤ X ├┤M├───┼────░──|0>─┤ X ├┤ X ├─╫─»
                                  ░ └───┘└───┘└╥┘   │    ░      └───┘└───┘ ║ »
      code_bit: 2/═════════════════════════════╬════╪══════════════════════╩═»
                                               ║ ┌──┴──┐                   0 »
round_link_bit: 2/═════════════════════════════╩═╡ = 1 ╞═════════════════════»
                                               0 └─────┘                     »
«                        
«    code_qubit_0: ──────
«                     ┌─┐
«    code_qubit_1: ───┤M├
«                  ┌─┐└╥┘
«    link_qubit_0: ┤M├─╫─
«                  └╥┘ ║ 
«      code_bit: 2/═╬══╩═
«                   ║  1 
«round_link_bit: 2/═╩════
«                   1    
{'00 00': 497, '00 11': 527}

			========= circuit[ 1 ][ 2 ]=========

                  ┌───┐     ┌───┐ ░                      ░                ┌─┐»
    code_qubit_0: ┤ H ├──■──┤ X ├─░───■──────────────────░────────■───────┤M├»
                  └───┘┌─┴─┐├───┤ ░   │           ┌───┐  ░        │       └╥┘»
    code_qubit_1: ─────┤ X ├┤ Z ├─░───┼────■──────┤ X ├──░────────┼────■───╫─»
                       └───┘└───┘ ░ ┌─┴─┐┌─┴─┐┌─┐ └─┬─┘  ░      ┌─┴─┐┌─┴─┐ ║ »
    link_qubit_0: ────────────────░─┤ X ├┤ X ├┤M├───┼────░──|0>─┤ X ├┤ X ├─╫─»
                                  ░ └───┘└───┘└╥┘   │    ░      └───┘└───┘ ║ »
      code_bit: 2/═════════════════════════════╬════╪══════════════════════╩═»
                                               ║ ┌──┴──┐                   0 »
round_link_bit: 2/═════════════════════════════╩═╡ = 1 ╞═════════════════════»
                                               0 └─────┘                     »
«                        
«    code_qubit_0: ──────
«                     ┌─┐
«    code_qubit_1: ───┤M├
«                  ┌─┐└╥┘
«    link_qubit_0: ┤M├─╫─
«                  └╥┘ ║ 
«      code_bit: 2/═╬══╩═
«                   ║  1 
«round_link_bit: 2/═╩════
«                   1    
{'01 00': 512, '01 11': 512}

			========= circuit[ 2 ][ 0 ]=========

                  ┌───┐     ┌───┐ ░                      ░                ┌─┐»
    code_qubit_0: ┤ H ├──■──┤ Z ├─░───■──────────────────░────────■───────┤M├»
                  └───┘┌─┴─┐├───┤ ░   │           ┌───┐  ░        │       └╥┘»
    code_qubit_1: ─────┤ X ├┤ I ├─░───┼────■──────┤ X ├──░────────┼────■───╫─»
                       └───┘└───┘ ░ ┌─┴─┐┌─┴─┐┌─┐ └─┬─┘  ░      ┌─┴─┐┌─┴─┐ ║ »
    link_qubit_0: ────────────────░─┤ X ├┤ X ├┤M├───┼────░──|0>─┤ X ├┤ X ├─╫─»
                                  ░ └───┘└───┘└╥┘   │    ░      └───┘└───┘ ║ »
      code_bit: 2/═════════════════════════════╬════╪══════════════════════╩═»
                                               ║ ┌──┴──┐                   0 »
round_link_bit: 2/═════════════════════════════╩═╡ = 1 ╞═════════════════════»
                                               0 └─────┘                     »
«                        
«    code_qubit_0: ──────
«                     ┌─┐
«    code_qubit_1: ───┤M├
«                  ┌─┐└╥┘
«    link_qubit_0: ┤M├─╫─
«                  └╥┘ ║ 
«      code_bit: 2/═╬══╩═
«                   ║  1 
«round_link_bit: 2/═╩════
«                   1    
{'00 00': 496, '00 11': 528}

			========= circuit[ 2 ][ 1 ]=========

                  ┌───┐     ┌───┐ ░                      ░                ┌─┐»
    code_qubit_0: ┤ H ├──■──┤ Z ├─░───■──────────────────░────────■───────┤M├»
                  └───┘┌─┴─┐├───┤ ░   │           ┌───┐  ░        │       └╥┘»
    code_qubit_1: ─────┤ X ├┤ X ├─░───┼────■──────┤ X ├──░────────┼────■───╫─»
                       └───┘└───┘ ░ ┌─┴─┐┌─┴─┐┌─┐ └─┬─┘  ░      ┌─┴─┐┌─┴─┐ ║ »
    link_qubit_0: ────────────────░─┤ X ├┤ X ├┤M├───┼────░──|0>─┤ X ├┤ X ├─╫─»
                                  ░ └───┘└───┘└╥┘   │    ░      └───┘└───┘ ║ »
      code_bit: 2/═════════════════════════════╬════╪══════════════════════╩═»
                                               ║ ┌──┴──┐                   0 »
round_link_bit: 2/═════════════════════════════╩═╡ = 1 ╞═════════════════════»
                                               0 └─────┘                     »
«                        
«    code_qubit_0: ──────
«                     ┌─┐
«    code_qubit_1: ───┤M├
«                  ┌─┐└╥┘
«    link_qubit_0: ┤M├─╫─
«                  └╥┘ ║ 
«      code_bit: 2/═╬══╩═
«                   ║  1 
«round_link_bit: 2/═╩════
«                   1    
{'01 00': 515, '01 11': 509}

			========= circuit[ 2 ][ 2 ]=========

                  ┌───┐     ┌───┐ ░                      ░                ┌─┐»
    code_qubit_0: ┤ H ├──■──┤ Z ├─░───■──────────────────░────────■───────┤M├»
                  └───┘┌─┴─┐├───┤ ░   │           ┌───┐  ░        │       └╥┘»
    code_qubit_1: ─────┤ X ├┤ Z ├─░───┼────■──────┤ X ├──░────────┼────■───╫─»
                       └───┘└───┘ ░ ┌─┴─┐┌─┴─┐┌─┐ └─┬─┘  ░      ┌─┴─┐┌─┴─┐ ║ »
    link_qubit_0: ────────────────░─┤ X ├┤ X ├┤M├───┼────░──|0>─┤ X ├┤ X ├─╫─»
                                  ░ └───┘└───┘└╥┘   │    ░      └───┘└───┘ ║ »
      code_bit: 2/═════════════════════════════╬════╪══════════════════════╩═»
                                               ║ ┌──┴──┐                   0 »
round_link_bit: 2/═════════════════════════════╩═╡ = 1 ╞═════════════════════»
                                               0 └─────┘                     »
«                        
«    code_qubit_0: ──────
«                     ┌─┐
«    code_qubit_1: ───┤M├
«                  ┌─┐└╥┘
«    link_qubit_0: ┤M├─╫─
«                  └╥┘ ║ 
«      code_bit: 2/═╬══╩═
«                   ║  1 
«round_link_bit: 2/═╩════
«                   1    
{'00 00': 515, '00 11': 509}
"""

import qiskit
from qiskit import *

# Contruct takes the circuit as a parameter input and applies one of the gates among identiy, bit-filp and phase-flip depending on the value of 'i' to the qubit number 'qubit' in the register 'reg'
def switch(circuit, i, qubit, reg):
    if i==0:
        circuit.i(reg[qubit])
    elif i==1:
        circuit.x(reg[qubit])
    elif i==2:
        circuit.z(reg[qubit])
    
cq=QuantumRegister(2,'code_qubit') 		# This register contains the 2 code qubits, which has to be checked by the circuit
lq=QuantumRegister(1,'link_qubit') 		# This register contains the single link qubit, which acts as the ancilla bit and on which the error analysis is performed
cb=ClassicalRegister(2,'code_bit') 		# This register contains the classical bits which store the logical readouts of the code qubits
rlb=ClassicalRegister(2,'round_link_bit') 	# This register contains the classical bits, on which the syndrome measurements are performed in order to detect and correct the error introduced in the code qubits

# Loop iterates through all the possible combinations of the "error gates" and implements them into the circuit
for i in list(range(3)):
    for j in list(range(3)):
        qc=QuantumCircuit(cq,lq,cb,rlb)
        qc.h(cq[0])
        qc.cx(cq[0],cq[1])
        switch(qc, i, 0, cq)
        switch(qc, j, 1, cq)
        qc.barrier()
        qc.cx([cq[0],cq[1]],[lq[0],lq[0]])
        qc.measure(lq[0],rlb[0])
        qc.x(cq[1]).c_if(rlb,1)
        qc.barrier()
        qc.reset(lq[0])
        qc.cx([cq[0],cq[1]],[lq[0],lq[0]])
        qc.measure(lq[0],rlb[1])
        qc.measure(cq,cb)
        print('\n\t\t\t========= circuit[',i,'][',j,']=========\n')
        print(qc)
        job = execute( qc, Aer.get_backend('qasm_simulator') )
        print(job.result().get_counts(qc))
