# https://github.com/cagalvisf/Qubit_state_estimation/blob/928fd1519ec109e0f252915eae32f4625046f4de/Plus_QSE_test.py
## Graphical comparison between estimation in HELMI and IBMQ
## Imports
import sys
import argparse
from argparse import RawTextHelpFormatter

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer, BasicAer, execute
from qiskit.compiler import transpile
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit.visualization import plot_bloch_vector
from qiskit.quantum_info import state_fidelity
from qiskit.transpiler.passes import RemoveBarriers
from qiskit import IBMQ

import matplotlib.pyplot as plt
import numpy as np
pi = np.pi
j = complex(0,1)

def get_args():

    args_parser = argparse.ArgumentParser(
        description="""Qubit state estimation""",
        formatter_class=RawTextHelpFormatter,
        epilog="""Example usage:
        python Plus_estim.py --backend simulator
        """,
    )

    # Parse Arguments

    args_parser.add_argument(
        "--backend",
        help="""
        Define the backend for running the program.
        'aer'/'simulator' runs on Qiskit's aer simulator,
        'helmi' runs on VTT Helmi Quantum Computer,
        'fake_helmi' runs in the fake helmi simulator
        """,
        required=False,
        type=str,
        default=None,
        choices=["helmi", "simulator", "IBMQ", "fake_helmi"],
    )


    return args_parser.parse_args()

def Initialization(angles = [0,0,0]):

    qr = QuantumRegister(3)
    qc = QuantumCircuit(qr)

    #Initialice qubits
    qc.h(qr[0])                                    #A meter
    qc.h(qr[2])                                    #B meter
    qc.u(angles[0],angles[1],angles[2],[qr[1]])    #System

    Init_gate = qc.to_gate()
    Init_gate.name= "Initialization"

    return Init_gate

def U_operator(A=[0,0,0,0,0,0],B=[0,0,0,0,0,0]):

    ## A y B are vectors with the rotation angles of the meters A y B
    qr = QuantumRegister(3)
    qc = QuantumCircuit(qr)

    #####==================================#####
    #####======= Evolution operator =======#####
    #####==================================#####



    ## We apply rotations around a CNOT between the S and A qubits
    qc.u(A[0],A[1],A[2],qr[0])
    qc.cx(qr[1],qr[0])
    qc.u(A[3],A[4],A[5],qr[0])


    ## We change the basis of the S qubit
    qc.h(qr[1])

    ## Now, we apply rotations around a CNOT between the S and B qubits
    qc.u(B[0],B[1],B[2],qr[2])
    qc.cx(qr[1],qr[2])
    qc.u(B[3],B[4],B[5],qr[2])

    ## We revert the hadamard transformation over S
    qc.h(qr[1])

    U_gate=qc.to_gate()
    U_gate.name="U Operator"

    return U_gate

def U_matrix(angles_a,angles_b):
    ## Definimos los qubits sobre los que actúa la compuerta
    A = QuantumRegister(1,'a')
    S = QuantumRegister(1,'s')
    B = QuantumRegister(1,'b')
    cr = ClassicalRegister(3)

    ## Ejecutamos la compuerta sobre estos regitros
    qc = QuantumCircuit(A,S,B, cr)
    qc.append(U_operator(angles_a,angles_b),[A,S,B])

    ## Usamos el simulador de IBM para obtener la matriz asociada al operador
    backend = BasicAer.get_backend('unitary_simulator')
    job = execute(qc, backend)
    U = job.result().get_unitary(qc)
    return U

def T_matrix(U):

    ## we define the U_{ij} operators
    U_00 = np.matrix(np.array([[U[0,0],U[0,2]],[U[2,0],U[2,2]]]) + np.array([[U[0,1],U[0,3]],[U[2,1],U[2,3]]]) + np.array([[U[0,4],U[0,6]],[U[2,4],U[2,6]]]) + np.array([[U[0,5],U[0,7]],[U[2,5],U[2,7]]]))
    U_01 = np.matrix(np.array([[U[1,0],U[1,2]],[U[3,0],U[3,2]]]) + np.array([[U[1,1],U[1,3]],[U[3,1],U[3,3]]]) + np.array([[U[1,4],U[1,6]],[U[3,4],U[3,6]]]) + np.array([[U[1,5],U[1,7]],[U[3,5],U[3,7]]]))
    U_10 = np.matrix(np.array([[U[4,0],U[4,2]],[U[6,0],U[6,2]]]) + np.array([[U[4,1],U[4,3]],[U[6,1],U[6,3]]]) + np.array([[U[4,4],U[4,6]],[U[6,4],U[6,6]]]) + np.array([[U[4,5],U[4,7]],[U[6,5],U[6,7]]]))
    U_11 = np.matrix(np.array([[U[5,0],U[5,2]],[U[7,0],U[7,2]]]) + np.array([[U[5,1],U[5,3]],[U[7,1],U[7,3]]]) + np.array([[U[5,4],U[5,6]],[U[7,4],U[7,6]]]) + np.array([[U[5,5],U[5,7]],[U[7,5],U[7,7]]]))

    ## and theyr adjoints
    U_00d = U_00.getH()
    U_01d = U_01.getH()
    U_10d = U_10.getH()
    U_11d = U_11.getH()

    ## The auxiliary operators A and B
    A = 1/16*(np.matmul(U_00d,U_10) + np.matmul(U_01d,U_11) + np.matmul(U_10d,U_00) + np.matmul(U_11d,U_01))
    B = 1/16*(np.matmul(U_00d,U_01) + np.matmul(U_01d,U_00) + np.matmul(U_10d,U_11) + np.matmul(U_11d,U_10))
    C = 1/16*(np.matmul(U_00d,U_11) + np.matmul(U_01d,U_10) + np.matmul(U_10d,U_01) + np.matmul(U_11d,U_00))

    ## We define the Pauli matrices and the identiry
    X = np.matrix([[0,1],[1,0]])
    Y = np.matrix([[0,-j],[j,0]])
    Z = np.matrix([[1,0],[0,-1]])
    I = np.matrix([[1,0],[0,1]])

    ## We evaluate the components of the a_\mu, b_\mu and c_\mu vectors
    a0 = 0.5*np.trace(np.matmul(A,I))
    a1 = 0.5*np.trace(np.matmul(A,X))
    a2 = 0.5*np.trace(np.matmul(A,Y))
    a3 = 0.5*np.trace(np.matmul(A,Z))

    b0 = 0.5*np.trace(np.matmul(B,I))
    b1 = 0.5*np.trace(np.matmul(B,X))
    b2 = 0.5*np.trace(np.matmul(B,Y))
    b3 = 0.5*np.trace(np.matmul(B,Z))

    c0 = 0.5*np.trace(np.matmul(C,I))
    c1 = 0.5*np.trace(np.matmul(C,X))
    c2 = 0.5*np.trace(np.matmul(C,Y))
    c3 = 0.5*np.trace(np.matmul(C,Z))

    T = np.matrix([[0.25+a0+b0+c0, a1+b1+c1, a2+b2+c2, a3+b3+c3],[0.25+a0-b0-c0, a1-b1-c1, a2-b2-c2, a3-b3-c3],[0.25-a0+b0-c0, -a1+b1-c1, -a2+b2-c2, -a3+b3-c3],[0.25-a0-b0+c0, -a1-b1+c1, -a2-b2+c2, -a3-b3+c3]])
    return T

def bloch_vector_to_state(s):
    ## We transform the bloch vector into a qubit state
    s1 = s[0]
    s2 = s[1]
    s3 = s[2]

    if (s3>0):
        th=np.arctan(np.sqrt(s1*s1 + s2*s2)/s3)
    elif (s3 == 0):
        th = pi/2
    elif (s3 < 0):
        th=np.arctan(np.sqrt(s1*s1 + s2*s2)/s3) + pi

    if (s1>0 and s2>0):
        phi = np.arctan(s2/s1)
    elif (s1>0 and s2<0):
        phi = np.arctan(s2/s1) + 2*pi
    elif (s1 == 0):
        phi = pi/2*np.sign(s2)
    elif (s1<0):
        phi = np.arctan(s2/s1) + pi

    ## We write the alpha and beta amplitudes
    c_0 = np.cos(th/2)
    c_1 = (np.cos(phi) + complex(0,1)*np.sin(phi))*np.sin(th/2)

    state = np.array([c_0,c_1])
    return state

def init_ang_to_bloch_vector(angles):

    th, phi, lam = angles

    s = np.array([np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi), np.cos(th)])

    return s


def disc_ML_est(mm,freq):
    ## Function for maximun-likeihood estimation
    ## mm = measurement matrix
    ## freq = probabilities from the experiment
    mm = np.array(mm)
    se = np.array([1.0,0.0,0.0,0.0])
    nint = 10000
    for k in range(1,nint):
        pe = np.dot(mm,se)

        re = np.dot(np.transpose(mm),(freq/pe))

        ge = re[1]**2 + re[2]**2 + re[3]**2 - re[0]**2
        se[1] = (2*re[1]-se[1]*ge)/(2*re[0]+ge)
        se[2] = (2*re[2]-se[2]*ge)/(2*re[0]+ge)
        se[3] = (2*re[3]-se[3]*ge)/(2*re[0]+ge)
    return se

def main():

    args = get_args()
    shots = 1024
    th_a1,phi_a1,lam_a1,th_b1,phi_b1,lam_b1 = 0.5871626,  1.57737493, 2.52063619, 0.70004151, 4.30553732, 3.45993977
    th_a2,phi_a2,lam_a2,th_b2,phi_b2,lam_b2 = 2.55283129, 1.93819982, 0.30976956, 0.67288561, 6.47455126, 4.4695403

    angles_a=[th_a1,phi_a1,lam_a1,th_a2,phi_a2,lam_a2]
    angles_b=[th_b1,phi_b1,lam_b1,th_b2,phi_b2,lam_b2]



    U = U_matrix(angles_a,angles_b)
    T = T_matrix(U)

    ## Initial angles of the rotation that defines the initial state of the qubit S
    angle_i=[pi/2,0,0]

    ## Quantum registers of the A, B and S qubits
    A = QuantumRegister(1,'a')
    S = QuantumRegister(1,'s')
    B = QuantumRegister(1,'b')
    cr = ClassicalRegister(3)

    ## Circuit initialization
    tomography_circuit = QuantumCircuit(A,S,B, cr)

    ## We initialize our qubits
    tomography_circuit.append(Initialization(angle_i),[A,S,B])
    tomography_circuit.barrier()

    ## Apply the evolution operator
    tomography_circuit.append(U_operator(angles_a,angles_b),[A,S,B])
    tomography_circuit.barrier()

    ## Change the measurement basis
    tomography_circuit.h(A)                     #A
    tomography_circuit.h(B)                     #B
    tomography_circuit.barrier()

    ## Measurement of the A and B meters
    tomography_circuit.measure(A,cr[0])         #A
    tomography_circuit.measure(B,cr[2])         #B

    print("Running on backend = ", args.backend)

    if args.backend == "simulator" or args.backend == "aer":

        backend = Aer.get_backend('qasm_simulator')
        basis_gates = backend.configuration().basis_gates
        transpiled_tomography_circuit = transpile(tomography_circuit, basis_gates=basis_gates)

        transpiled_tomography_circuit = RemoveBarriers()(transpiled_tomography_circuit)
        Sim_result_tomography_circuit = backend.run(transpiled_tomography_circuit, shots=1024).result()
    
    elif args.backend == "IBMQ":
        IBMQ.load_account()

        provider=IBMQ.get_provider('ibm-q')
        backend = provider.get_backend('ibmq_lima')

        basis_gates = backend.configuration().basis_gates
        transpiled_tomography_circuit = transpile(tomography_circuit, basis_gates=basis_gates)
        transpiled_tomography_circuit = RemoveBarriers()(transpiled_tomography_circuit)

        print("Running on ", backend.configuration().backend_name)

        Sim_result_tomography_circuit = backend.run(transpiled_tomography_circuit, shots=1024).result()

    elif args.backend == "helmi":
        from csc_qu_tools.qiskit import Helmi as helmi
        provider = helmi()
        backend = provider.set_backend()
        basis_gates = provider.basis_gates

        transpiled_tomography_circuit = transpile(tomography_circuit, basis_gates=basis_gates)
        transpiled_tomography_circuit = RemoveBarriers()(transpiled_tomography_circuit)

        virtual_qubits = transpiled_tomography_circuit.qubits
        qubit_mapping = {
                virtual_qubits[0]: "QB1",
                virtual_qubits[1]: "QB3",
                virtual_qubits[2]: "QB2",
            }
        Sim_result_tomography_circuit = backend.run(transpiled_tomography_circuit, shots=1024, qubit_mapping=qubit_mapping).result()

    elif args.backend == "fake_helmi":
        from csc_qu_tools.qiskit.mock import FakeHelmi

        print(
                "Inducing artificial noise into Simulator with FakeHelmi Noise Model"
            )
        basis_gates = ["r", "cz"]
        backend = FakeHelmi()

        transpiled_tomography_circuit = transpile(tomography_circuit, basis_gates=basis_gates)
        transpiled_tomography_circuit = RemoveBarriers()(transpiled_tomography_circuit)

        virtual_qubits = transpiled_tomography_circuit.qubits
        qubit_mapping = {
                virtual_qubits[0]: "QB1",
                virtual_qubits[1]: "QB3",
                virtual_qubits[2]: "QB2",
            }
        Sim_result_tomography_circuit = backend.run(transpiled_tomography_circuit, shots=1024, qubit_mapping=qubit_mapping).result()


    else:
        sys.exit("Backend option not recognized")

    Sim_result_counts_tomography_circuit = Sim_result_tomography_circuit.get_counts()

    ## We build the vector P
    if '000' in Sim_result_counts_tomography_circuit:
        p00 = Sim_result_counts_tomography_circuit['000']/shots
    else:
        p00 = 0

    if '001' in Sim_result_counts_tomography_circuit:
        p01 = Sim_result_counts_tomography_circuit['001']/shots
    else:
        p01 = 0
    if '100' in Sim_result_counts_tomography_circuit:
        p10 = Sim_result_counts_tomography_circuit['100']/shots
    else:
        p10 = 0
    if '101' in Sim_result_counts_tomography_circuit:
        p11 = Sim_result_counts_tomography_circuit['101']/shots
    else:
        p11 = 0

    p = np.array([p00, p01, p10, p11])
    print(p*shots)
    U = U_matrix(angles_a, angles_b)
    T = T_matrix(U)
    
    s = np.array(np.matmul(np.linalg.inv(T),p), ndmin=0)
    s = np.reshape(s,4)

    s_ml = disc_ML_est(T, p)
    s_test_ml = [s_ml[1], s_ml[2], s_ml[3]]
    print(s)

    s1=s[1].real
    s2=s[2].real
    s3=s[3].real

    s_test = [s1,s2,s3]

    s_ideal = init_ang_to_bloch_vector(angle_i)
    ## Initialize a 1 qubit circuit
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)

    qc_i = QuantumCircuit(qr,cr)

    ## Use the U rotation that we used for initializing the qubit S
    qc_i.u(angle_i[0],angle_i[1],angle_i[2],qr[0])

    backend = BasicAer.get_backend('statevector_simulator')
    result = execute(qc_i, backend).result()

    ## Take the statevector
    initial_state  = result.get_statevector(qc_i)
    estim_state = bloch_vector_to_state(s_test)
    estim_state_ml = bloch_vector_to_state(s_test_ml)

    ## Calculate the fidelity
    fidelity = state_fidelity(initial_state,estim_state)
    fidelity_ml = state_fidelity(initial_state,estim_state_ml)
    print('Estimation fidelity: ',fidelity)
    print('Estimation norm: ', np.linalg.norm(s_test))

    print('Estimation fidelity: ',fidelity_ml)
    print('Estimation norm: ', np.linalg.norm(s_test_ml))


    ## Save the plots for visualization
    plot_bloch_vector(s_test, title="Estimated state").savefig("figures/estimation_test_"+args.backend+".pdf")
    plot_bloch_vector(s_test_ml, title="Estimated state").savefig("figures/estimation_test_"+args.backend+"_ml.pdf")
    plot_bloch_vector(s_ideal, title="Initial state").savefig("figures/initial_test.pdf")


if __name__ == "__main__":
    main()