# https://github.com/avirajs/QuantumROM/blob/a3841331b7d3047a1a4b15b2cf5c7f7249d2cb1c/angle_QROM_code/angle_QROM_code/angle_QROM_optimized.py
import numpy as np
from scipy.linalg import hadamard



def get_angle_optim_QROM(vals, hadamards = True,measurement=True):

    # create gray code permuted hadamard
    np.set_printoptions(suppress=True)

    def gray_hadamard(dim = 4 ):
        #hadamard matrix size must be power of 2, not odd
        n=np.int(np.ceil(np.log2(dim)))
        gray_ints = []
        for i in range(0, 1<<n):
            gray=i^(i>>1)
            gray_ints.append(gray)


        mat = hadamard(dim)[:,gray_ints]
        return mat

    def return_optim_control_idx(num_lines, control_idx, idxs = []):

        if(control_idx<0 or num_lines<control_idx):
            return idxs



        return_optim_control_idx(num_lines,control_idx-1, idxs)


        idxs.append(control_idx)
        # print(idxs)

        return_optim_control_idx(num_lines,control_idx-1, idxs)

        return idxs
        # return_optim_control_idx(num_lines,control_idx-1)


    curr_vector_size = len(vals)
    new_vector_size = int(2** np.ceil(np.log2(curr_vector_size)))
    #since we need to apply a hadamard which dimension size must be 2**n, where n is positive int, we need to pad our vector
    
    # print("og vector",vals)
    # print("value of pads",new_vector_size,curr_vector_size)
    vals = np.pad(vals,(0,new_vector_size-curr_vector_size))
    # print("Vector paded",vals)
    #normalization
    vals = vals*np.pi/np.max(vals)


    M = gray_hadamard(new_vector_size)

    k = int(np.log2(new_vector_size))

    theta = 2**-k *(M.T @ vals)

    vals_size = len(vals)
    control_size = int(np.ceil(np.log2(vals_size)))

    # print("theta vectors",thetas)
    
    
    
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

    addr_reg = QuantumRegister(control_size,"addr")
    data_reg = QuantumRegister(1,"data")
    ca = ClassicalRegister(control_size,"ca_m")
    cm = ClassicalRegister(1,"data_m")
    qc_min = QuantumCircuit(addr_reg,data_reg,cm,ca)

    if hadamards:
        qc_min.h(addr_reg[:])





    #the CNOTs follow a rising and falling depths of an inorder binary search tree; more easily created using recursion
    cnot_controls = np.array(return_optim_control_idx(control_size,control_size-1))
    cnot_controls = cnot_controls.astype(np.uint8)
    cnot_controls = control_size-1 - cnot_controls

    # print("Theta size",theta.size)
    # print("Controls ",cnot_controls, cnot_controls.size)
    
    
    #CZ rotations are inbetween all CNOTS
    for idx,control in enumerate(cnot_controls):
        if theta[idx] != 0:
            qc_min.rx(theta[idx],data_reg[0])
        qc_min.cz(control,data_reg[0])


    # there is aditional theta and CNOT, that starts back over
    qc_min.rx(theta[-1],data_reg[0])


    qc_min.cz(0,data_reg[0])

    if measurement:
        qc_min.measure(data_reg,cm)

        qc_min.measure(addr_reg,ca)

    return qc_min



