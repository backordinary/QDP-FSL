# https://github.com/avirajs/QuantumROM/blob/a3841331b7d3047a1a4b15b2cf5c7f7249d2cb1c/angle_QROM_code/angle_QROM_code/angle_QROM_baseline.py
import numpy as np


def get_angle_QROM(vals, hadamards = True,measurement=True):

    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    vals = np.array(vals)
    vals = vals*np.pi/np.max(vals)

    #get length of value vector
    vals_size = len(vals)

    #identify how many control lines are needed
    control_size = int(np.ceil(np.log2(vals_size)))
    #create address for every location
    addr = np.arange(vals_size)
    #expand address to binary to use as controls
    addr_bits = np.unpackbits(np.array(addr,np.uint8).reshape(-1,1),axis=1)[:,-control_size:]
    #put pauli-x around the address lines with negative controls
    negative_controls = [np.where(addr==0)[0] for addr in addr_bits]

    #make the circuit
    addr_reg = QuantumRegister(control_size,"addr")
    data_reg = QuantumRegister(1,"data")
    ca = ClassicalRegister(control_size,"ca_m")
    cm = ClassicalRegister(1,"data_m")
    qc = QuantumCircuit(addr_reg,data_reg,cm,ca)

    if hadamards:
        qc.h(addr_reg[:])


    for idx,controls in enumerate(negative_controls):

        try:
            if len(controls)!=0:
                qc.x(controls.tolist())

            qc.mcrx(vals[idx],addr_reg,data_reg[0])

            if len(controls)!=0:
                qc.x(controls.tolist())
            # qc.barrier()

        except Exception as e:
            print(e)


    if measurement:
        qc.measure(data_reg,cm)

        qc.measure(addr_reg,ca)

    return qc
