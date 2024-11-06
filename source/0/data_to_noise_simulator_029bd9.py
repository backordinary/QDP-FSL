# https://github.com/mliu2023/quantum-Proof-of-Work/blob/407ace59c4869146d84eb5c852169fdc6c77b1a9/qpow_demos/data_to_noise_simulator.py
def data_to_noise_simulator(bit_flip = False, thermal_relaxation = False):
    """
    data_to_noise_simulator builds a noise model based on the data provided by IBM
    
    :param bit_flip: whether or not to include bit flip error
    :param thermal_relaxation: whether or not to include thermal relaxation error
    :return: the noise model for noise simulation
    """    

    import numpy as np
    import pandas as pd
    from qiskit.providers.aer.noise import NoiseModel, ReadoutError, depolarizing_error
    import qiskit.providers.aer.noise as noise

    # select_backend = input('Select a backend: ')
    select_backend = 'ibm_peekskill'
    df = pd.read_csv(str(select_backend)+'_calibrations.csv')
    qubit_arr = [1, 4, 7, 10, 12, 15, 18, 21, 23, 24, 25, 22, 19, 16, 14, 11, 8, 5]

    noise_model = NoiseModel()
    coupling_map = []

    #for i in range(len(df)):
    for i, qubit in enumerate(qubit_arr):

        if(bit_flip):
            # single qubit gate errors
            id_error = df.loc[qubit, 'ID error ']
            # assuming that the errors of all single qubit gates are the same
            single_qubit_gate_error = depolarizing_error(id_error, 1)
            noise_model.add_quantum_error(single_qubit_gate_error, ['id', 'sx', 'x', 'ry'], qubits = [i])

            # readout error
            p0given1 = df.loc[qubit, 'Prob meas0 prep1 ']
            p1given0 = df.loc[qubit, 'Prob meas1 prep0 ']
            noise_model.add_readout_error(ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]]), qubits = [i])
            
        cnot_errors = df.loc[qubit, 'CNOT error '].split('; ')
        for error in cnot_errors:
            x = error.split(':')


            first_qubit = int(x[0].split('_')[0])
            second_qubit = int(x[0].split('_')[1])
            if(second_qubit in qubit_arr):
                coupling_map.append([qubit_arr.index(first_qubit), qubit_arr.index(second_qubit)])
                if(bit_flip):
                    cnot_error = depolarizing_error(float(x[1]), 2)
                    noise_model.add_quantum_error(cnot_error, ['cx'], qubits = [qubit_arr.index(first_qubit), qubit_arr.index(second_qubit)])
    if(thermal_relaxation):
        # T1s = np.random.normal(50e3, 10e3, 4)
        # T2s = np.random.normal(70e3, 10e3, 4)
        T1s = df.loc[np.array(qubit_arr), 'T1 (us)']
        T2s = df.loc[np.array(qubit_arr), 'T2 (us)']

        # Truncate random T2s <= T1s
        T2s = [min(T2s[j], 2 * T1s[j]) for j in qubit_arr]

        # Instruction times (in nanoseconds)
        time_single_gate = 100 # (two X90 pulses)
        time_cx = 300

        # QuantumError objects
        errors_id = [noise.thermal_relaxation_error(1000*t1, 1000*t2, time_single_gate)
                        for t1, t2 in zip(T1s, T2s)]
        errors_sx = [noise.thermal_relaxation_error(1000*t1, 1000*t2, time_single_gate)
                        for t1, t2 in zip(T1s, T2s)]
        errors_x  = [noise.thermal_relaxation_error(1000*t1, 1000*t2, time_single_gate)
                    for t1, t2 in zip(T1s, T2s)]
        errors_rz  = [noise.thermal_relaxation_error(1000*t1, 1000*t2, time_single_gate)
                    for t1, t2 in zip(T1s, T2s)]
        errors_cx = [[noise.thermal_relaxation_error(1000*t1a, 1000*t2a, time_cx).expand(
                    noise.thermal_relaxation_error(1000*t1b, 1000*t2b, time_cx))
                    for t1a, t2a in zip(T1s, T2s)]
                    for t1b, t2b in zip(T1s, T2s)]

        # Add errors to noise model
        for j in range(len(qubit_arr)):
            noise_model.add_quantum_error(errors_id[j], "id", [j])
            noise_model.add_quantum_error(errors_sx[j], "sx", [j])
            noise_model.add_quantum_error(errors_x[j], "x", [j])
            noise_model.add_quantum_error(errors_rz[j], "rz", [j])
            noise_model.add_quantum_error(errors_rz[j], "ry", [j])
            for k in range(len(qubit_arr)):
                noise_model.add_quantum_error(errors_cx[j][k], "cx", [j, k])    

    return noise_model, coupling_map
