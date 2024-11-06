# https://github.com/ZZSquare/InteractionRepresentation/blob/17eb05fba3efe0197da33f06cf419689db8a66ea/main.py
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qiskit.providers.aer import noise
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit.transpiler import Layout
from qiskit import Aer, QuantumRegister, QuantumCircuit, execute, IBMQ, ClassicalRegister
from qfunctions import adiabaticramp, theta, twospin_instruction, exactm, interactionm


def twospin_df(J, Bx, dB, Bz_max, dt_steps, dt_steps_bool, gamma, skip, provider, backend, pq1, pq2):
    count_list = []
    calibrated_count_list = []
    tarray, Bzarray, dt_or_steps, _ = adiabaticramp(J, Bx, dB, Bz_max, dt_steps, dt_steps_bool, gamma)

    if dt_steps_bool == 'dt':
        dt = dt_steps
    else:
        dt = dt_or_steps

    thetaarray, _ = theta(Bzarray, dt)

    # Calibration Matrix
    meas_calibs, state_labels = complete_meas_cal(qubit_list=[0, 1], qr=QuantumRegister(2), circlabel='2spin')
    cal_results = execute(meas_calibs, backend=provider.get_backend(backend), shots=1000).result()
    meas_calibs, state_labels = complete_meas_cal(qubit_list=[0, 1], qr=QuantumRegister(2), circlabel='2spin')
    meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='2spin')
    meas_fitter.plot_calibration()

    qmap = Layout()

    i = 0
    while (i < len(thetaarray)):
        print('Bz = %f' % (Bzarray[i]))
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circ = QuantumCircuit(qr, cr)
        qmap.from_dict(input_dict={qr[0]: pq1, qr[1]: pq2})
        circ.initialize([0, 0, 0, 1], [0, 1])

        twospin_instruction(circ, J, Bx, thetaarray[:i + 1], dt)
        circ.measure([0, 1], [0, 1])

        result = execute(circ, provider.get_backend(backend), shots=5000).result()
        counts = result.get_counts()
        counts['Time'] = tarray[i]
        counts['Bz'] = Bzarray[i]
        count_list.append(counts)
        with open("count_list.txt", "w") as output:
            output.write(str(count_list))

        mitigated_counts = meas_fitter.filter.apply(result).get_counts()
        mitigated_counts['Time'] = tarray[i]
        mitigated_counts['Bz'] = Bzarray[i]
        calibrated_count_list.append(mitigated_counts)
        with open("calibrated_count_list.txt", "w") as output:
            output.write(str(calibrated_count_list))

        i = i + 1 + skip

    # Creating dataframe
    df = pd.DataFrame(count_list)
    time_col = df.pop('Time')
    df.insert(0, 'Time', time_col)
    df['Exact'] = exactm(J, Bx, Bzarray, dt)
    df['Interaction'] = interactionm(J, Bx, thetaarray, dt)

    calibrated_df = pd.DataFrame(calibrated_count_list)
    time_col = df.pop('Time')
    df.insert(0, 'Time', time_col)

    if '00' not in df:
        df['00'] = 0
    if '01' not in df:
        df['01'] = 0
    if '10' not in df:
        df['10'] = 0
    if '11' not in df:
        df['11'] = 0
    df = df.fillna(0)

    if '00' not in calibrated_df:
        calibrated_df['00'] = 0
    if '01' not in calibrated_df:
        calibrated_df['01'] = 0
    if '10' not in calibrated_df:
        calibrated_df['10'] = 0
    if '11' not in calibrated_df:
        calibrated_df['11'] = 0
    calibrated_df = calibrated_df.fillna(0)

    # Calculating mz
    total = df['00'] + df['01'] + df['10'] + df['11']
    df['mz'] = -(df['00'] / total - df['11'] / total)
    calibrated_df['mz'] = -(df['00'] / total - df['11'] / total)

    # Creating Files
    if dt_steps_bool == 'dt':
        df.to_csv('J={:.3f} Bx={:.3f} dB={:.3f} BzMax={:.3f} dt={:.3f} gamma={:.3f}.csv'.format(J, Bx, dB, Bz_max,
                                                                                                      dt_steps, gamma))
        calibrated_df.to_csv(
            'Calibrated J={:.3f} Bx={:.3f} dB={:.3f} BzMax={:.3f} dt={:.3f} gamma={:.3f}.csv'.format(J, Bx, dB,
                                                                                                           Bz_max,
                                                                                                           dt_steps,
                                                                                                           gamma))
    else:
        df.to_csv(
            'J={:.3f} Bx={:.3f} dB={:.3f} BzMax={:.3f} BzSteps={:.3f} gamma={:.3f}.csv'.format(J, Bx, dB, Bz_max,
                                                                                                     dt_steps, gamma))
        calibrated_df.to_csv(
            'Calibrated J={:.3f} Bx={:.3f} dB={:.3f} BzMax={:.3f} BzSteps={:.3f} gamma={:.3f}.csv'.format(J, Bx,
                                                                                                                dB,
                                                                                                                Bz_max,
                                                                                                                dt_steps,
                                                                                                                gamma))

    return df, dt_or_steps, thetaarray


def main():

    # Set account credentials here
    #IBMQ_KEY =
    #IBMQ.save_account(IBMQ_KEY)
    provider = IBMQ.load_account()

    # Set name of backend
    # eg. 'ibmq_santiago'
    backend ='ibmq_santiago'

    # Map virtual qubits to physical qubits
    physicalq1 =0
    physicalq2 =7

    J = 1
    Bx = 0.03
    dB = 0.01
    Bz_max = 2
    dt_steps = 2
    dt = 'dt'
    gamma = 3

    twospin_df(J, Bx, dB, Bz_max, dt_steps, dt, gamma, 0, provider, backend, physicalq1, physicalq2)

if __name__ == "__main__":
    main()