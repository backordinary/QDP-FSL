# https://github.com/arecibokck/Moonlight/blob/f0c4dd82002d8d79c9391376cb7c1a5be0a634a9/Qiskit/GHZ_ibmqx4.py
import qiskit as qk
import time
import Qconfig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec

try:
    # Define the Quantum and Classical Registers
    q = qk.QuantumRegister(3)
    c = qk.ClassicalRegister(3)

    # Build the circuit
    qcircuit = qk.QuantumCircuit(q, c)

    qcircuit.h(q[0])
    qcircuit.h(q[1])
    qcircuit.x(q[2])
    qcircuit.cx(q[1], q[2])
    qcircuit.cx(q[0], q[2])
    qcircuit.h(q[0])
    qcircuit.h(q[1])
    qcircuit.h(q[2])

    qcircuit.measure(q, c)

    #Default Device: IBM 5 Qubit 'Tenerife', for Public use
    least_busy_device = 'ibmqx4'
    shots_num = 1024
    credits = 3

    try:
        # select least busy available device and execute.
        least_busy_device = qk.providers.ibmq.least_busy(qk.IBMQ.backends(simulator=False))
    except:
        print("All devices are currently unavailable.")

    print("Running on device: ", least_busy_device)

    #Compile and Execute the circuit on a real device backend
    qk.IBMQ.enable_account(Qconfig.APItoken)
    ibmqxf = qk.IBMQ.get_backend(least_busy_device)
    print(ibmqxf.status())

    job_exp = qk.execute(qcircuit, backend=ibmqxf, shots = shots_num, max_credits=credits)

    lapse = 0
    interval = 5
    while lapse < 60:
        print('Status @ {} seconds'.format(interval * lapse))
        print(job_exp.status) # !? How to monitor job status !?
        time.sleep(interval)
        lapse += 1

    result = job_exp.result()
    data = result.get_counts(qcircuit)

    # Print the result
    print(data)

    #Draw and Save the circuit
    diagram = qcircuit.draw(output = "mpl")
    diagram.savefig('GHZState.png', dpi = 100)

    fig = plt.figure(figsize = (15,5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    #Show Circuit
    a = fig.add_subplot(gs[0])
    a.set_title('Quantum Circuit')
    a.set_xticks([])
    a.set_yticks([])
    img = mpimg.imread('./GHZState.png')
    imgplot = plt.imshow(img)

    #Plot Histogram
    a = fig.add_subplot(gs[1])
    a.set_title('IBM_Q_5 Tenerife Result')
    #plt.xlabel('States', fontsize=11)
    plt.ylabel('Probability', fontsize=11)
    dk = list(data.keys())
    dv = list(data.values())
    dv = [round(x / shots_num, 3) for x in dv]
    index = np.arange(len(dk))
    plt.xticks(index, dk, fontsize=11, rotation=30)
    plt.bar(index, dv)
    for i in range(len(dk)):
        plt.text(x = index[i]-0.7 , y = dv[i]+0.005, s = str(dv[i]), size = 11)
    plt.show()
    fig.savefig('Exp_Result.png')

except qk.QiskitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))

#2019-03-28 11:45 First Run Data = {'100': 34, '111': 260, '101': 70, '110': 85, '011': 44, '000': 384, '010': 105, '001': 42}
