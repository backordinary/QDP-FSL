# https://github.com/LivingTheCoderDream/Quantum-Computing/blob/af9542f505b64628dd9656646f540e86d992ac88/RNG2.py
from qiskit import QuantumCircuit, execute, Aer
from math import pi, sqrt
from qiskit.visualization import plot_bloch_multivector, plot_histogram

sim = Aer.get_backend('statevector_simulator')


def check_limits():
    print("Let us locate the range in which you want your random number to be...")
    while True:
        try:
            lower_limit = int(input("Enter the lower limit: "))
            upper_limit = int(input("Enter the upper limit: "))
            if upper_limit > 0 and lower_limit >= 0:
                if upper_limit > lower_limit:
                    tup = (lower_limit, upper_limit)
                    return tup
                    break
                else:
                    print("Invalid entries. Try again!")
            else:
                print("Ivalid entries. Try again!")
        except:
            print("Invalid entries. Try again!")

            
def binary_to_decimal(b):
    b_string = str(b)
    len_b = len(b_string)
    s_empty = ""
    while len_b > 0:
        b_reverse = (b_string[len_b-1])
        s_empty = s_empty + b_reverse
        len_b = len_b - 1

    tot_add = 0
    hei = len(s_empty)
    i = 0
    while i < hei:
        p = int(b_string[i])*(2**i)
        tot_add = tot_add + p
        i = i + 1
    return tot_add

def get_binary_from_qubits(qc, sim_result):
    counts = sim_result.get_counts(qc)
    return list(counts.keys())[0].split(' ')[0]


def min_qubits_needed(ul, ll):
    difference = ul-ll
    i = 0
    while True:
        if ((2**i)-1) >= difference:
            tup = (i, difference)
            return tup
            break
        i += 1

def run (l, u, qc, Aer, sim, difference):
    backend_sim = Aer.get_backend('qasm_simulator')
    sim = execute(qc, backend_sim, shots=1)
    sim_result = sim.result()

    random_number = binary_to_decimal(get_binary_from_qubits(qc, sim_result))
    
    check_output_range(random_number, l, u, qc, Aer, sim, difference)



def check_output_range(number, ll, ul, qc, Aer, sim, difference):
    
    if difference >= number and number>= 0:
        print(f"Your PURE QUANTUM RANDOM NUMBER is : {number + (ll)}")
    else:
        run(ll, ul, qc, Aer, sim, difference)
        
        
        

tup = check_limits()
u = tup[1]
l = tup[0]


tup2 = min_qubits_needed(u, l)
num_of_qubits = tup2[0]
difference = tup2[1]

qc = QuantumCircuit(num_of_qubits,num_of_qubits)

qc.h(range(num_of_qubits))

qc.measure_all()
qc.draw('mpl')


run(l, u, qc, Aer, sim, difference)

