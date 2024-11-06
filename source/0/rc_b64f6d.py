# https://github.com/hmy98213/Fault-Simulation/blob/e96bcde84f27f0470c94a6438761063c7e9bc1aa/Circuit_Generators/rc.py
from qiskit import quantum_info

def gen_rand_cliff(n):

    cliff=quantum_info.random_clifford(n)
    AG=quantum_info.decompose_clifford(cliff,method='AG')
    GD=quantum_info.decompose_clifford(cliff,method='greedy')
    
    return AG.qasm(),GD.qasm()

if __name__ == "__main__":
    path = "tmp_test_panda2/"
    # AG, GD = gen_rand_cliff(20)
    # print(AG)
    for n in range(55, 60, 5):
        AG, GD = gen_rand_cliff(n)
        file_name = "rand_cliff_%d_AD.qasm"%n
        with open(path+file_name, 'w') as f:
            f.write(AG)
        file_name = "rand_cliff_%d_GD.qasm"%n
        with open(path+file_name, 'w') as f:
            f.write(GD)