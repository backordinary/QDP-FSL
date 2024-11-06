# https://github.com/ace314/RB_2-qubit/blob/939d0156368b145606d93c1b34b61f911b449b0b/RB_1q/noisy%20pulse%20sorting.py
import copy as cp
from lib.oneqrb import *
from qiskit import quantum_info
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.quantum_info.operators.channel import ptm
import matplotlib.pyplot as plt
import scipy.stats as stats
'''
1. Sort the Hamiltonian noise into left unitary operation.
2. Extract the noise for each Clifford as unitary matrices. (numpy array)
'''

class Pulse:
    def __init__(self, pulse_type, axis, sign, angle):
        if not (pulse_type == 'noise' or pulse_type == 'pulse'):
            raise Exception('"pulse_type" could only has value "noise" or "pulse".')
        self.pulse_type = pulse_type

        if pulse_type == 'noise':
            if not (axis == 'X' or axis == 'Y' or axis == 'Z'):
                raise Exception('"axis" attribute for "pulse_type" = "noise" '
                                'could only has value \'X\', \'Y\' or \'Z\'.')
        elif pulse_type == 'pulse':
            if not (axis == 'X' or axis == 'Z'):
                raise Exception('"axis" attribute for "pulse_type" = "pulse" could only has value \'X\' or \'Z\'.')
        self.axis = axis

        if not (sign == 1 or sign == -1):
            raise Exception('"sign" could only has value 1 or -1.')
        self.sign = sign

        if pulse_type == 'noise':
            if not (angle == 'epsilon' or angle == 'gamma'):
                raise Exception('"angle" attribute for "pulse_type" = "noise" '
                                'could only has value "epsilon" or "gamma" .')
        elif pulse_type == 'pulse':
            if not (angle == 'pi/2'):
                raise Exception('"angle" attribute for "pulse_type" = "pulse" could only has value "pi/2".')
        self.angle = angle


def eijk(p1, p2, p3):   # Levi-Civita symbol
    a = (p1, p2, p3)
    if a == ('X', 'Y', 'Z') or a == ('Y', 'Z', 'X') or a == ('Z', 'X', 'Y'):
        return 1
    elif a == ('Z', 'Y', 'X') or a == ('Y', 'X', 'Z') or a == ('X', 'Z', 'Y'):
        return -1
    else:
        return None

def third_pauli(a1, a2):
    return chr(267 - ord(a1) - ord(a2))

def commute_transform(p1, p2):
    if not (isinstance(p1, Pulse) and isinstance(p2, Pulse)):
        raise TypeError('The two inputs should be of type "Pulse" objects.')
    if p1.pulse_type == 'noise' and p2.pulse_type == 'pulse':
        if p1.axis == p2.axis:
            temp = cp.deepcopy(p2)

            p2.pulse_type = p1.pulse_type
            p2.axis = p1.axis
            p2.sign = p1.sign
            p2.angle = p1.angle

            p1.pulse_type = temp.pulse_type
            p1.axis = temp.axis
            p1.sign = temp.sign
            p1.angle = temp.angle
        else:
            ax = third_pauli(p1.axis, p2.axis)
            sgn = p1.sign * p2.sign * eijk(p1.axis, p2.axis, ax)
            temp = cp.deepcopy(p1)

            p1.pulse_type = p2.pulse_type       # 'pulse'
            p1.axis = p2.axis
            p1.sign = p2.sign
            p1.angle = p2.angle

            p2.pulse_type = temp.pulse_type     # 'noise'
            p2.axis = ax
            p2.sign = sgn
            p2.angle = temp.angle
        return True
    else:
        return False

def print_pulse(p):
    if not (isinstance(p, Pulse)):
        raise TypeError('The input should be of type "Pulse" object.')
    print("{" + p.pulse_type + " ; " + p.axis + " ; " + str(p.sign) + " * " + p.angle + "}")

def pulse_noisy_x(sign):
    if not (sign == 1 or sign == -1):
        raise Exception('"sign" of noisy_X pulse could only has value 1 or -1.')
    else:
        a1 = Pulse('noise', 'Y', -1 * sign, 'gamma')
        a2 = Pulse('pulse', 'X', sign, 'pi/2')
        a3 = Pulse('noise', 'X', sign, 'epsilon')
        a4 = Pulse('noise', 'Y', sign, 'gamma')
        return [a1, a2, a3, a4]

def pulse_z(sign):
    if not (sign == 1 or sign == -1):
        raise Exception('"sign" of noisy_X pulse could only has value 1 or -1.')
    a = Pulse('pulse', 'Z', sign, 'pi/2')
    return [a]

def pulse_x(sign):
    if not (sign == 1 or sign == -1):
        raise Exception('"sign" of noisy_X pulse could only has value 1 or -1.')
    a = Pulse('pulse', 'X', sign, 'pi/2')
    return [a]

def prim_key_to_pulse(key):
    if not (0 <= key <= 3):
        raise Exception('Primitive key is out of range (should be 0~3).')
    if key == 0:
        return pulse_noisy_x(1)
    elif key == 1:
        return pulse_noisy_x(-1)
    elif key == 2:
        return pulse_z(1)
    else:
        return pulse_z(-1)

def pulse_noisy_clifford(idx):
    if not (0 <= idx <= 23):
        raise Exception('Clifford index is out of range (should be 0~23).')
    key_list = Cliff_decompose_1q[idx]
    pulse_clifford = []
    for j in range(len(key_list)):
        pulse_clifford.append(prim_key_to_pulse(key_list[j]))
    return list_flatten(pulse_clifford)

def list_flatten(t):
    return [item for sublist in t for item in sublist]

def noise_transform_to_right_channel(pulse_list):
    for j in range(len(pulse_list)):
        if not (isinstance(pulse_list[j], Pulse)):
            raise TypeError('The input should be list of type "Pulse" objects.')

    # noise simplify
    for j in range(len(pulse_list)-1):
        if j >= len(pulse_list)-1:
            break
        elif pulse_list[j].pulse_type == pulse_list[j+1].pulse_type\
                and pulse_list[j].axis == pulse_list[j+1].axis\
                and pulse_list[j].angle == pulse_list[j+1].angle:
            if pulse_list[j].sign != pulse_list[j+1].sign:
                del pulse_list[j:j+2]

    # transform noises to LHS
    unsorted = True
    while unsorted:
        boo = False
        for j in range(len(pulse_list)-1):
            boo = boo or commute_transform(pulse_list[j], pulse_list[j+1])
        unsorted = boo


pauli_dict_1q = {'I': I_1q, 'X': X_1q, 'Y': Y_1q, 'Z': Z_1q}

def pulse_to_unitary(p, delta):
    if not (isinstance(p, Pulse)):
        raise TypeError('The first input should be of type "Pulse" object.')

    sigma = pauli_dict_1q[p.axis]           # rotation pauli direction
    angle = 0
    if p.pulse_type == 'pulse':
        angle = np.pi / 2
    else:                                   # p.pulse_type == 'noise'
        if p.angle == 'epsilon':
            angle = 1 / 2 * np.sqrt(np.pi**2 + 4 * delta**2) - np.pi / 2
            # angle = 1 / 2 * delta**2      # 1st order approximation
        elif p.angle == 'gamma':
            angle = np.arcsin(delta / np.sqrt(np.pi**2 / 4 + delta ** 2))
    m = np.cos(angle / 2) * I_1q - p.sign * 1j * np.sin(angle / 2) * sigma
    return m

def find_theoretical_p(noisy_clifford_list):
    m = np.zeros((16, 16))
    for i in range(len(noisy_clifford_list)):
        g_ptm = ptm.PTM(quantum_info.Operator(get_perfect_cliff(i))).data
        depol_ch_ptm = ptm.PTM(depolarizing_error(1, 1)).data
        g_u_ptm = g_ptm - depol_ch_ptm
        g_tilde_ptm = ptm.PTM(quantum_info.Operator(noisy_clifford_list[i])).data
        m = m + np.kron(g_u_ptm, g_tilde_ptm)
    m = 1 / 24 * m
    w, v = np.linalg.eig(m)
    return w

def pdf_delta_list(mid, sigma, deltas):
    res = [0] * len(deltas)
    d = stats.norm(mid, sigma)
    for i in range(len(res)):
        res[i] = d.pdf(deltas[i])
    return res


# Single delta demonstration
'''
noise_unitary = []
delta = 0.5
noisy_clifford_unitary = []

for n in range(24):
    L = pulse_noisy_clifford(n)
    pulse_noise = []
    noise_transform_to_right_channel(L)

    # print clifford after transformation in terms of the "Pulse" objects
    print("Clifford #", n)

    for i in range(len(L)):
        print_pulse(L[i])
        if L[i].pulse_type == 'noise':
            pulse_noise.append(L[i])

    m = np.identity(2)
    for i in reversed(range(len(pulse_noise))):
        m = pulse_to_unitary(pulse_noise[i], delta/2) @ m   # delta/2: 1/2 factor here because of Clifford decomposition
    noise_unitary.append(m)
    # print("Noise channel for Clifford #", n)
    # print(m)

    print("Noise trace fidelity = ", gate_fidelity_1q(m, I_1q))
    c_thr = get_perfect_cliff(n) @ m
    noisy_clifford_unitary.append(c_thr)
    c_exp = get_cliff_1q(n, delta_t=100, noise_type=HAMILTONIAN_NOISE, noise_angle=delta)
    print("Clifford trace fidelity (thr vs. exp) = ", gate_fidelity_1q(c_exp, c_thr))
    print(" ")

d = 2
p = np.real(np.amax(find_theoretical_p(noisy_clifford_unitary)))
print((p * (d - 1) + 1) / d)

F_hamiltonian = []
d = 2
for i in range(len(noise_unitary)):
    ch = quantum_info.Operator(get_perfect_cliff(i) @ noise_unitary[i] @ get_perfect_cliff(i).conj().T)
    F_ave = quantum_info.average_gate_fidelity(ch)
    # p = (d * F_ave - 1) / (d - 1)
    # depolarizing_str.append(p)
    F_hamiltonian.append(F_ave)


dephasing_unitary = np.cos(delta) * I_1q - 1j * np.sin(delta) * Z_1q
dephasing_ch = quantum_info.Operator(dephasing_unitary)
F_dephasing = quantum_info.average_gate_fidelity(dephasing_ch)
# p_dephasing = (d * F_dephasing - 1) / (d - 1)

# print("Depolarizing strength: ", depolarizing_str)
# print("Hamiltonain noise average gate fidelity: ", (np.mean(depolarizing_str)*(d-1)+1)/2)
print("Hamiltonain noise average gate fidelity: ", np.mean(F_hamiltonian))
print("Dephasing channel noise average gate fidelity: ", F_dephasing)
'''
# Single delta demonstration end here


# Depolarizing strength comparison for different "delta"
'''
delta_list = [x * 0.01 for x in list(range(1, 51))]
d = 2

avg_depol_str = []
thr_depol_str = []
depha_str = []

for delta in delta_list:
    noise_unitary = []
    noisy_clifford_unitary = []
    for n in range(24):
        L = pulse_noisy_clifford(n)
        pulse_noise = []
        noise_transform_to_right_channel(L)

        for i in range(len(L)):
            if L[i].pulse_type == 'noise':
                pulse_noise.append(L[i])

        m = np.identity(2)
        for i in reversed(range(len(pulse_noise))):
            m = pulse_to_unitary(pulse_noise[i], delta/2) @ m  # delta/2: 1/2 here because of Clifford decomposition
        noise_unitary.append(m)
        c_thr = get_perfect_cliff(n) @ m
        noisy_clifford_unitary.append(c_thr)

    depol_str = []
    for i in range(len(noise_unitary)):
        ch = quantum_info.Operator(noise_unitary[i])
        F_ave = quantum_info.average_gate_fidelity(ch)
        p1 = (d * F_ave - 1) / (d - 1)
        depol_str.append(p1)
    avg_depol_str.append(np.mean(depol_str))

    p2 = np.real(np.amax(find_theoretical_p(noisy_clifford_unitary)))
    thr_depol_str.append(p2)

    depha_m = np.cos(delta) * I_1q - 1j * np.sin(delta) * Z_1q
    depha_ch = quantum_info.Operator(depha_m)
    F_depha = quantum_info.average_gate_fidelity(depha_ch)
    p_depha = (d * F_depha - 1) / (d - 1)
    depha_str.append(p_depha)

plot1 = plt.figure(1)
plt.plot(delta_list, depha_str, 'ro', markersize=2, label='channel noise p')
plt.plot(delta_list, avg_depol_str, 'bo', markersize=2, label='Hamiltonian noise averaged p')
plt.plot(delta_list, thr_depol_str, 'go', markersize=2, label='Hamiltonian noise RB theoretical p')

plt.title('Depolarizing strength p with constant noise')
plt.xlabel("noise strength delta (rad)")
plt.ylabel("depolarizing strength")
plt.legend()
plt.show()

fidelity_channel = [(x*(d-1)+1)/d for x in depha_str]
fidelity_thr_hamiltonian = [(x*(d-1)+1)/d for x in thr_depol_str]
fidelity_avg_hamiltonian = [(x*(d-1)+1)/d for x in avg_depol_str]

f1 = open('thr_const_delta_list_1q.pkl', 'wb')
pickle.dump(delta_list, f1)
f1.close()

f2 = open('thr_const_delta_f_channel_1q.pkl', 'wb')
pickle.dump(fidelity_channel, f2)
f2.close()

f3 = open('thr_const_delta_f_hamiltonian_avg_1q.pkl', 'wb')
pickle.dump(fidelity_avg_hamiltonian, f3)
f3.close()

f4 = open('thr_const_delta_f_hamiltonian_rb_1q.pkl', 'wb')
pickle.dump(fidelity_thr_hamiltonian, f4)
f4.close()

plot2 = plt.figure(2)
plt.plot(delta_list, fidelity_channel, 'ro', markersize=2, label='channel noise')
plt.plot(delta_list, fidelity_avg_hamiltonian, 'bo', markersize=2, label='Hamiltonian noise average')
plt.plot(delta_list, fidelity_thr_hamiltonian, 'go', markersize=2, label='Hamiltonian noise RB theoretical')

plt.title('Average gate fidelity F with constant noise')
plt.xlabel("noise strength delta (rad)")
plt.ylabel("Clifford average gate fidelity")
plt.legend()
plt.show()
'''
# Depolarizing strength comparison for different "delta" end here


# Theoretical fidelity for different sigma ensemble noise start here

sigma_list = [x * 0.01 for x in list(range(1, 51))]
sigma_max = sigma_list[-1]

n = 100
d = 2

delta_list = [-3 * sigma_max + 6 * x * sigma_max / n for x in range(n+1)]

# cliffords_ensemble_list[i][j] : i-th Clifford with noise parameter delta_list[j]
cliffords_ensemble_list = []
for i in range(24):
    cliffords_ensemble_list.append([])

# noise_ensemble_list[i][j] : right-noise gate on i-th Clifford with noise parameter delta_list[j]
noise_ensemble_list = []
for i in range(24):
    noise_ensemble_list.append([])

for i in range(24):
    L = pulse_noisy_clifford(i)
    pulse_noise = []
    noise_transform_to_right_channel(L)

    for j in range(len(L)):
        if L[j].pulse_type == 'noise':
            pulse_noise.append(L[j])

    for delta in delta_list:
        m = np.identity(2)
        for k in reversed(range(len(pulse_noise))):
            m = pulse_to_unitary(pulse_noise[k], delta / 2) @ m  # delta/2: 1/2 here because of Clifford decomposition
        noise_ensemble_list[i].append(m)
        c = get_perfect_cliff(i) @ m
        cliffords_ensemble_list[i].append(c)

f1 = open('cliffords_ensemble_noises_samples.pkl', 'wb')
pickle.dump(cliffords_ensemble_list, f1)
f1.close()

f2 = open('ensemble_noises_samples.pkl', 'wb')
pickle.dump(delta_list, f2)
f2.close()

f_delta_list = [[0] * len(delta_list)] * 24
for i in range(24):
    for j in range(len(delta_list)):
        ch = quantum_info.Operator(noise_ensemble_list[i][j])
        f_delta_list[i][j] = quantum_info.average_gate_fidelity(ch)

f3 = open('f_ave_ensemble_noises_samples.pkl', 'wb')
pickle.dump(f_delta_list, f3)
f3.close()
print(f_delta_list)

fidelity_avg_hamiltonian = []
fidelity_thr_hamiltonian = []

for sigma in sigma_list:
    # average gate fidelity
    f_clifford = [0] * 24
    for i in range(len(f_clifford)):
        f = 0
        pdf = pdf_delta_list(0, sigma, delta_list)
        for j in range(len(pdf)):
            f += pdf[j] * (6 * sigma_max / n) * f_delta_list[i][j]
        f_clifford[i] = f
    fidelity_avg_hamiltonian.append(np.mean(f_clifford))

    # Wallman fidelity
    g_tilde_ptm_list = []
    for i in range(24):
        g_tilde_ptm = np.zeros((4, 4))
        for j in range(len(delta_list)):
            g_tilde_ptm = g_tilde_ptm + pdf[j] * (6 * sigma_max / n) * ptm.PTM(quantum_info.Operator(cliffords_ensemble_list[i][j])).data
        g_tilde_ptm_list.append(g_tilde_ptm)

    m = np.zeros((16, 16))
    for i in range(24):
        g_ptm = ptm.PTM(quantum_info.Operator(get_perfect_cliff(i))).data
        depol_ch_ptm = ptm.PTM(depolarizing_error(1, 1)).data
        g_u_ptm = g_ptm - depol_ch_ptm
        g_tilde_ptm = g_tilde_ptm_list[i]
        m = m + np.kron(g_u_ptm, g_tilde_ptm)
    m = 1 / 24 * m
    w, v = np.linalg.eig(m)

    p2 = np.real(np.amax(w))
    fidelity_thr_hamiltonian.append((p2*(d-1)+1)/d)


f1 = open('thr_ensemble_sigma_list_1q.pkl', 'wb')
pickle.dump(sigma_list, f1)
f1.close()

f3 = open('thr_ensemble_sigma_f_hamiltonian_avg_1q.pkl', 'wb')
pickle.dump(fidelity_avg_hamiltonian, f3)
f3.close()

f4 = open('thr_ensemble_sigma_f_hamiltonian_rb_1q.pkl', 'wb')
pickle.dump(fidelity_thr_hamiltonian, f4)
f4.close()

plot2 = plt.figure(1)
plt.plot(sigma_list[1:], fidelity_avg_hamiltonian[1:], 'bo', markersize=2, label='Hamiltonian noise average')
plt.plot(sigma_list[1:], fidelity_thr_hamiltonian[1:], 'go', markersize=2, label='Hamiltonian noise RB theoretical')

plt.title('Average gate fidelity F with constant noise')
plt.xlabel("noise strength delta (rad)")
plt.ylabel("Clifford average gate fidelity")
plt.legend()
plt.show()


# Theoretical fidelity for different sigma ensemble noise end here

# test block start here

# delta = 0.1
# dt = 1000
#
# for n in range(24):
#     p = pulse_noisy_clifford(n)
#     noise_transform_to_left_channel(p)
#
#     for i in range(len(p)):
#         print_pulse(p[i])
#
#     m1 = np.identity(2)
#     for i in reversed(range(len(p))):
#         m1 = pulse_to_unitary(p[i], delta) @ m1
#
#     m2 = get_cliff_1q(n, delta_t=dt, noise_type=HAMILTONIAN_NOISE, noise_angle=2*delta)
#
#     print(gate_fidelity_1q(m1, m2))

# t_slice = np.linspace(0, np.pi/2, dt + 1)
# x_pi2 = I_1q
# hx = 1 / 2 * X_1q
# hz = ((delta / 2) / (np.pi / 2)) * 1 / 2 * Z_1q
# h = hx + hz
# for t in t_slice[1:]:
#     x_pi2 = np.dot(expm(-1j * h * t_slice[1]), x_pi2)
#
# x_pi2m = I_1q
# hx = -1 / 2 * X_1q
# hz = delta / 2 / (np.pi / 2) * 1 / 2 * Z_1q
# h = hx + hz
# for t in t_slice[1:]:
#     x_pi2m = np.dot(expm(-1j * h * t_slice[1]), x_pi2m)
#
# p = pulse_noisy_x(1)
# # p = [Pulse(pulse_type='pulse', axis='X', sign=1, angle='pi/2')]
#
# for i in range(len(p)):
#     print_pulse(p[i])
# m1 = np.identity(2)
# for i in reversed(range(len(p))):
#     m1 = pulse_to_unitary(p[i], delta / 2) @ m1
#
#
# delta_tilde = delta / (2 * (np.pi / 2))
# omega = np.sqrt(1 + delta_tilde**2)
# m2 = np.cos(np.pi/4 * omega) * I_1q - 1j * np.sin(np.pi/4 * omega) * (1 / omega * X_1q + delta_tilde / omega * Z_1q)
#
# print(gate_fidelity_1q(x_pi2, m2))
# print(gate_fidelity_1q(x_pi2, m1))

# test block end here

