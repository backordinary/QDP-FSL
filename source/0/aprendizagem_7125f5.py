# https://github.com/thiagom123/Circuito-Quantico-Parametrizado/blob/0866c7cf39a42ff5d446b8c02f7f943563e530b0/Aprendizagem.py
import numpy as np
from qiskit import execute, Aer, QuantumRegister, QuantumCircuit
from gates import Rz, Ry, H
from CircuitoEncoder import CircuitoEncoder
from CircuitoParametrizado import TesteHadamard, CQV
from qiskit import Aer, transpile, assemble

from qiskit import IBMQ, BasicAer, Aer
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.providers.aer.noise import NoiseModel


NUM_SHOTS = 10000

np.random.seed(999999)


def get_probability_distribution(counts):
    output_distr = [v / NUM_SHOTS for v in counts.values()]
    if len(output_distr) == 1:
        output_distr.append(1 - output_distr[0])
    return output_distr


def valor_esperado(theta, x):
    '''

    :param theta:
    :param x:
    :return:
    '''
    # Obtain a quantum circuit instance from the paramters
    qc = CQV(theta, x)
    backend = Aer.get_backend("qasm_simulator")
    # Execute the quantum circuit to obtain the probability distribution associated with the current parameters
    t_qc = transpile(qc, backend)
    qobj = assemble(t_qc, shots=NUM_SHOTS)
    result = backend.run(qobj).result()
    # Obtain the counts for each measured state, and convert those counts into a probability vector
    output_distr = get_probability_distribution(result.get_counts(qc))
    # Calcula o Valor esperado do operador Z
    Mx1 = +1 * output_distr[0] - 1 * output_distr[1]
    return Mx1


def gradient_descent(x, y, epochs, theta_inicial):
    learning_rate = 0.02
    theta = theta_inicial
    print(theta)
    n_theta = len(theta)
    n_samples = len(x)
    print("N Samples:")
    print(n_samples)
    dimensao = 1

    params = 0.5

    for e in range(epochs):
        y_pred = np.zeros((n_samples, dimensao))
        ValorEsperado = np.zeros(n_samples)
        DerivadaValorEsperado = np.zeros((n_samples, n_theta))
        ValorEsperadoPlus = np.zeros(n_theta)
        ValorEsperadoMinus = np.zeros(n_theta)
        for i in range(n_samples):
            Param_Shift= [[np.pi/2, 0, 0], [0, np.pi/2, 0], [0, 0, np.pi/2]]
            ValorEsperado[i] = valor_esperado(theta, x[i])
            for j in range(n_theta):
                thetaP = np.zeros(n_theta)
                thetaM = np.zeros(n_theta)
                for m in range(n_theta):
                    thetaP[m] = theta[m] + Param_Shift[j][m]
                    thetaM[m] = theta[m] - Param_Shift[j][m]
                ValorEsperadoPlus[j] = valor_esperado(thetaP, x[i])
                ValorEsperadoMinus[j] = valor_esperado(thetaM, x[i])
                DerivadaValorEsperado[i][j] = (ValorEsperadoPlus[j] - ValorEsperadoMinus[j]) / 2.0
            y_pred[i] = params * ValorEsperado[i]
        # Calculando as derivadas
        correcao_theta = np.zeros(n_theta)
        correcao_params = 0.0
        print("Epoch", e)
        print("Y:", y_pred)
        print("Theta:", theta )
        print("Param:", params)
        for k in range(n_theta):
            for i in range(n_samples):
                Del_y_DelTheta = DerivadaValorEsperado[i][0] * params
                correcao_theta[k] += 2.0 * (y_pred[i] - y[i]) * Del_y_DelTheta
            correcao_theta[k] = correcao_theta[k] / n_samples
        for i in range(n_samples):
            Del_y_DelParams = ValorEsperado[i]
            correcao_params += 2.0 * (y_pred[i] - y[i]) * Del_y_DelParams
        correcao_params = correcao_params/n_samples

        for k in range(n_theta):
            theta[k] = theta[k] - learning_rate * correcao_theta[k]
        params = params - learning_rate * correcao_params

        # print(history_theta)
        # print(history_params)
    return [theta, params, y_pred]



if __name__ == '__main__':
    X_train = [0.4, 0.6, 0.8]
    Y_train = [0.4, 0.6, 0.8]
    # Initialize the COBYLA optimizer
    res = gradient_descent(x=X_train, y=Y_train, epochs=30, theta_inicial=[np.pi / 3, np.pi/3, np.pi/3])
    # Create the initial parameters (noting that our single qubit variational form has 3 parameters)
    print("Y Calculados:")
    print(res[2])
