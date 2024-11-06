# https://github.com/Aderson10086/Hidden-Quantum-Markov-model-and-its-extensions/blob/8d8f8ae03e81f8e252fea769c37013803331439a/HQMM_Backup.py
# 求解hidden quantum Markov model 和 split hidden quantum Markov model的代码
import autograd.numpy as np
from autograd import elementwise_grad as egrad
from matplotlib import pyplot as plt
from tqdm import tqdm
from numpy.linalg import matrix_rank as rank
from qiskit import *
from qiskit.circuit import Parameter
from qiskit.quantum_info import partial_trace
from autograd.numpy import pi
from matplotlib.patches import Circle
import os
import sys


class hidden_quantum_Markov_model_setup:
    """
    hidden_quantum_Markov_model_setup
    data, vla_data, test_data,必须全部是待训练的整型数据
    maxIter:最大迭代步数，默认取值是10，可以调整，必须是整型
    tau:学习率因子，默认是0.95，可以调整必须是0-1之间的小数
    alpha:学习率衰减因子，默认是0.95，可以调整必须是0-1之间的小数
    beta:记忆因子，记忆上次计算的梯度值在本次计算中应该保留多少，默认是0.90，可以调整，必须是0-1之间的小数
    batch_num:训练数据的batch数目，可以调整，整型数据
    batch_num_val:验证数据的batch数目，可以调整，整型数据
    batch_num_test:测试数据的batch数据，可以调整，整型数据
    len_each_batch:每一个batch所对应的数据长度，整型
    qubits:需要的计算比特数目，目前只支持1比特和2比特的计算，整型
    random_seed:初始化Kraus算符所需要的随机数种子，默认值为float
    output_dims:模型输出的状态空间的维度，只能取整型
    class_num:Kraus算符的种类，整型数据
    space:生成的Kraus算符所在的空间，只能取值为Complex或Real，默认值是Complex
    """

    def __init__(self, data, val_data, test_data, maxIter=10, tau=0.95, alpha=0.95, beta=0.90,
                 batch_num=None, batch_num_val=None,
                 batch_num_test=None, len_each_batch=None, qubits=1, random_seed=None, output_dims=None, class_num=None,
                 space="Complex", random_seed_for_density=None):
        self.data = data
        self.val_data = val_data
        self.test_data = test_data
        self.maxIter = maxIter
        self.tau = tau
        self.tau_copy = tau
        self.alpha = alpha
        self.beta = beta
        self.batch_num = batch_num
        self.batch_num_val = batch_num_val
        self.batch_num_test = batch_num_test
        self.len_each_batch = len_each_batch
        self.qubits = qubits
        self.dim_K = 2 ** qubits
        self.random_seed = random_seed
        self.output_dims = output_dims
        self.class_num = class_num
        self.space = space
        self.random_seed_for_density = random_seed_for_density
        self.check_data()

    def data_Pre(self, mode="train"):
        """
        预处理输入的数据，把数据重新改变大小为 batch数 * 长度
        :param mode: 可选参数，"train","validation","test"分别对应预处理训练数据，验证数据，测试数据
        :return: 预处理后的数据
        """
        if mode == "train":
            train_seq = self.data.reshape(self.batch_num, self.len_each_batch)
            return train_seq
        if mode == "validation":
            val_seq = self.val_data.reshape(self.batch_num_val, self.len_each_batch)
            return val_seq
        if mode == "test":
            test_seq = self.test_data.reshape(self.batch_num_test, self.len_each_batch)
            return test_seq

    def check_data(self):
        """
        检查数据类型是否正确，目前只能处理离散的数据，所以数据类型只能取int型，根据机器的不同int32或int64
        :return: None
        """
        if self.data.dtype != "int64":
            raise TypeError("待训练的数据必须是整型")

    @staticmethod
    def dagger(M):
        """
        计算矩阵M的dagger：先进行元素共轭，然后进行转置操作
        :param M: 矩阵
        :return: 矩阵M的dagger
        """
        return np.transpose(np.conj(M))

    def DA(self, L_train, seq_len=200):
        """
        计算模型的DA值
        :param L_train:计算出来的似然函数值
        :param seq_len: 带入训练的每个batch的序列长度再减去burnin的长度
        :return: 似然函数值
        """

        def fun(t):
            if t > 0:
                return t
            else:
                f = (1 - np.exp(-0.25 * t)) / (1 + np.exp(-0.25 * t))
            return f

        z = fun(1 - L_train / (np.log(self.output_dims) * seq_len))
        return z

    @staticmethod
    def Linear_Dependent(M):
        """
        判断一个矩阵中的行向量是否线性相关
        :param M: 矩阵
        :return: 布尔值，True:线性相关， False:线性无关
        """
        if rank(M) < min(M.shape):
            return True
        else:
            return False

    @staticmethod
    def KMR(Kraus, rho):
        "计算Kraus算符夹密度矩阵"
        z = np.dot(np.conj(Kraus), np.dot(rho, np.transpose(Kraus)))
        return z

    def Generate_Kraus(self):
        """
        产生初始的的Kraus算符
        :return:
        """

        np.random.seed(self.random_seed)
        if self.space == "Real":
            A = np.random.random([self.dim_K * self.output_dims * self.class_num, self.dim_K])
        elif self.space == "Complex":
            A = np.random.random([self.dim_K * self.output_dims * self.class_num, self.dim_K * 2])
        else:
            raise ValueError("参数space只能接受Real或者Complex")
        ref = self.Linear_Dependent(A)
        B = np.transpose(A)
        if ref:
            print("Linear Dependent!")
            quit()
        else:
            # 对每一行进行施密特正交化,第一行为基础行，不进行正交
            for i in range(1, B.shape[0]):
                for j in range(0, i):
                    B[i, :] = B[i, :] - np.dot(B[i, :], B[j, :].T) / (np.linalg.norm(B[j, :], ord=2) ** 2) * B[j, :]

        if self.space == "Real":
            for i in range(0, B.shape[0]):
                B[i, :] = B[i, :] / np.linalg.norm(B[i, :], ord=2)
            return B.T
        else:
            C = np.zeros([self.dim_K * self.output_dims * self.class_num, self.dim_K], dtype=complex).T
            for i in range(0, B.shape[0], 2):
                C[int(i / 2), :] = B[i, :] + 1j * B[i + 1, :]
                C[int(i / 2), :] = C[int(i / 2), :] / np.linalg.norm(C[int(i / 2), :], ord=2)
            return C.T

    def Initial_Density_Matrix(self):
        """
        用量子线路产生初始的密度矩阵，目前只能产生1比特和2比特的密度矩阵
        :return: 密度矩阵
        """
        theta0, theta1, theta2 = Parameter('θ0'), Parameter('θ1'), Parameter('Θ2')
        qr = QuantumRegister(4, 'q')
        qc = QuantumCircuit(qr)
qc.h(qr[0])qc.h(qr[1])qc.h(qr[2])qc.h(qr[3])        qc.crx(theta0, qr[0], qr[1])
        qc.cry(theta1, qr[1], qr[2])
        qc.crz(theta2, qr[2], qr[3])
        np.random.seed(self.random_seed_for_density)
        parameters = np.random.rand(1, 3)
        parameters = 2 * pi * parameters
        job = execute(qc, backend=BasicAer.get_backend('statevector_simulator'),
                      parameter_binds=[{theta0: parameters[0, 0], theta1: parameters[0, 1], theta2: parameters[0, 2]}],
                      shots=5000, seed_simulator=100, seed_transpiler=100)
        value = job.result().get_statevector()
        if self.dim_K == 2:
            subsystem_density_matrix = partial_trace(value, qargs=[0, 1, 2])
        elif self.dim_K == 4:
            subsystem_density_matrix = partial_trace(value, qargs=[0, 1])
        elif self.dim_K == 8:
            subsystem_density_matrix = partial_trace(value, qargs=[0])

        else:
            raise ValueError("当前输入的比特只支持单比特计算和双比特")
        return subsystem_density_matrix._data


class hidden_quantum_Markov_model(hidden_quantum_Markov_model_setup):

    def __init__(self, data, val_data, test_data, max_iter=10, tau=0.75, alpha=0.92, beta=0.9, batch_num=200
                 , len_each_batch=300, qubits=1, random_seed=100, output_dims=6, class_num=1, space="Complex",
                 random_seed_for_density=80):
        super(hidden_quantum_Markov_model, self).__init__(data, val_data, test_data, maxIter=max_iter,
                                                          tau=tau, alpha=alpha, beta=beta, batch_num=batch_num,
                                                          len_each_batch=len_each_batch,
                                                          qubits=qubits, random_seed=random_seed,
                                                          output_dims=output_dims, class_num=class_num, space=space,
                                                          random_seed_for_density=random_seed_for_density)

        rho_initial = self.Initial_Density_Matrix()
        self.rho_initial = rho_initial
        self.rho_initial_copy = rho_initial
        Kraus_initial = self.Generate_Kraus()
        self.Kraus_initial = Kraus_initial
        # 自动调用检查数据格式的函数
        self.check_initial()

    def check_initial(self):
        eps = 1e-5
        if np.trace(self.rho_initial) >= 1 + eps or np.trace(self.rho_initial) < 1 - eps or \
                self.dagger(self.rho_initial).any() != self.rho_initial.any():
            raise ValueError("密度矩阵初始化失败，初始化密度矩阵不满足条件！")
        if np.dot(self.dagger(self.Kraus_initial), self.Kraus_initial).any() != np.identity(2).any():
            raise ValueError("Kraus算符初始化失败，不满足Stiefel流形的条件！")

    @staticmethod
    def like_hood(Kraus, rho_initial=0, train_data_seq=0, burnin=100):
        """

        :param Kraus: Kraus算符矩阵
        :param rho_initial:初始化密度矩阵
        :param train_data_seq:训练数据
        :param burnin: 预处理的数据个数
        :return: 似然函数值
        """
        # burn in
        rho = 0
        for i in train_data_seq[0:burnin]:
            rho_new = np.dot(np.conj(Kraus[2 * i - 2:2 * i, :]),
                             np.dot(rho_initial, np.transpose(Kraus[2 * i - 2:2 * i, :])))
            rho_initial = rho_new / np.trace(rho_new)
        # calculate like-hood
        for j in train_data_seq[burnin:]:
            rho = np.dot(np.conj(Kraus[2 * j - 2:2 * j, :]),
                         np.dot(rho_initial, np.transpose(Kraus[2 * j - 2:2 * j, :])))
            rho_initial = rho
        p = np.trace(rho)
        return -np.log(np.real(p))

    def __compute_like_hood_grad_fun(self):
        self.grad_like_hood = egrad(self.like_hood, 0)
        self.grad_fun = self.grad_like_hood
        return self.grad_fun

    def compute_grad_of_like_hood(self, train_data_seq):
        func = self.__compute_like_hood_grad_fun()
        return func(np.conj(self.Kraus_initial), rho_initial=self.rho_initial, train_data_seq=train_data_seq,
                    burnin=100)

    def Iteration_Step(self):
        DA_value = []
        G_old = 0
        for i in tqdm(range(self.maxIter)):
            loss = []
            for j in range(self.batch_num):
                data_train_seq = self.data_Pre(mode="train")[j, :]
                G = self.compute_grad_of_like_hood(train_data_seq=data_train_seq)
                F = np.linalg.norm(G, ord=2)
                G = G / F
                G = self.beta * G_old + (1 - self.beta) * G
                E = np.linalg.norm(G, ord=2)
                G = G / E
                U = np.hstack((G, self.Kraus_initial))
                V = np.hstack((self.Kraus_initial, -G))
                Inverse = np.identity(2 * self.dim_K) + self.tau / 2 * np.dot(self.dagger(V), U)
                item1 = np.dot(U, np.linalg.inv(Inverse))
                item2 = np.dot(self.dagger(V), self.Kraus_initial)
                self.Kraus_initial = self.Kraus_initial - self.tau * np.dot(item1, item2)
                G_old = G
                L_train = self.like_hood(np.conj(self.Kraus_initial), rho_initial=self.rho_initial,
                                         train_data_seq=data_train_seq,
                                         burnin=100)
                loss.append(L_train)
            DA_value.append(self.DA(np.mean(loss)))
            self.tau = self.alpha * self.tau
        return DA_value, self.Kraus_initial

    def show_Result(self):
        DA_value, Kraus = self.Iteration_Step()
        plt.plot(range(self.maxIter), DA_value)
        plt.show()
        print("最终迭代出来的Kraus算符为：")
        print(Kraus)


class One_Hot_Encoding_of_Float_Number:
    """
    把任意的数字进行one-hot编码成矩阵，顺序为：'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.'
    """

    def __init__(self, num):
        self.value = num
        self.__query = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']

    def __transform(self):
        num_str = str(self.value)
        Encoding_Matrix = np.zeros([len(num_str), len(self.__query)])
        for i in range(len(num_str)):
            for j in range(len(self.__query)):
                if num_str[i] == self.__query[j]:
                    Encoding_Matrix[i, j] = 1
                    break

        return Encoding_Matrix

    def get_Encoding_result(self):
        result = self.__transform()
        return result


class split_hidden_quantum_Markov_model(hidden_quantum_Markov_model_setup):
    """
    计算split hidden quantum Markov model
    """

    def __init__(self, data, val_data, test_data,
                 maxIter=10, tau=0.95, alpha=0.95, beta=0.90, batch_num=200, batch_num_val=100,
                 batch_num_test=100, len_each_batch=300, qubits=1, random_seed=100, output_dims=6, class_num=3,
                 space="Complex", random_seed_for_density=80, IP=3):
        super(split_hidden_quantum_Markov_model, self).__init__(data, val_data, test_data, maxIter=maxIter,
                                                                tau=tau, alpha=alpha, beta=beta, batch_num=batch_num,
                                                                batch_num_val=batch_num_val,
                                                                batch_num_test=batch_num_test,
                                                                len_each_batch=len_each_batch, qubits=qubits,
                                                                random_seed=random_seed,
                                                                output_dims=output_dims, class_num=class_num,
                                                                space=space,
                                                                random_seed_for_density=random_seed_for_density)
        self.IP = IP
        Kappa_initial = self.Generate_Kraus()
        if self.class_num == 3:
            K_initial, R_initial, A_initial = self.Kappa_to_Kraus(Kappa_initial)
            self.K_initial, self.R_initial, self.A_initial = K_initial, R_initial, A_initial
            self.K_initial_copy, self.R_initial_copy, self.A_initial_copy = K_initial, R_initial, A_initial
        elif self.class_num == 5:
            K_initial, R_initial, A_initial, U_initial, S_initial = self.Kappa_to_Kraus(Kappa_initial)
            self.K_initial, self.R_initial, self.A_initial, self.U_initial, self.S_initial = K_initial, R_initial, \
                                                                                             A_initial, U_initial, S_initial
            self.K_initial_copy, self.R_initial_copy, self.A_initial_copy, self.U_initial_copy, \
            self.S_initial_copy = K_initial, R_initial, A_initial, U_initial, S_initial
        elif self.class_num == 7:
            K_initial, R_initial, A_initial, U_initial, S_initial, V_initial, Gamma_initial = self.Kappa_to_Kraus(Kappa_initial)
            self.K_initial, self.R_initial, self.A_initial, self.U_initial, self.S_initial, self.V_initial, self.Gamma_initial = K_initial, R_initial, \
                                                                                             A_initial, U_initial, S_initial, V_initial, Gamma_initial
            self.K_initial_copy, self.R_initial_copy, self.A_initial_copy, self.U_initial_copy, \
            self.S_initial_copy, self.V_initial_copy, self.Gamma_initial_copy = K_initial, R_initial, A_initial, U_initial, S_initial, V_initial, Gamma_initial
        rho_initial = self.Initial_Density_Matrix()
        self.rho_initial = rho_initial
        self.rho_initial_copy = rho_initial
        self.check_initial(Kappa_initial)

    def check_initial(self, Kappa):
        eps = 1e-5
        if np.trace(self.rho_initial) >= 1 + eps or np.trace(self.rho_initial) <= 1 - eps or self.dagger(
                self.rho_initial).any() != self.rho_initial.any():
            raise ValueError("密度矩阵初始化失败，初始化密度矩阵不满足条件！")
        if np.dot(self.dagger(Kappa), Kappa).any() != np.identity(self.dim_K).any():
            raise ValueError("Kraus算符初始化失败，不满足Stiefel流形的条件！")

    def rho_initial_distribution(self):
        if self.IP == 3:
            a, b = 0.2, 0.5
            if (a + b) > 1 or a<0 or b<0:
                raise ValueError("pesudo density matrix initialize failed!")
            else:
                rho_initial0, rho_initial1, rho_initial2 = a * self.rho_initial, b * self.rho_initial, \
                                                           (1 - a - b) * self.rho_initial
                return rho_initial0, rho_initial1, rho_initial2
        if self.IP == 4:
            a, b, c = 0.2, 0.5, 0.1
            if a + b + c > 1 or a < 0 or b < 0 or c < 0:
                raise ValueError("pesudo density matrix initialize failed!")
            else:
                rho_initial0, rho_initial1, rho_initial2, rho_initial3 = a * self.rho_initial, b * self.rho_initial, c \
                                                                         * self.rho_initial, (1 - a - b - c) \
                                                                         * self.rho_initial
            return rho_initial0, rho_initial1, rho_initial2, rho_initial3

    def compute_like_hood(self, train_data_seq, mode="gradient"):
        """
        计算似然函数及其梯度
        :param train_seq_data: 数据
        :param mode: 两个取值gradient和value分别返回梯度值和似然函数值
        :return:
        """
        if self.class_num == 3 and self.IP == 3:
            def like_hood(K, R, A, rho_initial0=0, rho_initial1=0, rho_initial2=0, train_data_seq=0, burnin=100):
                # 添加边界条件补全的项数
                # burn in
                for i in train_data_seq[0:burnin]:
                    rho0 = np.add(self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0),
                                  np.add(
                                      self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1),
                                      self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2)))

                    rho1 = np.add(self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1),
                                  np.add(
                                      self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0),
                                      self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2)))

                    rho2 = np.add(self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2),
                                  np.add(
                                      self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1),
                                      self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0)))
                    rho_total = np.add(rho0, np.add(rho1, rho2))
                    P = np.trace(rho_total)
                    rho0, rho1, rho2 = rho0 / P, rho1 / P, rho2 / P
                    rho_initial0, rho_initial1, rho_initial2 = rho0, rho1, rho2
                # calculate like_hood
                for i in train_data_seq[burnin:]:
                    rho0 = np.add(self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0),
                                  np.add(
                                      self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1),
                                      self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2)))

                    rho1 = np.add(self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1),
                                  np.add(
                                      self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0),
                                      self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2)))

                    rho2 = np.add(self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2),
                                  np.add(
                                      self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1),
                                      self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0)))

                    rho_initial0, rho_initial1, rho_initial2 = rho0, rho1, rho2

                rho_total = np.add(rho_initial0, np.add(rho_initial1, rho_initial2))
                P = np.trace(rho_total)
                return -np.log(np.real(P))

            rho_initial0, rho_initial1, rho_initial2 = self.rho_initial_distribution()
            if mode == "gradient":
                grad_fn_K, grad_fn_R, grad_fn_A = egrad(like_hood, 0), egrad(like_hood, 1), egrad(like_hood, 2)
                grad_fn_K_value = grad_fn_K(np.conj(self.K_initial), np.conj(self.R_initial), np.conj(self.A_initial),
                                            rho_initial0=rho_initial0, rho_initial1=rho_initial1,
                                            rho_initial2=rho_initial2,
                                            train_data_seq=train_data_seq, burnin=100)
                grad_fn_R_value = grad_fn_R(np.conj(self.K_initial), np.conj(self.R_initial), np.conj(self.A_initial),
                                            rho_initial0=rho_initial0, rho_initial1=rho_initial1,
                                            rho_initial2=rho_initial2,
                                            train_data_seq=train_data_seq, burnin=100)
                grad_fn_A_value = grad_fn_A(np.conj(self.K_initial), np.conj(self.R_initial), np.conj(self.A_initial),
                                            rho_initial0=rho_initial0, rho_initial1=rho_initial1,
                                            rho_initial2=rho_initial2,
                                            train_data_seq=train_data_seq, burnin=100)
                return grad_fn_K_value, grad_fn_R_value, grad_fn_A_value

            if mode == "value":
                like_hood_value = like_hood(np.conj(self.K_initial), np.conj(self.R_initial), np.conj(self.A_initial),
                                            rho_initial0=rho_initial0, rho_initial1=rho_initial1,
                                            rho_initial2=rho_initial2,
                                            train_data_seq=train_data_seq, burnin=100)
                return like_hood_value

        elif self.class_num == 3 and self.IP == 4:
            def like_hood(K, R, A, rho_initial0=0, rho_initial1=0, rho_initial2=0, rho_initial3=0, train_data_seq=0,
                          burnin=100):
                for i in train_data_seq[0:burnin]:
                    rho0 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3)

                    rho1 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2)

                    rho2 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                           self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3)

                    rho3 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3) + \
                           self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0)

                    rho_total = rho0 + rho1 + rho2 + rho3
                    P = np.trace(rho_total)
                    rho_initial0, rho_initial1, rho_initial2, rho_initial3 = rho0 / P, rho1 / P, rho2 / P, rho3 / P
                for i in train_data_seq[burnin:]:
                    rho0 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3)

                    rho1 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2)

                    rho2 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                           self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3)

                    rho3 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3) + \
                           self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0)
                    rho_initial0, rho_initial1, rho_initial2, rho_initial3 = rho0, rho1, rho2, rho3
                rho_total = rho_initial0 + rho_initial1 + rho_initial2 + rho_initial3
                P = np.trace(rho_total)
                return -np.log(np.real(P))

            rho_initial0, rho_initial1, rho_initial2, rho_initial3 = self.rho_initial_distribution()
            if mode == "gradient":
                grad_fn_K, grad_fn_R, grad_fn_A = egrad(like_hood, 0), egrad(like_hood, 1), egrad(like_hood, 2)
                grad_fn_K_value = grad_fn_K(self.K_initial, self.R_initial, self.A_initial, rho_initial0=rho_initial0,
                                            rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            rho_initial3=rho_initial3,
                                            train_data_seq=train_data_seq, burnin=100)
                grad_fn_R_value = grad_fn_R(self.K_initial, self.R_initial, self.A_initial, rho_initial0=rho_initial0,
                                            rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            rho_initial3=rho_initial3,
                                            train_data_seq=train_data_seq, burnin=100)
                grad_fn_A_value = grad_fn_A(self.K_initial, self.R_initial, self.A_initial, rho_initial0=rho_initial0,
                                            rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            rho_initial3=rho_initial3,
                                            train_data_seq=train_data_seq, burnin=100)
                return grad_fn_K_value, grad_fn_R_value, grad_fn_A_value
            elif mode == "value":
                like_hood_value = like_hood(self.K_initial, self.R_initial, self.A_initial, rho_initial0=rho_initial0,
                                            rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            rho_initial3=rho_initial3,
                                            train_data_seq=train_data_seq, burnin=100)
                return like_hood_value
        elif self.class_num == 5 and self.IP == 4:
            def like_hood(K, R, A, U, S, rho_initial0=0, rho_initial1=0, rho_initial2=0, rho_initial3=0,
                          train_data_seq=0, burnin=100):
                for i in train_data_seq[0:burnin]:
                    rho0 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(U[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                           self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3) + \
                           self.KMR(S[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2)

                    rho1 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                           self.KMR(U[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3) + \
                           self.KMR(S[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3)

                    rho2 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                           self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(S[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3) + \
                           self.KMR(U[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0)

                    rho3 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3) + \
                           self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                           self.KMR(S[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(U[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1)

                    rho_total = rho0 + rho1 + rho2 + rho3
                    P = np.trace(rho_total)
                    rho_initial0, rho_initial1, rho_initial2, rho_initial3 = rho0 / P, rho1 / P, rho2 / P, rho3 / P

                for i in train_data_seq[burnin:]:
                    rho0 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(U[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                           self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3) + \
                           self.KMR(S[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2)

                    rho1 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                           self.KMR(U[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3) + \
                           self.KMR(S[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3)

                    rho2 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                           self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(S[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3) + \
                           self.KMR(U[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0)

                    rho3 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3) + \
                           self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                           self.KMR(S[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(U[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1)

                    rho_initial0, rho_initial1, rho_initial2, rho_initial3 = rho0, rho1, rho2, rho3
                rho_total = rho_initial0 + rho_initial1 + rho_initial2 + rho_initial3
                return -np.log(np.real(np.trace(rho_total)))

            rho_initial0, rho_initial1, rho_initial2, rho_initial3 = self.rho_initial_distribution()
            if mode == "gradient":
                grad_fn_K, grad_fn_R, grad_fn_A, grad_fn_U, grad_fn_S = egrad(like_hood, 0), egrad(like_hood, 1), \
                                                                        egrad(like_hood, 2), egrad(like_hood, 3), egrad(
                    like_hood, 4)
                grad_fn_K_value = grad_fn_K(np.conj(self.K_initial), np.conj(self.R_initial), np.conj(self.A_initial),
                                            np.conj(self.U_initial), np.conj(self.S_initial), rho_initial0=rho_initial0,
                                            rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            rho_initial3=rho_initial3,
                                            train_data_seq=train_data_seq, burnin=100)

                grad_fn_R_value = grad_fn_R(np.conj(self.K_initial), np.conj(self.R_initial), np.conj(self.A_initial),
                                            np.conj(self.U_initial), np.conj(self.S_initial), rho_initial0=rho_initial0,
                                            rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            rho_initial3=rho_initial3,
                                            train_data_seq=train_data_seq, burnin=100)

                grad_fn_A_value = grad_fn_A(np.conj(self.K_initial), np.conj(self.R_initial), np.conj(self.A_initial),
                                            np.conj(self.U_initial), np.conj(self.S_initial), rho_initial0=rho_initial0,
                                            rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            rho_initial3=rho_initial3,
                                            train_data_seq=train_data_seq, burnin=100)

                grad_fn_U_value = grad_fn_U(np.conj(self.K_initial), np.conj(self.R_initial), np.conj(self.A_initial),
                                            np.conj(self.U_initial), np.conj(self.S_initial), rho_initial0=rho_initial0,
                                            rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            rho_initial3=rho_initial3,
                                            train_data_seq=train_data_seq, burnin=100)
                grad_fn_S_value = grad_fn_S(np.conj(self.K_initial), np.conj(self.R_initial), np.conj(self.A_initial),
                                            np.conj(self.U_initial), np.conj(self.S_initial), rho_initial0=rho_initial0,
                                            rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            rho_initial3=rho_initial3,
                                            train_data_seq=train_data_seq, burnin=100)
                return grad_fn_K_value, grad_fn_R_value, grad_fn_A_value, grad_fn_U_value, grad_fn_S_value
            if mode == "value":
                like_hood_value = like_hood(np.conj(self.K_initial), np.conj(self.R_initial), np.conj(self.A_initial),
                                            np.conj(self.U_initial), np.conj(self.S_initial), rho_initial0=rho_initial0,
                                            rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            rho_initial3=rho_initial3,
                                            train_data_seq=train_data_seq, burnin=100)
                return like_hood_value
        elif self.class_num == 5 and self.IP == 3:
            def like_hood(K, R, A, U ,S, rho_initial0=0, rho_initial1=0, rho_initial2=0, train_data_seq=0, burnin=100):
                for i in train_data_seq[0:burnin]:
                    rho0 =  self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                            self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                            self.KMR(U[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                            self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                            self.KMR(S[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1)

                    rho1 =  self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                            self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                            self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                            self.KMR(U[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                            self.KMR(S[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2)

                    rho2 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                           self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(S[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(U[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1)
                    rho_total = rho0 + rho1 + rho2
                    P = np.trace(rho_total)
                    rho_initial0, rho_initial1, rho_initial2 = rho0 / P, rho1 / P, rho2 / P
                for i in train_data_seq[burnin:]:
                    rho0 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(U[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                           self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                           self.KMR(S[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1)

                    rho1 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                           self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(U[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(S[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2)

                    rho2 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                           self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(S[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(U[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1)
                    rho_initial0, rho_initial1, rho_initial2 = rho0, rho1, rho2
                P = np.trace(rho_initial0 + rho_initial1 + rho_initial2)
                return -np.log(np.real(P))
            rho_initial0, rho_initial1, rho_initial2 = self.rho_initial_distribution()
            if mode == "gradient":
                grad_fn_K, grad_fn_R, grad_fn_A, grad_fn_U, grad_fn_S= egrad(like_hood, 0), egrad(like_hood, 1), egrad(like_hood, 2), egrad(like_hood, 3), egrad(like_hood, 4)

                grad_fn_K_value = grad_fn_K(self.K_initial, self.R_initial, self.A_initial,self.U_initial, self.S_initial, rho_initial0=rho_initial0,
                                            rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            train_data_seq=train_data_seq, burnin=100)

                grad_fn_R_value = grad_fn_R(self.K_initial, self.R_initial, self.A_initial,self.U_initial, self.S_initial, rho_initial0=rho_initial0,
                                            rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            train_data_seq=train_data_seq, burnin=100)

                grad_fn_A_value = grad_fn_A(self.K_initial, self.R_initial, self.A_initial,self.U_initial, self.S_initial, rho_initial0=rho_initial0,
                                            rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            train_data_seq=train_data_seq, burnin=100)

                grad_fn_U_value = grad_fn_U(self.K_initial, self.R_initial, self.A_initial,self.U_initial, self.S_initial, rho_initial0=rho_initial0,
                                            rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            train_data_seq=train_data_seq, burnin=100)

                grad_fn_S_value = grad_fn_S(self.K_initial, self.R_initial, self.A_initial,self.U_initial, self.S_initial, rho_initial0=rho_initial0,
                                            rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            train_data_seq=train_data_seq, burnin=100)


                return grad_fn_K_value, grad_fn_R_value, grad_fn_A_value, grad_fn_U_value, grad_fn_S_value
            elif mode == "value":
                like_hood_value = like_hood(self.K_initial, self.R_initial, self.A_initial,self.U_initial, self.S_initial, rho_initial0=rho_initial0,
                                            rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            train_data_seq=train_data_seq, burnin=100)
                return like_hood_value

        elif self.class_num == 7 and self.IP == 4:
            def like_hood(K, R, A, S, U, V, Gamma, rho_initial0=0, rho_initial1=0, rho_initial2=0, rho_initial3=0, train_data_seq=0, burnin=100):
                for i in train_data_seq[0:burnin]:
                    rho0 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(U[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + self.KMR(Gamma[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3) + \
                           self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3) + self.KMR(S[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                           self.KMR(V[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1)

                    rho1 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + self.KMR(U[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3) + \
                           self.KMR(S[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3) + self.KMR(Gamma[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(V[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2)

                    rho2 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(S[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3) + \
                           self.KMR(U[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + self.KMR(Gamma[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(V[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3)

                    rho3 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3) + self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                           self.KMR(S[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + self.KMR(V[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + self.KMR(U[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(Gamma[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2)
                    P = np.trace(rho0 + rho1 + rho2 + rho3)
                    rho_initial0, rho_initial1, rho_initial2, rho_initial3 = rho0 / P, rho1 / P, rho2 / P, rho3 / P
                for i in train_data_seq[burnin:]:
                    rho0 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + self.KMR(
                        A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(U[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + self.KMR(
                        Gamma[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3) + \
                           self.KMR(R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3) + self.KMR(
                        S[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                           self.KMR(V[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1)

                    rho1 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + self.KMR(
                        R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + self.KMR(
                        U[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3) + \
                           self.KMR(S[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3) + self.KMR(
                        Gamma[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(V[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2)

                    rho2 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + self.KMR(
                        R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(S[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + self.KMR(
                        A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3) + \
                           self.KMR(U[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + self.KMR(
                        Gamma[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(V[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3)

                    rho3 = self.KMR(K[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial3) + self.KMR(
                        R[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2) + \
                           self.KMR(S[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + self.KMR(
                        V[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + \
                           self.KMR(A[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial0) + self.KMR(
                        U[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial1) + \
                           self.KMR(Gamma[self.dim_K * i - self.dim_K:self.dim_K * i, :], rho_initial2)
                    rho_initial0, rho_initial1, rho_initial2, rho_initial3 = rho0, rho1, rho2, rho3
                P = np.trace(rho_initial0 + rho_initial1 + rho_initial2 + rho_initial3)
                return -np.log(np.real(P))
            rho_initial0, rho_initial1, rho_initial2, rho_initial3 = self.rho_initial_distribution()
            if mode == "gradient":
                grad_fn_K, grad_fn_R, grad_fn_A, grad_fn_S, grad_fn_U, grad_fn_V, grad_fn_Gamma = egrad(like_hood, 0) ,\
                                                    egrad(like_hood, 1), egrad(like_hood, 2), egrad(like_hood, 3), egrad(like_hood, 4), egrad(like_hood, 5), egrad(like_hood, 6)
                grad_fn_K_value = grad_fn_K(self.K_initial, self.R_initial, self.A_initial, self.S_initial, self.U_initial, self.V_initial, self.Gamma_initial,
                                            rho_initial0=rho_initial0, rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            rho_initial3=rho_initial3, train_data_seq=train_data_seq, burnin=100)

                grad_fn_R_value = grad_fn_R(self.K_initial, self.R_initial, self.A_initial, self.S_initial, self.U_initial, self.V_initial, self.Gamma_initial,
                                            rho_initial0=rho_initial0, rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            rho_initial3=rho_initial3, train_data_seq=train_data_seq, burnin=100)

                grad_fn_A_value = grad_fn_A(self.K_initial, self.R_initial, self.A_initial, self.S_initial, self.U_initial, self.V_initial, self.Gamma_initial,
                                            rho_initial0=rho_initial0, rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            rho_initial3=rho_initial3, train_data_seq=train_data_seq, burnin=100)

                grad_fn_S_value = grad_fn_S(self.K_initial, self.R_initial, self.A_initial, self.S_initial, self.U_initial, self.V_initial, self.Gamma_initial,
                                            rho_initial0=rho_initial0, rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            rho_initial3=rho_initial3, train_data_seq=train_data_seq, burnin=100)

                grad_fn_U_value = grad_fn_U(self.K_initial, self.R_initial, self.A_initial, self.S_initial, self.U_initial, self.V_initial, self.Gamma_initial,
                                            rho_initial0=rho_initial0, rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            rho_initial3=rho_initial3, train_data_seq=train_data_seq, burnin=100)
                grad_fn_V_value = grad_fn_V(self.K_initial, self.R_initial, self.A_initial, self.S_initial, self.U_initial, self.V_initial, self.Gamma_initial,
                                            rho_initial0=rho_initial0, rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            rho_initial3=rho_initial3, train_data_seq=train_data_seq, burnin=100)
                grad_fn_Gamma_value = grad_fn_Gamma(self.K_initial, self.R_initial, self.A_initial, self.S_initial, self.U_initial, self.V_initial, self.Gamma_initial,
                                            rho_initial0=rho_initial0, rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            rho_initial3=rho_initial3, train_data_seq=train_data_seq, burnin=100)
                return grad_fn_K_value, grad_fn_R_value, grad_fn_A_value, grad_fn_S_value, grad_fn_U_value, grad_fn_V_value, grad_fn_Gamma_value
            elif mode == "value":
                like_hood_value = like_hood(self.K_initial, self.R_initial, self.A_initial, self.S_initial, self.U_initial, self.V_initial, self.Gamma_initial,
                                            rho_initial0=rho_initial0, rho_initial1=rho_initial1, rho_initial2=rho_initial2,
                                            rho_initial3=rho_initial3, train_data_seq=train_data_seq, burnin=100)
                return like_hood_value
        else:
            raise ValueError("输入有误")

    def Iteration_Step(self, k):
        G_old = 0
        DA_value = []
        if self.class_num == 3:
            Kappa = np.vstack((self.K_initial, self.R_initial, self.A_initial))
        elif self.class_num == 5:
            Kappa = np.vstack((self.K_initial, self.R_initial, self.A_initial, self.U_initial, self.S_initial))
        elif self.class_num == 7:
            Kappa = np.vstack((self.K_initial, self.R_initial, self.A_initial, self.S_initial, self.U_initial, self.V_initial, self.Gamma_initial))
        for i in (range(self.maxIter)):
            loss = []
            # print("percent of K:\n", np.trace(self.__Inner_Product(self.K_initial))/2)
            # print("percent of R:\n", np.trace(self.__Inner_Product(self.R_initial)) /2)
            # print("percent of A:\n", np.trace(self.__Inner_Product(self.R_initial)) /2)
            NumBatch = int(self.data.shape[1] / self.len_each_batch)
            for j in range(NumBatch * k, NumBatch * k + NumBatch, 1):
                train__data_seq = self.data_Pre()[j, :]
                if self.class_num == 3:
                    K_grad, R_grad, A_grad = self.compute_like_hood(train__data_seq, mode="gradient")
                    G = np.vstack((K_grad, R_grad, A_grad))
                elif self.class_num == 5:
                    K_grad, R_grad, A_grad, U_grad, S_grad = self.compute_like_hood(train__data_seq, mode="gradient")
                    G = np.vstack((K_grad, R_grad, A_grad, U_grad, S_grad))
                elif self.class_num == 7:
                    K_grad, R_grad, A_grad, S_grad, U_grad, V_grad, Gamma_grad = self.compute_like_hood(train__data_seq, mode="gradient")
                    G = np.vstack((K_grad, R_grad, A_grad, S_grad, U_grad, V_grad, Gamma_grad))
                F = np.linalg.norm(G, ord=2)
                G = G / F
                G = self.beta * G_old + (1 - self.beta) * G
                E = np.linalg.norm(G, ord=2)
                G = G / E
                U = np.hstack((G, Kappa))
                V = np.hstack((Kappa, -G))
                Inverse = np.identity(2 * self.dim_K) + self.tau / 2 * np.dot(self.dagger(V), U)
                item1 = np.dot(U, np.linalg.inv(Inverse))
                item2 = np.dot(self.dagger(V), Kappa)
                Kappa = Kappa - self.tau * np.dot(item1, item2)
                G_old = G
                if self.class_num == 3:
                    self.K_initial, self.R_initial, self.A_initial = self.Kappa_to_Kraus(Kappa)
                elif self.class_num == 5:
                    self.K_initial, self.R_initial, self.A_initial, self.U_initial, self.S_initial = self.Kappa_to_Kraus(
                        Kappa)
                elif self.class_num == 7:
                    self.K_initial, self.R_initial, self.A_initial, self.S_initial, self.U_initial, self.V_initial, self.Gamma_initial = self.Kappa_to_Kraus(
                        Kappa)

                L_train = self.compute_like_hood(train_data_seq=train__data_seq, mode="value")
                loss.append(L_train)

            DA_value.append(self.DA(np.mean(loss)))
            self.tau = self.alpha * self.tau
        if self.class_num == 3:
            return DA_value, self.K_initial, self.R_initial, self.A_initial
        elif self.class_num == 5:
            return DA_value, self.K_initial, self.R_initial, self.A_initial, self.U_initial, self.S_initial
        elif self.class_num == 7:
            return DA_value, self.K_initial, self.R_initial, self.A_initial, self.S_initial, self.U_initial, self.V_initial, self.Gamma_initial


    def forward(self):
        DA_value_all = []
        model_epoch = self.data.shape[0]
        # model_epoch = 2
        for k in tqdm(range(model_epoch)):
            if self.class_num == 3:
                DA_value, self.K_initial, self.R_initial, self.A_initial = self.Iteration_Step(k)
                self.saveData(k)
                DA_value_all.append(DA_value)
                self.K_initial = self.K_initial_copy
                self.R_initial = self.R_initial_copy
                self.A_initial = self.A_initial_copy
            elif self.class_num == 5:
                DA_value, self.K_initial, self.R_initial, self.A_initial, self.U_initial, self.S_initial = self.Iteration_Step(
                    k)
                self.saveData(k)
                DA_value_all.append(DA_value)
                self.K_initial = self.K_initial_copy
                self.R_initial = self.R_initial_copy
                self.A_initial = self.A_initial_copy
                self.U_initial = self.U_initial_copy
                self.S_initial = self.S_initial_copy
            elif self.class_num == 7:
                DA_value, self.K_initial, self.R_initial, self.A_initial, self.S_initial, self.U_initial, self.V_initial, self.Gamma_initial = self.Iteration_Step(k)
                self.saveData(k)
                DA_value_all.append(DA_value)
                self.K_initial = self.K_initial_copy
                self.R_initial = self.R_initial_copy
                self.A_initial = self.A_initial_copy
                self.U_initial = self.U_initial_copy
                self.S_initial = self.S_initial_copy
                self.V_initial = self.V_initial_copy
                self.Gamma_initial = self.Gamma_initial_copy
            self.rho_initial = self.rho_initial_copy
            self.tau = self.tau_copy
        DA_value_all = np.array(DA_value_all).reshape(model_epoch, self.maxIter)
        return DA_value_all

    def validation_process(self):
        DA_val = []
        val_seq = self.data_Pre(mode="validation")
        model_epoch = self.data.shape[0]
        # model_epoch = 2
        for k in range(model_epoch):
            if self.class_num == 3:
                K_trained, R_trained, A_trained = self.loadData(k)
                self.K_initial, self.R_initial, self.A_initial = K_trained, R_trained, A_trained
            elif self.class_num == 5:
                K_trained, R_trained, A_trained, U_trained, S_trained = self.loadData(k)
                self.K_initial, self.R_initial, self.A_initial, self.U_initial, self.S_initial = \
                    K_trained, R_trained, A_trained, U_trained, S_trained
            elif self.class_num == 7:
                K_trained, R_trained, A_trained, S_trained, U_trained, V_trained, Gamma_trained = self.loadData(k)
                self.K_initial, self.R_initial, self.A_initial, self.U_initial, self.S_initial, self.V_initial, self.Gamma_initial= \
                    K_trained, R_trained, A_trained, U_trained, S_trained, V_trained, Gamma_trained
            DA_val_model = []
            for j in range(self.batch_num_val):
                DA_val_model.append(self.compute_like_hood(val_seq[j, :], mode="value"))
            DA_val.append(self.DA(np.mean(DA_val_model)))

        return np.array(DA_val)

    def test_process(self, k):
        DA_test = []
        test_seq = self.data_Pre(mode="test")
        test_epoch = test_seq.shape[0]
        if self.class_num == 3:
            K_best, R_best, A_best = self.loadData(k)
            self.K_initial, self.R_initial, self.A_initial = K_best, R_best, A_best
        elif self.class_num == 5:
            K_best, R_best, A_best, U_best, S_best = self.loadData(k)
            self.K_initial, self.R_initial, self.A_initial, self.U_initial, self.S_initial = \
                K_best, R_best, A_best, U_best, S_best
        elif self.class_num == 7:
            K_best, R_best, A_best, S_best, U_best, V_best, Gamma_best = self.loadData(k)
            self.K_initial, self.R_initial, self.A_initial, self.U_initial, self.S_initial, self.V_initial, self.Gamma_initial = \
                K_best, R_best, A_best, U_best, S_best, V_best, Gamma_best
        for j in range(test_epoch):
            DA_test.append(self.compute_like_hood(test_seq[j, :], mode="value"))
        print("the best DA for test data is %s" % (self.DA(np.mean(DA_test))))
        return self.DA(np.mean(DA_test))

    def Kappa_to_Kraus(self, Kappa):
        if self.class_num == 3:
            K = Kappa[0:self.dim_K * self.output_dims, :]
            R = Kappa[self.dim_K * self.output_dims:2 * self.dim_K * self.output_dims, :]
            A = Kappa[self.dim_K * self.output_dims * 2:self.dim_K * self.output_dims * 3, :]
            return K, R, A
        elif self.class_num == 5:
            K = Kappa[0:self.dim_K * self.output_dims, :]
            R = Kappa[self.dim_K * self.output_dims: 2 * self.dim_K * self.output_dims, :]
            A = Kappa[self.dim_K * self.output_dims * 2:self.dim_K * self.output_dims * 3, :]
            U = Kappa[self.dim_K * self.output_dims * 3:self.dim_K * self.output_dims * 4, :]
            S = Kappa[self.dim_K * self.output_dims * 4:self.dim_K * self.output_dims * 5, :]
            return K, R, A, U, S
        elif self.class_num == 7:
            K = Kappa[0:self.dim_K * self.output_dims, :]
            R = Kappa[self.dim_K * self.output_dims: 2 * self.dim_K * self.output_dims, :]
            A = Kappa[self.dim_K * self.output_dims * 2:self.dim_K * self.output_dims * 3, :]
            S = Kappa[self.dim_K * self.output_dims * 3:self.dim_K * self.output_dims * 4, :]
            U = Kappa[self.dim_K * self.output_dims * 4:self.dim_K * self.output_dims * 5, :]
            V = Kappa[self.dim_K * self.output_dims * 5:self.dim_K * self.output_dims * 6, :]
            Gamma = Kappa[self.dim_K * self.output_dims * 6:self.dim_K * self.output_dims * 7, :]
            return K, R, A, S, U, V, Gamma


    def __Inner_Product(self, M):
        return np.dot(self.dagger(M), M)
#保存计算得到的Kraus算符的矩阵解
    def saveData(self, k):
        if 'win' in sys.platform:
            if self.class_num == 3 and self.IP == 3:
                re = os.path.exists("Result\\Kraus_Training_Result33")
                if re:
                    np.savetxt("Result\\Kraus_Training_Result33\\K%d_real" % k, np.real(self.K_initial))
                    np.savetxt("Result\\Kraus_Training_Result33\\K%d_imag" % k, np.imag(self.K_initial))
                    np.savetxt("Result\\Kraus_Training_Result33\\R%d_real" % k, np.real(self.R_initial))
                    np.savetxt("Result\\Kraus_Training_Result33\\R%d_imag" % k, np.imag(self.R_initial))
                    np.savetxt("Result\\Kraus_Training_Result33\\A%d_real" % k, np.real(self.A_initial))
                    np.savetxt("Result\\Kraus_Training_Result33\\A%d_imag" % k, np.imag(self.A_initial))
                else:
                    os.mkdir("Result\\Kraus_Training_Result33")
                    np.savetxt("Result\\Kraus_Training_Result33\\K%d_real" % k, np.real(self.K_initial))
                    np.savetxt("Result\\Kraus_Training_Result33\\K%d_imag" % k, np.imag(self.K_initial))
                    np.savetxt("Result\\Kraus_Training_Result33\\R%d_real" % k, np.real(self.R_initial))
                    np.savetxt("Result\\Kraus_Training_Result33\\R%d_imag" % k, np.imag(self.R_initial))
                    np.savetxt("Result\\Kraus_Training_Result33\\A%d_real" % k, np.real(self.A_initial))
                    np.savetxt("Result\\Kraus_Training_Result33\\A%d_imag" % k, np.imag(self.A_initial))

            elif self.class_num == 3 and self.IP == 4:
                re = os.path.exists("Result\\Kraus_Training_Result34")
                if re:
                    np.savetxt("Result\\Kraus_Training_Result34\\K%d_real" % k, np.real(self.K_initial))
                    np.savetxt("Result\\Kraus_Training_Result34\\K%d_imag" % k, np.imag(self.K_initial))
                    np.savetxt("Result\\Kraus_Training_Result34\\R%d_real" % k, np.real(self.R_initial))
                    np.savetxt("Result\\Kraus_Training_Result34\\R%d_imag" % k, np.imag(self.R_initial))
                    np.savetxt("Result\\Kraus_Training_Result34\\A%d_real" % k, np.real(self.A_initial))
                    np.savetxt("Result\\Kraus_Training_Result34\\A%d_imag" % k, np.imag(self.A_initial))
                else:
                    os.mkdir("Result\\Kraus_Training_Result34")
                    np.savetxt("Result\\Kraus_Training_Result34\\K%d_real" % k, np.real(self.K_initial))
                    np.savetxt("Result\\Kraus_Training_Result34\\K%d_imag" % k, np.imag(self.K_initial))
                    np.savetxt("Result\\Kraus_Training_Result34\\R%d_real" % k, np.real(self.R_initial))
                    np.savetxt("Result\\Kraus_Training_Result34\\R%d_imag" % k, np.imag(self.R_initial))
                    np.savetxt("Result\\Kraus_Training_Result34\\A%d_real" % k, np.real(self.A_initial))
                    np.savetxt("Result\\Kraus_Training_Result34\\A%d_imag" % k, np.imag(self.A_initial))

            elif self.class_num == 5 and self.IP == 4:
                re = os.path.exists("Result\\Kraus_Training_Result54")
                if re:
                    np.savetxt("Result\\Kraus_Training_Result54\\K%d_real" % k, np.real(self.K_initial))
                    np.savetxt("Result\\Kraus_Training_Result54\\K%d_imag" % k, np.imag(self.K_initial))
                    np.savetxt("Result\\Kraus_Training_Result54\\R%d_real" % k, np.real(self.R_initial))
                    np.savetxt("Result\\Kraus_Training_Result54\\R%d_imag" % k, np.imag(self.R_initial))
                    np.savetxt("Result\\Kraus_Training_Result54\\A%d_real" % k, np.real(self.A_initial))
                    np.savetxt("Result\\Kraus_Training_Result54\\A%d_imag" % k, np.imag(self.A_initial))
                    np.savetxt("Result\\Kraus_Training_Result54\\U%d_real" % k, np.real(self.U_initial))
                    np.savetxt("Result\\Kraus_Training_Result54\\U%d_imag" % k, np.imag(self.U_initial))
                    np.savetxt("Result\\Kraus_Training_Result54\\S%d_real" % k, np.real(self.S_initial))
                    np.savetxt("Result\\Kraus_Training_Result54\\S%d_imag" % k, np.imag(self.S_initial))
                else:
                    os.mkdir("Result\\Kraus_Training_Result54")
                    np.savetxt("Result\\Kraus_Training_Result54\\K%d_real" % k, np.real(self.K_initial))
                    np.savetxt("Result\\Kraus_Training_Result54\\K%d_imag" % k, np.imag(self.K_initial))
                    np.savetxt("Result\\Kraus_Training_Result54\\R%d_real" % k, np.real(self.R_initial))
                    np.savetxt("Result\\Kraus_Training_Result54\\R%d_imag" % k, np.imag(self.R_initial))
                    np.savetxt("Result\\Kraus_Training_Result54\\A%d_real" % k, np.real(self.A_initial))
                    np.savetxt("Result\\Kraus_Training_Result54\\A%d_imag" % k, np.imag(self.A_initial))
                    np.savetxt("Result\\Kraus_Training_Result54\\U%d_real" % k, np.real(self.U_initial))
                    np.savetxt("Result\\Kraus_Training_Result54\\U%d_imag" % k, np.imag(self.U_initial))
                    np.savetxt("Result\\Kraus_Training_Result54\\S%d_real" % k, np.real(self.S_initial))
                    np.savetxt("Result\\Kraus_Training_Result54\\S%d_imag" % k, np.imag(self.S_initial))
            elif self.class_num == 5 and self.IP == 3:
                re = os.path.exists("Result\\Kraus_Training_Result53")
                if re:
                    np.savetxt("Result\\Kraus_Training_Result53\\K%d_real" % k, np.real(self.K_initial))
                    np.savetxt("Result\\Kraus_Training_Result53\\K%d_imag" % k, np.imag(self.K_initial))
                    np.savetxt("Result\\Kraus_Training_Result53\\R%d_real" % k, np.real(self.R_initial))
                    np.savetxt("Result\\Kraus_Training_Result53\\R%d_imag" % k, np.imag(self.R_initial))
                    np.savetxt("Result\\Kraus_Training_Result53\\A%d_real" % k, np.real(self.A_initial))
                    np.savetxt("Result\\Kraus_Training_Result53\\A%d_imag" % k, np.imag(self.A_initial))
                    np.savetxt("Result\\Kraus_Training_Result53\\U%d_real" % k, np.real(self.U_initial))
                    np.savetxt("Result\\Kraus_Training_Result53\\U%d_imag" % k, np.imag(self.U_initial))
                    np.savetxt("Result\\Kraus_Training_Result53\\S%d_real" % k, np.real(self.S_initial))
                    np.savetxt("Result\\Kraus_Training_Result53\\S%d_imag" % k, np.imag(self.S_initial))
                else:
                    os.mkdir("Result\\Kraus_Training_Result53")
                    np.savetxt("Result\\Kraus_Training_Result53\\K%d_real" % k, np.real(self.K_initial))
                    np.savetxt("Result\\Kraus_Training_Result53\\K%d_imag" % k, np.imag(self.K_initial))
                    np.savetxt("Result\\Kraus_Training_Result53\\R%d_real" % k, np.real(self.R_initial))
                    np.savetxt("Result\\Kraus_Training_Result53\\R%d_imag" % k, np.imag(self.R_initial))
                    np.savetxt("Result\\Kraus_Training_Result53\\A%d_real" % k, np.real(self.A_initial))
                    np.savetxt("Result\\Kraus_Training_Result53\\A%d_imag" % k, np.imag(self.A_initial))
                    np.savetxt("Result\\Kraus_Training_Result53\\U%d_real" % k, np.real(self.U_initial))
                    np.savetxt("Result\\Kraus_Training_Result53\\U%d_imag" % k, np.imag(self.U_initial))
                    np.savetxt("Result\\Kraus_Training_Result53\\S%d_real" % k, np.real(self.S_initial))
                    np.savetxt("Result\\Kraus_Training_Result53\\S%d_imag" % k, np.imag(self.S_initial))
            elif self.class_num == 7 and self.IP == 4:
                re = os.path.exists("Result\\Kraus_Training_Result74")
                if re:
                    np.savetxt("Result\\Kraus_Training_Result74\\K%d_real" % k, np.real(self.K_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\K%d_imag" % k, np.imag(self.K_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\R%d_real" % k, np.real(self.R_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\R%d_imag" % k, np.imag(self.R_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\A%d_real" % k, np.real(self.A_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\A%d_imag" % k, np.imag(self.A_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\U%d_real" % k, np.real(self.U_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\U%d_imag" % k, np.imag(self.U_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\S%d_real" % k, np.real(self.S_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\S%d_imag" % k, np.imag(self.S_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\V%d_real" % k, np.real(self.V_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\V%d_imag" % k, np.imag(self.V_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\Gamma%d_real" % k, np.real(self.Gamma_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\Gamma%d_imag" % k, np.imag(self.Gamma_initial))
                else:
                    os.mkdir("Result\\Kraus_Training_Result74")
                    np.savetxt("Result\\Kraus_Training_Result74\\K%d_real" % k, np.real(self.K_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\K%d_imag" % k, np.imag(self.K_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\R%d_real" % k, np.real(self.R_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\R%d_imag" % k, np.imag(self.R_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\A%d_real" % k, np.real(self.A_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\A%d_imag" % k, np.imag(self.A_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\U%d_real" % k, np.real(self.U_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\U%d_imag" % k, np.imag(self.U_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\S%d_real" % k, np.real(self.S_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\S%d_imag" % k, np.imag(self.S_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\V%d_real" % k, np.real(self.V_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\V%d_imag" % k, np.imag(self.V_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\Gamma%d_real" % k, np.real(self.Gamma_initial))
                    np.savetxt("Result\\Kraus_Training_Result74\\Gamma%d_imag" % k, np.imag(self.Gamma_initial))

#根据操作系统不同保存数据的方式也不同
        elif 'linux' in sys.platform:
            if self.class_num == 3 and self.IP == 3:
                re = os.path.exists("Result/Kraus_Training_Result33")
                if re:
                    np.savetxt("Result/Kraus_Training_Result33/K%d_real" % k, np.real(self.K_initial))
                    np.savetxt("Result/Kraus_Training_Result33/K%d_imag" % k, np.imag(self.K_initial))
                    np.savetxt("Result/Kraus_Training_Result33/R%d_real" % k, np.real(self.R_initial))
                    np.savetxt("Result/Kraus_Training_Result33/R%d_imag" % k, np.imag(self.R_initial))
                    np.savetxt("Result/Kraus_Training_Result33/A%d_real" % k, np.real(self.A_initial))
                    np.savetxt("Result/Kraus_Training_Result33/A%d_imag" % k, np.imag(self.A_initial))

                else:
                    os.mkdir("Result/Kraus_Training_Result33")
                    np.savetxt("Result/Kraus_Training_Result33/K%d_real" % k, np.real(self.K_initial))
                    np.savetxt("Result/Kraus_Training_Result33/K%d_imag" % k, np.imag(self.K_initial))
                    np.savetxt("Result/Kraus_Training_Result33/R%d_real" % k, np.real(self.R_initial))
                    np.savetxt("Result/Kraus_Training_Result33/R%d_imag" % k, np.imag(self.R_initial))
                    np.savetxt("Result/Kraus_Training_Result33/A%d_real" % k, np.real(self.A_initial))
                    np.savetxt("Result/Kraus_Training_Result33/A%d_imag" % k, np.imag(self.A_initial))


            elif self.class_num == 3 and self.IP == 4:
                re = os.path.exists("Result/Kraus_Training_Result34")
                if re:
                    np.savetxt("Result/Kraus_Training_Result34/K%d_real" % k, np.real(self.K_initial))
                    np.savetxt("Result/Kraus_Training_Result34/K%d_imag" % k, np.imag(self.K_initial))
                    np.savetxt("Result/Kraus_Training_Result34/R%d_real" % k, np.real(self.R_initial))
                    np.savetxt("Result/Kraus_Training_Result34/R%d_imag" % k, np.imag(self.R_initial))
                    np.savetxt("Result/Kraus_Training_Result34/A%d_real" % k, np.real(self.A_initial))
                    np.savetxt("Result/Kraus_Training_Result34/A%d_imag" % k, np.imag(self.A_initial))
                else:
                    os.mkdir("Result/Kraus_Training_Result34")
                    np.savetxt("Result/Kraus_Training_Result34/K%d_real" % k, np.real(self.K_initial))
                    np.savetxt("Result/Kraus_Training_Result34/K%d_imag" % k, np.imag(self.K_initial))
                    np.savetxt("Result/Kraus_Training_Result34/R%d_real" % k, np.real(self.R_initial))
                    np.savetxt("Result/Kraus_Training_Result34/R%d_imag" % k, np.imag(self.R_initial))
                    np.savetxt("Result/Kraus_Training_Result34/A%d_real" % k, np.real(self.A_initial))
                    np.savetxt("Result/Kraus_Training_Result34/A%d_imag" % k, np.imag(self.A_initial))

            elif self.class_num == 5 and self.IP == 4:
                re = os.path.exists("Result/Kraus_Training_Result54")
                if re:
                    np.savetxt("Result/Kraus_Training_Result54/K%d_real" % k, np.real(self.K_initial))
                    np.savetxt("Result/Kraus_Training_Result54/K%d_imag" % k, np.imag(self.K_initial))
                    np.savetxt("Result/Kraus_Training_Result54/R%d_real" % k, np.real(self.R_initial))
                    np.savetxt("Result/Kraus_Training_Result54/R%d_imag" % k, np.imag(self.R_initial))
                    np.savetxt("Result/Kraus_Training_Result54/A%d_real" % k, np.real(self.A_initial))
                    np.savetxt("Result/Kraus_Training_Result54/A%d_imag" % k, np.imag(self.A_initial))
                    np.savetxt("Result/Kraus_Training_Result54/U%d_real" % k, np.real(self.U_initial))
                    np.savetxt("Result/Kraus_Training_Result54/U%d_imag" % k, np.imag(self.U_initial))
                    np.savetxt("Result/Kraus_Training_Result54/S%d_real" % k, np.real(self.S_initial))
                    np.savetxt("Result/Kraus_Training_Result54/S%d_imag" % k, np.imag(self.S_initial))
                else:
                    os.mkdir("Result/Kraus_Training_Result54")
                    np.savetxt("Result/Kraus_Training_Result54/K%d_real" % k, np.real(self.K_initial))
                    np.savetxt("Result/Kraus_Training_Result54/K%d_imag" % k, np.imag(self.K_initial))
                    np.savetxt("Result/Kraus_Training_Result54/R%d_real" % k, np.real(self.R_initial))
                    np.savetxt("Result/Kraus_Training_Result54/R%d_imag" % k, np.imag(self.R_initial))
                    np.savetxt("Result/Kraus_Training_Result54/A%d_real" % k, np.real(self.A_initial))
                    np.savetxt("Result/Kraus_Training_Result54/A%d_imag" % k, np.imag(self.A_initial))
                    np.savetxt("Result/Kraus_Training_Result54/U%d_real" % k, np.real(self.U_initial))
                    np.savetxt("Result/Kraus_Training_Result54/U%d_imag" % k, np.imag(self.U_initial))
                    np.savetxt("Result/Kraus_Training_Result54/S%d_real" % k, np.real(self.S_initial))
                    np.savetxt("Result/Kraus_Training_Result54/S%d_imag" % k, np.imag(self.S_initial))
            elif self.class_num == 5 and self.IP == 3:
                re = os.path.exists("Result/Kraus_Training_Result53")
                if re:
                    np.savetxt("Result/Kraus_Training_Result53/K%d_real" % k, np.real(self.K_initial))
                    np.savetxt("Result/Kraus_Training_Result53/K%d_imag" % k, np.imag(self.K_initial))
                    np.savetxt("Result/Kraus_Training_Result53/R%d_real" % k, np.real(self.R_initial))
                    np.savetxt("Result/Kraus_Training_Result53/R%d_imag" % k, np.imag(self.R_initial))
                    np.savetxt("Result/Kraus_Training_Result53/A%d_real" % k, np.real(self.A_initial))
                    np.savetxt("Result/Kraus_Training_Result53/A%d_imag" % k, np.imag(self.A_initial))
                    np.savetxt("Result/Kraus_Training_Result53/U%d_real" % k, np.real(self.U_initial))
                    np.savetxt("Result/Kraus_Training_Result53/U%d_imag" % k, np.imag(self.U_initial))
                    np.savetxt("Result/Kraus_Training_Result53/S%d_real" % k, np.real(self.S_initial))
                    np.savetxt("Result/Kraus_Training_Result53/S%d_imag" % k, np.imag(self.S_initial))
                else:
                    os.mkdir("Result/Kraus_Training_Result53")
                    np.savetxt("Result/Kraus_Training_Result53/K%d_real" % k, np.real(self.K_initial))
                    np.savetxt("Result/Kraus_Training_Result53/K%d_imag" % k, np.imag(self.K_initial))
                    np.savetxt("Result/Kraus_Training_Result53/R%d_real" % k, np.real(self.R_initial))
                    np.savetxt("Result/Kraus_Training_Result53/R%d_imag" % k, np.imag(self.R_initial))
                    np.savetxt("Result/Kraus_Training_Result53/A%d_real" % k, np.real(self.A_initial))
                    np.savetxt("Result/Kraus_Training_Result53/A%d_imag" % k, np.imag(self.A_initial))
                    np.savetxt("Result/Kraus_Training_Result53/U%d_real" % k, np.real(self.U_initial))
                    np.savetxt("Result/Kraus_Training_Result53/U%d_imag" % k, np.imag(self.U_initial))
                    np.savetxt("Result/Kraus_Training_Result53/S%d_real" % k, np.real(self.S_initial))
                    np.savetxt("Result/Kraus_Training_Result53/S%d_imag" % k, np.imag(self.S_initial))
            elif self.class_num == 7 and self.IP == 4:
                re = os.path.exists("Result/Kraus_Training_Result74")
                if re:
                    np.savetxt("Result/Kraus_Training_Result74/K%d_real" % k, np.real(self.K_initial))
                    np.savetxt("Result/Kraus_Training_Result74/K%d_imag" % k, np.imag(self.K_initial))
                    np.savetxt("Result/Kraus_Training_Result74/R%d_real" % k, np.real(self.R_initial))
                    np.savetxt("Result/Kraus_Training_Result74/R%d_imag" % k, np.imag(self.R_initial))
                    np.savetxt("Result/Kraus_Training_Result74/A%d_real" % k, np.real(self.A_initial))
                    np.savetxt("Result/Kraus_Training_Result74/A%d_imag" % k, np.imag(self.A_initial))
                    np.savetxt("Result/Kraus_Training_Result74/S%d_real" % k, np.real(self.S_initial))
                    np.savetxt("Result/Kraus_Training_Result74/S%d_imag" % k, np.imag(self.S_initial))
                    np.savetxt("Result/Kraus_Training_Result74/U%d_real" % k, np.real(self.U_initial))
                    np.savetxt("Result/Kraus_Training_Result74/U%d_imag" % k, np.imag(self.U_initial))
                    np.savetxt("Result/Kraus_Training_Result74/V%d_real" % k, np.real(self.V_initial))
                    np.savetxt("Result/Kraus_Training_Result74/V%d_imag" % k, np.imag(self.V_initial))
                    np.savetxt("Result/Kraus_Training_Result74/Gamma%d_real" % k, np.real(self.Gamma_initial))
                    np.savetxt("Result/Kraus_Training_Result74/Gamma%d_imag" % k, np.imag(self.Gamma_initial))
                else:
                    os.mkdir("Result/Kraus_Training_Result74")
                    np.savetxt("Result/Kraus_Training_Result74/K%d_real" % k, np.real(self.K_initial))
                    np.savetxt("Result/Kraus_Training_Result74/K%d_imag" % k, np.imag(self.K_initial))
                    np.savetxt("Result/Kraus_Training_Result74/R%d_real" % k, np.real(self.R_initial))
                    np.savetxt("Result/Kraus_Training_Result74/R%d_imag" % k, np.imag(self.R_initial))
                    np.savetxt("Result/Kraus_Training_Result74/A%d_real" % k, np.real(self.A_initial))
                    np.savetxt("Result/Kraus_Training_Result74/A%d_imag" % k, np.imag(self.A_initial))
                    np.savetxt("Result/Kraus_Training_Result74/S%d_real" % k, np.real(self.S_initial))
                    np.savetxt("Result/Kraus_Training_Result74/S%d_imag" % k, np.imag(self.S_initial))
                    np.savetxt("Result/Kraus_Training_Result74/U%d_real" % k, np.real(self.U_initial))
                    np.savetxt("Result/Kraus_Training_Result74/U%d_imag" % k, np.imag(self.U_initial))
                    np.savetxt("Result/Kraus_Training_Result74/V%d_real" % k, np.real(self.V_initial))
                    np.savetxt("Result/Kraus_Training_Result74/V%d_imag" % k, np.imag(self.V_initial))
                    np.savetxt("Result/Kraus_Training_Result74/Gamma%d_real" % k, np.real(self.Gamma_initial))
                    np.savetxt("Result/Kraus_Training_Result74/Gamma%d_imag" % k, np.imag(self.Gamma_initial))


    def loadData(self, k):
        if 'win' in sys.platform:
            if self.class_num == 3 and self.IP == 3:
                K_trained = np.loadtxt("Result\\Kraus_Training_Result33\\K%d_real" % k) + 1j * np.loadtxt(
                    "Result\\Kraus_Training_Result33\\K%d_imag" % k)

                R_trained = np.loadtxt("Result\\Kraus_Training_Result33\\R%d_real" % k) + 1j * np.loadtxt(
                    "Result\\Kraus_Training_Result33\\R%d_imag" % k)

                A_trained = np.loadtxt("Result\\Kraus_Training_Result33\\A%d_real" % k) + 1j * np.loadtxt(
                    "Result\\Kraus_Training_Result33\\A%d_imag" % k)
                return K_trained, R_trained, A_trained
            elif self.class_num == 3 and self.IP == 4:
                K_trained = np.loadtxt("Result\\Kraus_Training_Result34\\K%d_real" % k) + 1j * np.loadtxt(
                    "Result\\Kraus_Training_Result34\\K%d_imag" % k)

                R_trained = np.loadtxt("Result\\Kraus_Training_Result34\\R%d_real" % k) + 1j * np.loadtxt(
                    "Result\\Kraus_Training_Result34\\R%d_imag" % k)

                A_trained = np.loadtxt("Result\\Kraus_Training_Result34\\A%d_real" % k) + 1j * np.loadtxt(
                    "Result\\Kraus_Training_Result34\\A%d_imag" % k)
                return K_trained, R_trained, A_trained
            elif self.class_num == 5 and self.IP == 4:
                K_trained = np.loadtxt("Result\\Kraus_Training_Result54\\K%d_real" % k) + 1j * np.loadtxt(
                    "Result\\Kraus_Training_Result54\\K%d_imag" % k)

                R_trained = np.loadtxt("Result\\Kraus_Training_Result54\\R%d_real" % k) + 1j * np.loadtxt(
                    "Result\\Kraus_Training_Result54\\R%d_imag" % k)

                A_trained = np.loadtxt("Result\\Kraus_Training_Result54\\A%d_real" % k) + 1j * np.loadtxt(
                    "Result\\Kraus_Training_Result54\\A%d_imag" % k)

                U_trained = np.loadtxt("Result\\Kraus_Training_Result54\\U%d_real" % k) + 1j * np.loadtxt(
                    "Result\\Kraus_Training_Result54\\U%d_imag" % k)

                S_trained = np.loadtxt("Result\\Kraus_Training_Result54\\S%d_real" % k) + 1j * np.loadtxt(
                    "Result\\Kraus_Training_Result54\\S%d_imag" % k)
                return K_trained, R_trained, A_trained, U_trained, S_trained
            elif self.class_num == 5 and self.IP == 3:
                K_trained = np.loadtxt("Result\\Kraus_Training_Result53\\K%d_real" % k) + 1j * np.loadtxt(
                    "Result\\Kraus_Training_Result53\\K%d_imag" % k)

                R_trained = np.loadtxt("Result\\Kraus_Training_Result53\\R%d_real" % k) + 1j * np.loadtxt(
                    "Result\\Kraus_Training_Result53\\R%d_imag" % k)

                A_trained = np.loadtxt("Result\\Kraus_Training_Result53\\A%d_real" % k) + 1j * np.loadtxt(
                    "Result\\Kraus_Training_Result53\\A%d_imag" % k)

                U_trained = np.loadtxt("Result\\Kraus_Training_Result53\\U%d_real" % k) + 1j * np.loadtxt(
                    "Result\\Kraus_Training_Result53\\U%d_imag" % k)

                S_trained = np.loadtxt("Result\\Kraus_Training_Result53\\S%d_real" % k) + 1j * np.loadtxt(
                    "Result\\Kraus_Training_Result53\\S%d_imag" % k)
                return K_trained, R_trained, A_trained, U_trained, S_trained
            elif self.class_num == 7 and self.IP == 4:
                K_trained = np.loadtxt("Result\\Kraus_Training_Result74\\K%d_real" % k) + 1j * np.loadtxt(
                    "Result\\Kraus_Training_Result74\\K%d_imag" % k)

                R_trained = np.loadtxt("Result\\Kraus_Training_Result74\\R%d_real" % k) + 1j * np.loadtxt(
                    "Result\\Kraus_Training_Result74\\R%d_imag" % k)

                A_trained = np.loadtxt("Result\\Kraus_Training_Result74\\A%d_real" % k) + 1j * np.loadtxt(
                    "Result\\Kraus_Training_Result74\\A%d_imag" % k)

                U_trained = np.loadtxt("Result\\Kraus_Training_Result74\\U%d_real" % k) + 1j * np.loadtxt(
                    "Result\\Kraus_Training_Result74\\U%d_imag" % k)

                S_trained = np.loadtxt("Result\\Kraus_Training_Result74\\S%d_real" % k) + 1j * np.loadtxt(
                    "Result\\Kraus_Training_Result74\\S%d_imag" % k)
                V_trained = np.loadtxt("Result\\Kraus_Training_Result74\\V%d_real" % k) + 1j * np.loadtxt(
                    "Result\\Kraus_Training_Result74\\V%d_imag" % k)
                Gamma_trained = np.loadtxt("Result\\Kraus_Training_Result74\\Gamma%d_real" % k) + 1j * np.loadtxt(
                    "Result\\Kraus_Training_Result74\\Gamma%d_imag" % k)
                return K_trained, R_trained, A_trained, S_trained, U_trained, V_trained, Gamma_trained


        elif 'linux' in sys.platform:
            if self.class_num == 3 and self.IP == 3:
                K_trained = np.loadtxt("Result/Kraus_Training_Result33/K%d_real" % k) + 1j * np.loadtxt(
                    "Result/Kraus_Training_Result33/K%d_imag" % k)

                R_trained = np.loadtxt("Result/Kraus_Training_Result33/R%d_real" % k) + 1j * np.loadtxt(
                    "Result/Kraus_Training_Result33/R%d_imag" % k)

                A_trained = np.loadtxt("Result/Kraus_Training_Result33/A%d_real" % k) + 1j * np.loadtxt(
                    "Result/Kraus_Training_Result33/A%d_imag" % k)
                return K_trained, R_trained, A_trained
            elif self.class_num == 3 and self.IP == 4:
                K_trained = np.loadtxt("Result/Kraus_Training_Result34/K%d_real" % k) + 1j * np.loadtxt(
                    "Result/Kraus_Training_Result34/K%d_imag" % k)

                R_trained = np.loadtxt("Result/Kraus_Training_Result34/R%d_real" % k) + 1j * np.loadtxt(
                    "Result/Kraus_Training_Result34/R%d_imag" % k)

                A_trained = np.loadtxt("Result/Kraus_Training_Result34/A%d_real" % k) + 1j * np.loadtxt(
                    "Result/Kraus_Training_Result34/A%d_imag" % k)
                return K_trained, R_trained, A_trained
            elif self.class_num == 5 and self.IP == 4:
                K_trained = np.loadtxt("Result/Kraus_Training_Result54/K%d_real" % k) + 1j * np.loadtxt(
                    "Result/Kraus_Training_Result54/K%d_imag" % k)

                R_trained = np.loadtxt("Result/Kraus_Training_Result54/R%d_real" % k) + 1j * np.loadtxt(
                    "Result/Kraus_Training_Result54/R%d_imag" % k)

                A_trained = np.loadtxt("Result/Kraus_Training_Result54/A%d_real" % k) + 1j * np.loadtxt(
                    "Result/Kraus_Training_Result54/A%d_imag" % k)

                U_trained = np.loadtxt("Result/Kraus_Training_Result54/U%d_real" % k) + 1j * np.loadtxt(
                    "Result/Kraus_Training_Result54/U%d_imag" % k)

                S_trained = np.loadtxt("Result/Kraus_Training_Result54/S%d_real" % k) + 1j * np.loadtxt(
                    "Result/Kraus_Training_Result54/S%d_imag" % k)
                return K_trained, R_trained, A_trained, U_trained, S_trained
            elif self.class_num == 5 and self.IP == 3:
                K_trained = np.loadtxt("Result/Kraus_Training_Result53/K%d_real" % k) + 1j * np.loadtxt(
                    "Result/Kraus_Training_Result53/K%d_imag" % k)

                R_trained = np.loadtxt("Result/Kraus_Training_Result53/R%d_real" % k) + 1j * np.loadtxt(
                    "Result/Kraus_Training_Result53/R%d_imag" % k)

                A_trained = np.loadtxt("Result/Kraus_Training_Result53/A%d_real" % k) + 1j * np.loadtxt(
                    "Result/Kraus_Training_Result53/A%d_imag" % k)

                U_trained = np.loadtxt("Result/Kraus_Training_Result53/U%d_real" % k) + 1j * np.loadtxt(
                    "Result/Kraus_Training_Result53/U%d_imag" % k)

                S_trained = np.loadtxt("Result/Kraus_Training_Result53/S%d_real" % k) + 1j * np.loadtxt(
                    "Result/Kraus_Training_Result53/S%d_imag" % k)
                return K_trained, R_trained, A_trained, U_trained, S_trained
            elif self.class_num == 7 and self.IP == 4:
                K_trained = np.loadtxt("Result/Kraus_Training_Result74/K%d_real" % k) + 1j * np.loadtxt(
                    "Result/Kraus_Training_Result74/K%d_imag" % k)

                R_trained = np.loadtxt("Result/Kraus_Training_Result74/R%d_real" % k) + 1j * np.loadtxt(
                    "Result/Kraus_Training_Result74/R%d_imag" % k)

                A_trained = np.loadtxt("Result/Kraus_Training_Result74/A%d_real" % k) + 1j * np.loadtxt(
                    "Result/Kraus_Training_Result74/A%d_imag" % k)

                U_trained = np.loadtxt("Result/Kraus_Training_Result74/U%d_real" % k) + 1j * np.loadtxt(
                    "Result/Kraus_Training_Result74/U%d_imag" % k)

                S_trained = np.loadtxt("Result/Kraus_Training_Result74/S%d_real" % k) + 1j * np.loadtxt(
                    "Result/Kraus_Training_Result74/S%d_imag" % k)
                V_trained = np.loadtxt("Result/Kraus_Training_Result74/V%d_real" % k) + 1j * np.loadtxt(
                    "Result/Kraus_Training_Result74/V%d_imag" % k)
                Gamma_trained = np.loadtxt("Result/Kraus_Training_Result74/Gamma%d_real" % k) + 1j * np.loadtxt(
                    "Result/Kraus_Training_Result74/Gamma%d_imag" % k)
                return K_trained, R_trained, A_trained, S_trained, U_trained, V_trained, Gamma_trained
#可视化不同条件密度矩阵之间的连接关系
    @staticmethod
    def Show_Network(str=False):
        fig = plt.figure(
            facecolor='white',
        )
        ax = fig.subplots()
        circle = Circle(xy=(1, 0), radius=0.25)
        ax.add_patch(p=circle)
        circle.set(fc='blue', ec='black', alpha=0.4, lw=2)

        x_shift = 3
        y_shift = 3
        circle1 = Circle(xy=(1, y_shift), radius=0.25)
        ax.add_patch(circle1)
        circle1.set(fc='blue', ec='black', alpha=0.4, lw=2)

        circle2 = Circle(xy=(1, y_shift * 2), radius=0.25)
        ax.add_patch(circle2)
        circle2.set(fc='blue', ec='black', alpha=0.4, lw=2)

        circle3 = Circle(xy=(1 + x_shift, y_shift * 2), radius=0.25)
        ax.add_patch(circle3)
        circle3.set(fc='blue', ec='black', alpha=0.4, lw=2)

        circle4 = Circle(xy=(1 + x_shift, y_shift), radius=0.25)
        ax.add_patch(circle4)
        circle4.set(fc='blue', ec='black', alpha=0.4, lw=2)

        circle5 = Circle(xy=(1 + x_shift, 0), radius=0.25)
        ax.add_patch(circle5)
        circle5.set(fc='blue', ec='black', alpha=0.4, lw=2)

        ax.axis([-1, 7, -1, 7])
        ax.set(aspect='equal', xticks=(), yticks=())

        plt.arrow(1.25, 0, x_shift - 0.8, 0, head_width=0.15, head_length=0.3, ec='red')
        plt.arrow(1.25, y_shift, x_shift - 0.8, 0, head_width=0.15, head_length=0.3, ec='red')
        plt.arrow(1.25, y_shift * 2, x_shift - 0.8, 0, head_width=0.15, head_length=0.3, ec='red')
        plt.arrow(1.25, y_shift, x_shift - 0.5 - 0.3 * 0.7, y_shift - 0.3, head_width=0.15, head_length=0.3, ec='black')
        plt.arrow(1.25, 0, x_shift - 0.5 - 0.3 * 0.7, y_shift - 0.3, head_width=0.15, head_length=0.3, ec='black')
        plt.arrow(1.25, y_shift, x_shift - 0.5 - 0.3 * 0.7, -y_shift + 0.3, head_width=0.15, head_length=0.3, ec='blue')
        plt.arrow(1.25, y_shift * 2, x_shift - 0.5 - 0.3 * 0.7, -y_shift + 0.3, head_width=0.15, head_length=0.3,
                  ec='blue')

        # boundary condition
        plt.arrow(1.25, y_shift * 2, x_shift - 0.5 - 0.15, -y_shift * 2 + 0.25, head_width=0.15,
                  head_length=0.3, ec='black', ls="--", lw=0.5, visible=str)
        plt.arrow(1.25, 0, x_shift - 0.5 - 0.15, y_shift * 2 - 0.25, head_width=0.15,
                  head_length=0.3, ec='blue', ls="--", lw=0.5, visible=str)

        plt.text(2.5, 0, r"K", color="red", fontsize=12)
        plt.text(2.5, y_shift, r"K", color="red", fontsize=12)
        plt.text(2.5, 2 * y_shift, r"K", color="red", fontsize=12)

        plt.text(3.5, y_shift * 0.7, r"R", color="black", fontsize=12)
        plt.text(3.5, y_shift + 2, r"R", color="black", fontsize=12)

        plt.text(1.5, y_shift * 0.7, r"A", color="blue", fontsize=12)
        plt.text(1.5, y_shift + 2, r"A", color="blue", fontsize=12)

        plt.text(0, 0, r'$\rho^{(0)}_{t}$', fontsize=12)
        plt.text(0, y_shift, r'$\rho^{(1)}_{t}$', fontsize=12)
        plt.text(0, y_shift * 2, r'$\rho^{(2)}_{t}$', fontsize=12)
        plt.text(4.5, 0, r'$\rho^{(0)}_{t+1}$', fontsize=12)
        plt.text(4.5, y_shift, r'$\rho^{(1)}_{t+1}$', fontsize=12)
        plt.text(4.5, y_shift * 2, r'$\rho^{(2)}_{t+1}$', fontsize=12)
        plt.title("The graphical model of split hidden quantum Markov model")
        plt.show()
