# https://github.com/arkmohba/ARK_study_Quclassi/blob/cb26cbf67fd499e4731a779877d7c3a4391484a5/src/quclassi_circuit.py
import copy
import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import qiskit


class QuClassiCircuit():
    """QuClassiにて、1つのクラスに対応する量子回路のクラス"""

    def __init__(self, input_size: int, name: Optional[str] = None) -> None:
        """量子レジスタと古典レジスタを生成し、それらから量子回路を生成する

        :param int input_size: 入力ベクトルの次元
        :param Optional[str] name: 量子回路の名前, defaults to None
        """
        # 入力次元を偶数化する
        if input_size % 2 != 0:
            input_size += 1
        self.modified_input_size = input_size  # 入力次元
        self.num_qubits = input_size + 1  # 全量子ビット数
        self.num_trained_qubits = input_size // 2  # 本回路の代表量子状態生成に使用する量子ビット数
        self.num_cbits = 1  # 古典ビット数

        # クラス変数を初期化する
        self.best_loss = None
        self.loss_history = []
        self.epochs = 0
        self.label = None

        # 量子回路を生成する
        self.control_quantum_register = qiskit.QuantumRegister(1, name="control_qubit")
        self.trained_quantum_register = qiskit.QuantumRegister(self.num_trained_qubits, name="trained_qubit")
        self.loaded_quantum_register = qiskit.QuantumRegister(self.num_trained_qubits, name="loaded_qubit")
        self.classical_register = qiskit.ClassicalRegister(self.num_cbits, name="classical_bit")
        self.quantum_circuit = qiskit.QuantumCircuit(self.control_quantum_register,
                                                     self.trained_quantum_register,
                                                     self.loaded_quantum_register,
                                                     self.classical_register,
                                                     name=name)

    def build_quantum_circuit(self, structure: List[str], thetas_list: List[List[float]], is_in_train: bool = False) -> None:
        """量子回路を構成する

        :param List[str] structure: 構造を定める文字列 (s,d,cのみからなる文字列)
        :param List[List[float]] thetas_list: structureで定めた構造における初期回転角
        :param bool is_in_train: 学習中かどうか, defaults to False
        :raises ValueError: structureにs,d,c以外の文字が含まれる場合
        :raises ValueError: structureとthis_listの長さが異なる場合
        """
        # 与えられた構造の妥当性を確認し、問題があればエラーを発生させる
        num_s = structure.count('s')
        num_d = structure.count('d')
        num_c = structure.count('c')
        if num_s + num_d + num_c != len(structure):
            msg = f"structure must have only 's', 'd', and 'c', but this structure is {structure}"
            raise ValueError(msg)
        if len(structure) != len(thetas_list):
            msg = f"length of structure must equal to length of thetas_list,\nbut len(structure) = {len(structure)} and len(thetas_list) = {len(thetas_list)}"
            raise ValueError(msg)

        # 制御用の量子ゲートを用意する
        self.quantum_circuit.h(0)
        self.quantum_circuit.barrier()

        # 代表量子状態生成のための量子ゲートを用意する
        for letter, thetas in zip(structure, thetas_list):
            if letter == 's':
                self.add_single_qubit_unitary_layer(thetas)
            elif letter == 'd':
                self.add_dual_qubit_unitary_layer(thetas)
            elif letter == 'c':
                self.add_controlled_qubit_unitary_layer(thetas)

            self.quantum_circuit.barrier()

        # 生成した量子回路の情報をクラス変数に保存する
        self.structure = structure
        self.thetas_list = thetas_list
        self.num_thetas = sum([len(np.array(thetas).reshape(-1)) for thetas in self.thetas_list])

        # データ読込のための量子ゲートを準備する
        self.add_load_structure()

        # SWAPテストのために必要な残りのゲートを準備する
        self.add_cswap()
        self.quantum_circuit.h(0)
        self.quantum_circuit.measure(0, 0)

        if not is_in_train:
            print("Successfully built.")

    def run(self, back_end: str, shots: int, on_ibmq: bool) -> float:
        """生成してある量子回路を実行し、量子状態フィデリティに準じるを求める
        ただし、厳密には量子状態フィデリティ|<x|y>|ではなく、(1 + |<x|y>|^2)/2を計算する
        任意のk < lに対して、(1 + k^2)/2 < (1 + l^2)/2であり、
        ここでの出力は大小関係を見るために使われるので、大きな問題はない

        :param str back_end: バックエンド
        :param int shots: 量子回路の実行回数(期待値を取る回数)
        :param bool on_ibmq: 実機を使うかどうか
        :raises ValueError: shots回の実効の中に0が一度も観測されなかった場合
        :return float fidelity_like: フィデリティと単調関係にある値
        """
        # 量子回路をshots回数だけ実行する
        if on_ibmq:
            job = qiskit.execute(qiskit.transpile(self.quantum_circuit, back_end), backend=back_end, shots=shots)
        else:
            simulator = qiskit.Aer.get_backend(back_end)
            job = qiskit.execute(self.quantum_circuit, simulator, shots=shots)
        # 結果を取得する
        result = job.result()
        counts = result.get_counts(self.quantum_circuit)

        try:
            fidelity_like = counts["0"] / shots
        except KeyError:
            # 結果0が得られなかった場合
            msg = "There is no observation '0' in the result of this execution."
            raise ValueError(msg)

        return fidelity_like

    def add_single_qubit_unitary_layer(self, thetas: List[float]) -> None:
        """量子回路にsingle qubit unitary層を追加する

        :param List[float] thetas: 回転角のリスト
        :raises ValueError: thetasの長さが代表量子状態用量子ビット数と異なる場合
        """
        if self.num_trained_qubits != len(thetas):
            msg = f"The shape of thetas must be ({self.num_trained_qubits}, 2), but ({len(thetas)}, 2)."
            raise ValueError(msg)

        # RYとRZゲートを用意する
        for qubit, theta in enumerate(thetas):
            qubit += 1
            theta_y, theta_z = theta
            self.quantum_circuit.ry(qubit=qubit, theta=theta_y)
            self.quantum_circuit.rz(qubit=qubit, phi=theta_z)

    def add_dual_qubit_unitary_layer(self, thetas: List[float]) -> None:
        """量子回路にdual qubit unitary層を追加する

        :param List[float] thetas: 回転角のリスト
        :raises ValueError: thetasの長さが異なる代表量子状態用量子ビット2つの組み合わせ数と異なる場合
        """
        num_combinations = np.arange(1, self.num_trained_qubits).sum()
        if num_combinations != len(thetas):
            msg = f"The shape of thetas must be ({num_combinations}, 2), but ({len(thetas)}, 2)."
            raise ValueError(msg)

        # RYYとRZZゲートを用意する
        for basis_qubit, theta in enumerate(thetas):
            basis_qubit += 1
            theta_y, theta_z = theta

            partner_qubits = np.arange(basis_qubit+1, self.num_trained_qubits+1)
            for partner_qubit in partner_qubits:
                self.quantum_circuit.ryy(theta=theta_y, qubit1=basis_qubit, qubit2=partner_qubit)
                self.quantum_circuit.rzz(theta=theta_z, qubit1=basis_qubit, qubit2=partner_qubit)

    def add_controlled_qubit_unitary_layer(self, thetas: List[float]) -> None:
        """量子回路にcontrolled qubit unitary層を追加する

        :param List[float] thetas: 回転角のリスト
        :raises ValueError: thetasの長さが代表量子状態用量子ビット数と異なる場合
        """
        if self.num_trained_qubits != len(thetas):
            msg = f"The shape of thetas must be ({self.num_trained_qubits}, 2), but ({len(thetas)}, 2)"
            raise ValueError(msg)

        # CRYとCRZゲートを用意する
        for qubit, theta in enumerate(thetas):
            qubit += 1
            theta_y, theta_z = theta
            self.quantum_circuit.cry(theta=theta_y, control_qubit=0, target_qubit=qubit)
            self.quantum_circuit.crz(theta=theta_z, control_qubit=0, target_qubit=qubit)

    def add_load_structure(self) -> None:
        """量子回路にデータ読込用の量子ゲートを追加する
        """
        thetas = np.array([0, 0] * self.num_trained_qubits).reshape(-1, 2)

        # RYとRZゲートを用意する
        for qubit, theta in enumerate(thetas):
            qubit += 1 + self.num_trained_qubits
            theta_y, theta_z = theta
            self.quantum_circuit.ry(qubit=qubit, theta=theta_y)
            self.quantum_circuit.rz(qubit=qubit, phi=theta_z)

    def add_cswap(self) -> None:
        """量子回路に制御スワップを追加する
        """
        # CSWAPゲート(Fredkinゲート)を用意する
        for qubit in range(self.num_trained_qubits):
            trained_qubit = qubit + 1
            loaded_qubit = trained_qubit + self.num_trained_qubits
            self.quantum_circuit.fredkin(0, trained_qubit, loaded_qubit)

    def load_into_qubits(self, data: List[float]) -> None:
        """古典データを回転角としてデータ読込用の量子ゲートに設定する

        :param List[float] data: 古典データ
        :raises ValueError: 与えた古典データと量子回路の入力次元が異なる場合
        """
        # 論文で提案されている方法で古典データを回転角に変換する
        thetas = 2 * np.arcsin(np.sqrt(data))

        # 回転角の数を偶数個にする
        if len(thetas) % 2 != 0:
            thetas = np.append(thetas, 0)

        # 量子回路の入力次元と異なればエラーを発生させる
        if len(thetas) != self.modified_input_size:
            mag = f"The length of data and self.modified_input_size must be same,\nbut the length is {len(thetas)} and self.modified_input_size is {self.modified_input_size}"
            raise ValueError(msg)

        theta_count = 0
        for gate_information in self.quantum_circuit.data:
            # 量子ゲートのうち以下を満たす量子ゲート(= データ読み込み用の量子ゲート)の回転角を変更する
            #    - 読込用量子ビットloaded_qubitに作用する量子ゲートである
            #    - RYあるいはRZゲートである
            if (gate_information[1][0].register.name == "loaded_qubit") and (gate_information[0].name in ["ry", "rz"]):
                gate_information[0]._params = [thetas[theta_count]]
                theta_count += 1

    def draw(self) -> Optional[object]:
        """現在の量子回路を可視化する
        matplotlibが使える場合はFigureを返し、使えない場合は標準出力に出力する

        :return Optional[object]: matplotlibが使える場合のみ量子回路のmatplotlib.figure.Figure
        """
        try:
            return self.quantum_circuit.draw("mpl")
        except:
            print(self.quantum_circuit.draw())

    def train(self, data: List[List[float]], label: str, epochs: int, learning_rate: float, back_end: str, shots: int,
              should_normalize: bool, should_show: bool, should_save_each_epoch: bool, on_ibmq: bool) -> None:
        """学習を実行する

        :param List[List[float]] data: 学習データ
        :param str label: 学習するラベル名
        :param int epochs: 学習回数
        :param float learning_rate: 学習率
        :param str back_end: バックエンド名
        :param int shots: 量子回路の実行回数(期待値を取る回数)
        :param bool should_normalize: 各データを正規化するかどうか
        :param bool should_show: 学習過程を標準出力に出すかどうか
        :param bool should_save_each_epoch: 1エポック毎に量子回路の情報を出力するかどうか
        :param bool on_ibmq: 実機を使うかどうか
        """
        # データの準備をする
        prepared_data = self.normalize_data(data) if should_normalize else data.copy()

        # ラベル名をクラス変数に保存する
        self.label = label

        # 学習中の出力のために学習データ数の1/10を計算する
        duration = len(data) // 10

        # 学習に関する情報を標準出力に出力する
        print(f"============ label {self.label}: Start training ============")
        print(f"the number of epochs is {epochs}")
        print(f"the number of data is {len(data)}")
        print(f"the number of thetas is {self.num_thetas}")
        print(f"the number of iterations per epoch is {len(data)} * {self.num_thetas} = {len(data) * self.num_thetas}")
        print(f"the number of all iterations is {epochs} * {len(data)} * {self.num_thetas} = {epochs * len(data) * self.num_thetas}")

        # 学習を開始する
        for epoch in range(1, epochs+1):
            print(f"epoch {epoch}: ", end="")

            total_loss_over_epochs = 0
            for data_index, vector in enumerate(prepared_data):

                total_loss = 0
                for first_theta_index, thetas in enumerate(self.thetas_list):  # 層に対応する回転角
                    for second_theta_index, theta_yz in enumerate(thetas):  # 代表量子状態用の各量子ビットに作用する量子ゲートたちの回転角
                        for third_theta_index in range(len(theta_yz)):  # 1つ量子ゲートの回転角
                            # フォワード状態のフィデリティを計算する (論文のアルゴリズムの7~13行目辺り)
                            forward_quclassi = QuClassiCircuit(self.modified_input_size)
                            forward_thetas_list = copy.deepcopy(self.thetas_list)
                            forward_thetas_list[first_theta_index][second_theta_index][third_theta_index] += np.pi / (2 * np.sqrt(epoch))
                            forward_quclassi.build_quantum_circuit(self.structure, forward_thetas_list, is_in_train=True)                        
                            forward_quclassi.load_into_qubits(vector)
                            forward_fidelity_like = forward_quclassi.run(back_end=back_end, shots=shots, on_ibmq=on_ibmq)

                            # バックワード状態のフィデリティを計算する (論文のアルゴリズムの15~20行目辺り)
                            backward_quclassi = QuClassiCircuit(self.modified_input_size)
                            backward_thetas_list = copy.deepcopy(self.thetas_list)
                            backward_thetas_list[first_theta_index][second_theta_index][third_theta_index] -= np.pi / (2 * np.sqrt(epoch))
                            backward_quclassi.build_quantum_circuit(self.structure, backward_thetas_list, is_in_train=True)
                            backward_quclassi.load_into_qubits(vector)
                            backward_fidelity_like = backward_quclassi.run(back_end=back_end, shots=shots, on_ibmq=on_ibmq)

                            # 損失値を求める (求めているものは尤度なので、大きい方がよい)
                            self.load_into_qubits(vector)
                            loss = self.run(back_end=back_end, shots=shots, on_ibmq=on_ibmq)
                            total_loss += loss

                            # パラメータを更新する (論文のアルゴリズムの20行目)
                            update_term = -0.5 * (np.log(forward_fidelity_like) - np.log(backward_fidelity_like))
                            self.thetas_list[first_theta_index][second_theta_index][third_theta_index] -= learning_rate * update_term

                            # 更新したパラメータで量子回路を再構成する (lossの計算が必要ない場合は最後に一度だけでもいい)
                            self.quantum_circuit.data = []
                            self.build_quantum_circuit(self.structure, self.thetas_list, is_in_train=True)

                # 設定に応じて、データセット数の約10分の1進捗毎に標準出力へメッセージを出力する
                if should_show and (data_index+1) % duration == 0:
                    print(f"\tCompleted training {data_index+1} data")

                total_loss_over_epochs += total_loss / self.num_thetas

            # クラス変数の学習情報を更新する
            total_loss_over_epochs = len(data) - total_loss_over_epochs  # 最大尤度は1なので、データ数から尤度(損失値)を減少
            self.loss_history.append(total_loss_over_epochs)
            self.epochs += 1
            if self.best_loss is None or total_loss_over_epochs < self.best_loss:
                self.best_loss = total_loss_over_epochs
                self.best_epochs = epoch
                print(f"\tloss = {total_loss_over_epochs} <- the best loss ever")
            else:
                print(f"\tloss = {total_loss_over_epochs}")

            if should_save_each_epoch:
                self.save_parameters_as_json(f"latest_{label}.json")

        print("Successfully trained.")
        print(f"The best loss is {self.best_loss} on {self.best_epochs} epochs")

    def normalize_data(self, data: List[List[float]]) -> np.array:
        """データを正規化する

        :param List[List[float]] data: 古典データ
        :return numpy.array data_normalized: 正規化済み古典データ
        """
        data_normalized = []
        for d in data:
            data_normalized.append(np.array(d) / np.linalg.norm(d))

        data_normalized = np.array(data_normalized)
        return data_normalized

    def save_parameters_as_json(self, output_path: str) -> None:
        """量子回路の情報をjsonで保存する

        :param str output_path: 出力パス
        :raises ValueError: output_pathの拡張子が.jsonではなかった場合
        """
        if ".json" != Path(output_path).suffix:
            msg = f"The suffix of output_path must be .json, but this output_path is {output_path}"
            raise ValueError(msg)

        # jsonでの保存に向けて辞書に必要なデータをまとめる
        model_information = dict()

        model_information["modified_input_size"] = self.modified_input_size
        model_information["structure"] = self.structure

        thetas_list = []
        for ts in self.thetas_list:
            thetas = []
            for theta_yz in ts:
                thetas.append(list(theta_yz))
            thetas_list.append(thetas)
        model_information["thetas_list"] = thetas_list

        model_information["label"] = self.label
        model_information["epochs"] = self.epochs
        model_information["loss_history"] = self.loss_history
        model_information["best_loss"] = self.best_loss

        # json形式で出力パスに出力する
        with open(output_path, 'w', encoding="utf-8") as output:
            json.dump(model_information, output, indent=4)

    @classmethod
    def load_parameters_from_json(cls, model_path: str) -> object:
        """量子回路をjsonファイルから生成する

        :param str model_path: 読込jsonファイルのパス
        :raises ValueError: model_pathの拡張子が.jsonではなかった場合
        :return QuClassiCircuit loaded_quclassi_circuit: 読込んだQuClassiCircuitオブジェクト
        """
        if ".json" != Path(model_path).suffix:
            msg = f"The suffix of model_path must be .json, but this model_path is {model_path}"
            raise ValueError(msg)

        # モデルの情報を読込む
        with open(model_path) as model_file:
            model_information = json.load(model_file)

        # QuClassiオブジェクトを生成し、読込んだ情報を格納する
        loaded_quclassi_circuit = cls(model_information["modified_input_size"])
        loaded_quclassi_circuit.build_quantum_circuit(model_information["structure"], model_information["thetas_list"])
        loaded_quclassi_circuit.label = model_information["label"]
        loaded_quclassi_circuit.epochs = model_information["epochs"]
        loaded_quclassi_circuit.loss_history = model_information["loss_history"]
        loaded_quclassi_circuit.best_loss = model_information["best_loss"]

        return loaded_quclassi_circuit

    def calculate_likelihood(self, data: Union[List[List[float]], List[float]], back_end: str, shots: int, should_normalize: bool, on_ibmq: bool) -> List[float]:
        """尤度(self.run()関数の出力)を計算する

        :param Union[List[List[float]], List[float]] data: 古典データ
        :param str back_end: バックエンド
        :param int shots: 量子回路の実行回数(期待値を取る回数)
        :param bool should_normalize: 各データを正規化するかどうか
        :param bool on_ibmq: 実機を使うかどうか
        :return List[float] likelihoods: 尤度
        """
        data_np = np.array(data)
        if should_normalize:
            prepared_data = self.normalize_data(data) if len(data_np.shape) != 1 else self.normalize_data([data])
        else:
            prepared_data = data if len(data_np.shape) != 1 else [data]

        likelihoods = []
        for vector in prepared_data:
            self.load_into_qubits(vector)
            fidelity_like = self.run(back_end=back_end, shots=shots, on_ibmq=on_ibmq)

            likelihoods.append(fidelity_like)

        return likelihoods
