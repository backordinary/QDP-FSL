# https://github.com/qis-unipr/dqc-circuit/blob/cc887a47f2f52a08229f24e9b40661b296cb431e/dqc_parser/lexer.py
import os

import qiskit.qasm.qasmlexer
from qiskit.qasm.qasmlexer import QasmLexer

CORE_LIBS_PATH = qiskit.qasm.qasmlexer.CORE_LIBS_PATH
CORE_LIBS = os.listdir(CORE_LIBS_PATH)


class DQCLexer(QasmLexer):

    def __init__(self, filename):
        self.reserved.update({"entswap": "ENTSWAP"})
        self.tokens.append("ENTSWAP")
        self.reserved.update({"remoteCx": "REMOTECX"})
        self.tokens.append("REMOTECX")
        self.reserved.update({"epr": "EPR"})
        self.tokens.append("EPR")
        super().__init__(filename)
