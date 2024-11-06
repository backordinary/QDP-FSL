# https://github.com/Sav0s/Quantum-RSA-Decrypter/blob/729086608e097997b1774a134b4c0963c764d315/Quantum-RSA-Decryptor/run.py
from qiskit import IBMQ

from Decrypt import IBMDecryptor, QSharpDecryptor, NumericDecryptor, IBMDecryptorReal
from rsa import RSA
from util import *

import argparse


def computate():
    parser = argparse.ArgumentParser(description="Decrypt a rsa-encrypted message.")
    parser.add_argument("-o", "--option", type=str, required=True,
                        help="the way you want to decrypt the message. ibmq, qsharp or numeric")
    parser.add_argument("-f", "--factor", type=int, required=False, help="the factor for the integer factorization")
    parser.add_argument("-k", "--keysize", type=int, required=False, help="the bitsize of the key for the numeric approach")
    
    args = parser.parse_args()

    if args.option == "ibmq":
        factor = parseDefaultFactor(args.factor)
        rsa = RSA(factor = factor)
        decryptor = IBMDecryptor(rsa.n)
    elif args.option == "ibmqreal":
        factor = parseDefaultFactor(args.factor)
        rsa = RSA(factor = factor)
        decryptor = IBMDecryptorReal(rsa.n)
    elif args.option == "qsharp":
        factor = parseDefaultFactor(args.factor)
        rsa = RSA(factor = factor)
        decryptor = QSharpDecryptor(rsa.n)
    elif args.option == "numeric":
        if args.keysize != None:
            bits = args.keysize
            if bits % 2 != 0:
                print("Please provide an even keysize")
                exit(0)
            rsa = RSA(bits=bits)
        else: 
            factor = parseDefaultFactor(args.factor)
            rsa = RSA(factor = factor)
        decryptor = NumericDecryptor(rsa.n)
    else:
        print("Wrong Arguments")
        exit(0)

    print("_____Starting Integer Factorization_____")
    p, q = decryptor.factorize()

    if p == None or q == None:
        return False

    print("_____Found factors!_____")
    print(f"The two factors are: {p} and {q}")

    phi = (p - 1) * (q - 1)
    d = mod_inverse(rsa.e, phi)
    print(f"The regenerated private key has the value\t(d={d}, {rsa.n})")
    return True

    # message = "This should be encrypted"
    # enc = encrypt(message, rsa.e, rsa.n)
    # print(str(enc))

    # dec = decrypt(enc, d, rsa.n)
    # print(str(dec))


if __name__ == '__main__':
    computate()
