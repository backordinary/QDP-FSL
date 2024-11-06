# https://github.com/thecookingsenpai/quantumoracle/blob/07ccf087fb073e50512c8e4e56e95a41905e8112/main.py
from qiskit import IBMQ
from qiskit_rng import Generator

IBMQ.load_account()
rng_provider = IBMQ.get_provider(hub='qrng')
backend = rng_provider.backends.ibmq_ourence

generator = Generator(backend=backend)
output = generator.sample(num_raw_bits=1024).block_until_ready()

random_bits = output.extract()
