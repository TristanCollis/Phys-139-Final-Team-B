import config

import matplotlib.pyplot as plt
import numpy as np
from scipy import special as sp
from scipy import stats

reps = int(config.get("reps"))
a = int(config.get("a"))
b = int(config.get("b"))
burn_period = int(config.get("burn_period"))
n = 100

with open(config.get("data_path"), "r") as f:
    x = np.array(tuple(int(value) for value in f.read().split()))

N = x.size

x_sum = np.add.accumulate(x)
j = np.arange(N, dtype=np.uint32)

rng = np.random.default_rng()

for i in range(reps):
    lambda_1 = rng.gamma(a + np.sum(x[:n]), 1/(b + n))
    lambda_2 = rng.gamma(a + np.sum(x[n:]), 1/(b + N-n))

    log_p_n = np.log(lambda_1) * np.take(x_sum, j) + np.log(lambda_2) * np.take(x_sum, N-j) - j * lambda_1 - (N-j) * lambda_2

    print(log_p_n)