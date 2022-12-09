import configparser
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy import special as sp
from scipy import stats


def g(k, l1, l2, x):
    mantissa = k * (l2 - l1) + np.log(l1 / l2) * np.take(np.add.accumulate(x), k)
    mantissa -= np.max(mantissa)
    return np.exp(
        mantissa
    )


config = configparser.ConfigParser()
config.read("config.ini")
get_config = partial(config.get, "DEFAULT")

reps = int(get_config("reps"))

a0 = int(get_config("a"))
b0 = int(get_config("b"))

with open(get_config("data_path"), "r") as f:
    data = np.array(tuple(int(value) for value in f.read().split()))

n0 = data.size // 2
burn_period = 1000

rng = np.random.default_rng()

lambda_1s = np.zeros(reps)
lambda_2s = np.zeros(reps)
n = np.zeros(reps)
gs = np.zeros(reps)

forward_sum = np.add.accumulate(data)
forward_index = np.arange(data.size)
reverse_sum = np.add.accumulate(np.flip(data))
reverse_index = np.flip(forward_index)

for k in range(reps):
    a1 = a0 + sum(data[:n0])
    b1 = b0 + n0
    lambda_1 = rng.gamma(a1, 1 / b1)

    a2 = a0 + sum(data[n0:])
    b2 = b0 + data.size - n0
    lambda_2 = rng.gamma(a2, 1 / b2)

    lambda_1s[k] = lambda_1
    lambda_2s[k] = lambda_2

    precomputed_g = g(forward_index, lambda_1, lambda_2, data)

    gs[k] = np.max(precomputed_g)
    
    logprob = precomputed_g / np.max(precomputed_g)

    prob = np.exp(logprob)

    pnorm = prob / np.sum(prob)

    n0 = rng.choice(forward_index, p=pnorm)
    n[k] = n0

if config["DEFAULT"].getboolean("save_plots"):
    plt.scatter(lambda_1s[burn_period:], lambda_2s[burn_period:])
    plt.savefig("optimized_scatter.png")

    plt.clf()
    plt.hist(n[burn_period:] + 2017, bins=data.size)
    plt.savefig("optimized_histogram.png")

if config["DEFAULT"].getboolean("save_values"):
    with open("results.txt", 'w') as f:
        f.writelines(
            [
                f"n: {int(np.mean(n[burn_period:] + 2017))}\n",
                f"lambda 1: {np.mean(lambda_1s[burn_period:])}\n",
                f"lambda_2: {np.mean(lambda_2s[burn_period:])}\n"
            ]
        )