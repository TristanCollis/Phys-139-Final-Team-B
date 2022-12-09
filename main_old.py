import configparser
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

config = configparser.ConfigParser()
config.read("config.ini")
get_config = partial(config.get, "DEFAULT")


reps = int(get_config("reps"))

a = int(get_config("a"))
b = int(get_config("b"))

with open(get_config("data_path"), "r") as f:
    data = np.array(tuple(int(value) for value in f.read().split()))

n0 = data.size // 2

rng = np.random.default_rng()

lambda_1s = np.zeros(reps)
lambda_2s = np.zeros(reps)
prob = np.zeros(data.shape)
n = np.zeros(reps)

for k in range(reps):
    a_1 = a + sum(data[:n0])
    b_1 = b + n0
    lambda_1 = rng.gamma(a_1, 1 / b_1)

    a_2 = a + sum(data[n0:])
    b_2 = b + data.size - n0
    lambda_2 = rng.gamma(a_2, 1 / b_2)

    lambda_1s[k] = lambda_1
    lambda_2s[k] = lambda_2

    for m in range(data.size):
        P = np.exp(
            np.log(lambda_1 * sum(data[:m]) - m * lambda_1)
            + np.log(lambda_2 * sum(data[m:] - (data.size - m) * lambda_2))
        )
        prob[m] = P

    pnorm = prob / sum(prob)

    for l in range(data.size):
        if rng.random() < pnorm[l]:
            n0 = l
            n[l] = n0

print("finished running")


if config["DEFAULT"].getboolean("save_plots"):
    plt.scatter(lambda_1s, lambda_2s)
    plt.savefig("unoptimized_scatter.png")

    plt.clf()
    plt.hist(n + 2016, bins=data.size)
    plt.savefig("unoptimized_histogram.png")

