# %%
from typing import NamedTuple

import jax
import matplotlib.pyplot as plt


# %%
class FuzzyParams(NamedTuple):
    threshold: float
    sharpness: float

    @classmethod
    def initialize(cls, *, key):
        th_key, sm_key = jax.random.split(key, 2)
        return cls(
            jax.random.normal(key=th_key) / 3, jax.random.normal(key=sm_key) ** 2
        )


# Basis
def sigmoid(x, alpha=1):
    return 1 / (1 + jax.numpy.exp(-alpha * (x)))


def fuzzy_and(x, y):
    return jax.numpy.log(jax.numpy.exp(x) + jax.numpy.exp(y))


def fuzzy_not(x):
    return 1 - x


@jax.jit
def gt_fuzzifier(x, params: FuzzyParams):
    return sigmoid(x - sigmoid(params.threshold), params.sharpness)


# Composite


def fuzzy_nor(x, y):
    return fuzzy_and(fuzzy_not(x), fuzzy_not(y))


def fuzzy_or(x, y):
    return fuzzy_not(fuzzy_nor(x, y))


@jax.jit
def lt_fuzzifier(x, params: FuzzyParams):
    return fuzzy_not(gt_fuzzifier(x, FuzzyParams(params.threshold, params.sharpness)))


@jax.jit
def eq_fuzzifier(x, params: FuzzyParams):
    return fuzzy_and(
        fuzzy_not(gt_fuzzifier(x, params)), fuzzy_not(lt_fuzzifier(x, params))
    )


def plot_fuzzy_gt(x, params: FuzzyParams, **kwargs):
    x = jax.numpy.array(x)
    plt.plot(x, gt_fuzzifier(x, params), **kwargs)


def plot_fuzzy_lt(x, params: FuzzyParams, **kwargs):
    x = jax.numpy.array(x)
    plt.plot(x, lt_fuzzifier(x, params), **kwargs)


def plot_fuzzy_eq(x, params: FuzzyParams, **kwargs):
    x = jax.numpy.array(x)
    plt.plot(x, eq_fuzzifier(x, params), **kwargs)
