#%% [markdown]

# # Unsuppervized learning from domain rules
# ### Associative Memories and Energy Function

# The mathematical framework for this endeavor revolves around associative memories. 
# Consider M memories, each denoted as ρμ ∈ ℝᵈ, where μ ∈ [M]. The key elements here are the energy function and attractor dynamics, 
# crucial for a clustering approach.

# A suitable energy function E(v) for clustering in a d-dimensional Euclidean space should be a continuous function of 
# a point v ∈ ℝᵈ. It should possess M local minima, each corresponding to a memory, and as a particle approaches a memory, 
# its energy should be primarily determined by that single memory.

# The energy function satisfying these requirements is given by:

# \[E(v) = -\frac{1}{\beta} \log \sum_{\mu=1}^{M} \exp(-\beta\| \rho_\mu - v \|_2^2)\]

# Here, β is a positive constant, interpreted as an inverse "temperature." As β increases, the energy function is dominated by 
# the term corresponding to the closest memory, creating M basins of attraction around each memory.

# ### Integration with Jazzy

# We initiate the fuzzy logic system using Jazzy, incorporating expert insights without explicit labels. The clustering loss, 
# inspired by the energy function, refines domain rules without labels by optimising for data separation. 
# Note that this approach requires a strong prior on domain rules to converge.

# ### Validation and Future Possibilities

# To validate the effectiveness of this approach, only a minimal subset of examples needs explicit labels for traditional validation purposes.

# This example stands at the intersection of fuzzy logic, associative memories, and clustering without labels. 
#It not only showcases the adaptability of Jazzy but also provides a glimpse into the potential of refining fuzzy logic systems 
#in real-world scenarios with limited labeled data.


# **Future Possibilities:**
# This first implementation is pretty rough, a lot more work has to be done to find the right loss and hparams.
# The integration of clustering loss within Jazzy opens the door to exciting possibilities. Future experiments could explore:

# - **Dynamic Clustering:** Adapting the clustering approach dynamically as the dataset evolves.
# - **Transfer Learning:** Applying insights gained from one dataset to another, even in different domains.
# - **Optimization Techniques:** Exploring various optimization methods for fine-tuning the fuzzy logic system.


# %%
from itertools import chain
import sys
import os

from sklearn.metrics import classification_report
import numpy as np
import equinox as eqx
import jax
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__)+ '/../src')

from jazzy import (
    fuzzy_not,
    fuzzy_and,
    fuzzy_or,
    gt_fuzzifier,
    lt_fuzzifier,
    FuzzyParams,
    plot_fuzzy_gt,
    plot_fuzzy_lt,
)


# %%
class FuzzyPenguinClassifier(eqx.Module):
    is_heavy_params: FuzzyParams
    has_long_flipper_params: FuzzyParams
    has_deep_culmen_params: FuzzyParams
    has_short_culmen_params: FuzzyParams

    def __init__(
        self,
        *,
        key,
    ) -> None:
        h_key, f_key, c_key, cl_key = jax.random.split(key, 4)
        self.is_heavy_params = FuzzyParams.initialize(key=h_key)
        self.has_long_flipper_params = FuzzyParams.initialize(key=f_key)
        self.has_deep_culmen_params = FuzzyParams.initialize(key=c_key)
        self.has_short_culmen_params = FuzzyParams.initialize(key=cl_key)

    def __call__(self, x):
        culmen_length, culmen_depth, flipper_length, body_mass = x

        is_heavy = gt_fuzzifier(body_mass, self.is_heavy_params)
        has_long_flipper = gt_fuzzifier(flipper_length, self.has_long_flipper_params)
        has_deep_culmen = gt_fuzzifier(culmen_depth, self.has_deep_culmen_params)
        has_short_culmen = lt_fuzzifier(culmen_length, self.has_short_culmen_params)

        is_gentoo = fuzzy_and(fuzzy_or(has_deep_culmen, is_heavy), has_long_flipper)
        is_adelie = fuzzy_and(fuzzy_not(is_gentoo), has_short_culmen)
        is_chinstrap = fuzzy_and(fuzzy_not(is_gentoo), fuzzy_not(is_adelie))

        indics = jax.numpy.stack((is_adelie, is_chinstrap, is_gentoo))
        return indics / indics.sum()

# %%

def get_centroids(cluster_probs, data):
    hard_assignments = jax.numpy.argmax(cluster_probs, axis=1)

    # Create an index array for each cluster
    mask = (hard_assignments[:, None] == jax.numpy.arange(cluster_probs.shape[1]))

    # Use vectorized operations to compute cluster centroids without explicit loops
    centroids = (mask[:, :, None] * data[:, None, :]).sum(axis=0) / mask.sum(axis=0)[:, None]

    return centroids

def get_soft_centroids(cluster_probs, data):
    # Use vectorized operations to compute cluster centroids without explicit loops
    centroids = (cluster_probs[:, :, None] * data[:, None, :]).sum(axis=0) / cluster_probs.sum(axis=0)[:, None]

    return centroids

def energy_function(data, centroids, beta):
    """
    Compute the energy function E(v) for given data and centroids.
    From arXiv:2306.03209v1

    Parameters:
    - data: Input data array of shape (n, features)
    - centroids: Centroids array of shape (n_clusters, features)
    - beta: Positive constant

    Returns:
    - energy: Computed energy value
    """
    # Calculate squared Euclidean distances
    distances_squared = jax.numpy.sum((centroids[:, None] - data[None, :])**2, axis=-1)

    # Exponentiate and sum over centroids
    exp_term = jax.numpy.exp(-beta * distances_squared)
    sum_exp_term = exp_term.sum(axis=0)

    # Normalize and compute the negative logarithm
    energy = -jax.numpy.log(sum_exp_term).mean() / beta

    return energy

#%%

def loss_fn(model, x, regu_weight=5, beta=5.0):
    y_hat = jax.vmap(model)(x)
    centroids = get_soft_centroids(y_hat, x)
    energy = energy_function(x, centroids, beta)

    sharpness = (
        model.is_heavy_params.sharpness
        + model.has_long_flipper_params.sharpness
        + model.has_deep_culmen_params.sharpness
        + model.has_short_culmen_params.sharpness
    )
    return energy + regu_weight / sharpness


grad_loss = jax.jit(jax.grad(loss_fn))
loss = jax.jit(loss_fn)


@jax.jit
def train_step(fuzzifier, x_test, learning_rate, regu_weight, beta):
    y_hat = jax.vmap(fuzzifier)(x_test)
    loss_step = loss(fuzzifier, x_test, regu_weight, beta)
    grads = grad_loss(fuzzifier, x_test)
    new_fuzzifier = jax.tree_map(lambda m, g: m - learning_rate * g, fuzzifier, grads)
    return y_hat, loss_step, grads, new_fuzzifier


# %%
penguins_df = pd.read_csv("penguins_size.csv")

X = penguins_df[
    ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]
]
y = pd.get_dummies(penguins_df["species"])[["Adelie", "Chinstrap", "Gentoo"]]

is_train = ((pd.util.hash_array(penguins_df.index.values) % 100) / 100) > 0.2

X_train, X_test = X[is_train], X[~is_train]
y_train, y_test = y[is_train], y[~is_train]

X_train_mean = X_train.mean()

X_test = ((X_test - X_train.mean()) / X_train.std()).fillna(0)
X_train = ((X_train - X_train.mean()) / X_train.std()).fillna(0)

# %%

clf = FuzzyPenguinClassifier(key=jax.random.PRNGKey(123))

print(clf(X_train.iloc[0]))
# %%
preds = jax.vmap(clf)(X_test.values)

print(
    classification_report(
        np.argmax(y_test.values, axis=1),
        np.argmax(preds, axis=1),
        target_names=y_test.columns,
    )
)

# %%
history = {"epoch": [], "loss": [], "grad_l1": [], "grad_linf": []}

learning_rate = 0.1
regu_weight = 1
beta=5.0

for epoch in tqdm(list(range(100000))):
    y_hat, loss_step, grads, clf = train_step(
        clf, X_train.values, learning_rate, regu_weight, beta
    )
    flat_grads = list(chain(*grads.tree_flatten()[0]))
    history["epoch"].append(epoch)
    history["loss"].append(loss_step)
    history["grad_l1"].append(sum(abs(g) for g in flat_grads) / len(flat_grads))
    history["grad_linf"].append(max(g for g in flat_grads))


# %%

plt.plot(history["epoch"], history["loss"])
plt.title("loss")
plt.figure()
plt.plot(history["epoch"], history["grad_linf"])
plt.title("grad_linf")
# %%
preds = jax.vmap(clf)(X_test.values)

print(
    classification_report(
        np.argmax(y_test.values, axis=1),
        np.argmax(preds, axis=1),
        target_names=y_test.columns,
    )
)

# %%
plot_fuzzy_gt(
    [i / 100 for i in range(100)], clf.is_heavy_params, label="is_heavy_params"
)
plot_fuzzy_gt(
    [i / 100 for i in range(100)], clf.has_long_flipper_params, label="has_long_flipper"
)
plot_fuzzy_gt(
    [i / 100 for i in range(100)],
    clf.has_deep_culmen_params,
    label="has_deep_culmen_params",
)
plot_fuzzy_lt(
    [i / 100 for i in range(100)],
    clf.has_short_culmen_params,
    label="has_short_culmen_params",
)

plt.legend()
plt.figure()
# %%
