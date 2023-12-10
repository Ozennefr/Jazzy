# %% [markdown]
# # Supervised Learning with Fuzzy Logic

# ### Context

# In this example, we leverage fuzzy logic for supervised learning in a classification task. 
# The provided code snippet serves as an example for the integration of fuzzy logic within JAX to tackle explicit labeled data.

# ### Fuzzy Logic in Action

# The fuzzy system, at the heart of this application, processes inputs through fuzzy membership functions.
# Domain logic is implemented using fuzzy AND, OR, and NOT operations, reflecting the imprecision and uncertainty often present in real-world decision-making.

# ### De-Fuzzification and Optimization

# A crucial step in the process is de-fuzzification, where the fuzzy output values are converted into probabilities. 
# This output is then harnessed to optimize thresholds, fine-tuning the fuzzy logic system.

# ### Training and Evaluation

# The training process is no different from any other Equinox model. 
# The provided classification report showcases the effectiveness of the fuzzy logic system in categorizing penguin species.

# ### Model Interpretation

# Visualizing the fuzzy parameters further enhances our understanding of the model. 
# The fuzzy membership functions for various features, such as body mass, flipper length, and culmen depth, 
# provide insights into how the model interprets and classifies different penguin species.



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
def loss_fn(model, x, y, regu_weight=10):
    y_hat = jax.vmap(model)(x)
    bce = y * jax.numpy.log(y_hat + 1e-8) + (1 - y) * jax.numpy.log(1 - y_hat + 1e-8)
    sharpness = (
        model.is_heavy_params.sharpness
        + model.has_long_flipper_params.sharpness
        + model.has_deep_culmen_params.sharpness
        + model.has_short_culmen_params.sharpness
    )
    return jax.numpy.mean(-bce) + regu_weight / sharpness


grad_loss = jax.jit(jax.grad(loss_fn))
loss = jax.jit(loss_fn)


@jax.jit
def train_step(fuzzifier, x_test, y_test, learning_rate, regu_weight):
    y_hat = jax.vmap(fuzzifier)(x_test)
    loss_step = loss(fuzzifier, x_test, y_test, regu_weight)
    grads = grad_loss(fuzzifier, x_test, y_test)
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
history = {"epoch": [], "loss": [], "grad_l1": [], "grad_linf": []}

learning_rate = 0.1
regu_weight = 10

for epoch in tqdm(list(range(5000))):
    y_hat, loss_step, grads, clf = train_step(
        clf, X_train.values, y_train.values, learning_rate, regu_weight
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
