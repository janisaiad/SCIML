# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

from sciml.model.fno import FNO
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.config.list_physical_devices('GPU')

p_1 = 400
p_2 = 400
p_3 = 400
epochs = 300 
index = 10


# +
first_network = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(p_1,)),
    tf.keras.layers.Dense(40, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(p_2, activation='relu'),
])

last_network = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(p_2,)),
    tf.keras.layers.Dense(40, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(p_3, activation='relu'),
])


# -

folder_path = "data/test_data/example_data_fno/heat2d/"

# +
n_layers = 3
n_modes = p_2 # we use the same number of modes for the network, perfect fourier transform with well known heisenberg inequality (supp )*(supp F) >= n_modes

activation = 'relu'
kernel_initializer = 'he_normal'
device = "GPU"
n_epochs = epochs

# -

model = FNO(regular_params={"first_network": first_network, "last_network": last_network},fourier_params={"n_layers": n_layers, "n_modes": n_modes, "activation": activation, "kernel_initializer": kernel_initializer}, hyper_params={"p_1": p_1, "p_2": p_2,'p_3':p_3,"device": device,"n_epochs":n_epochs,"index":index})

mus, xs, sol = model.get_data(folder_path)

print(mus.shape)
print(xs.shape)
print(sol.shape)

# +
import os

tf.get_logger().setLevel('ERROR')

# -

train_history = model.fit()

plt.plot(train_history)
plt.grid()
plt.show()


