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

import deepxde as dde
import numpy as np
from deepxde.backend import tf
import matplotlib.pyplot as plt

# Define unknown parameter C
C = dde.Variable(2.0)

# Define PDE
def pde(x, y):  # x: [batch, 2], y: [batch, 1]
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)  # [batch, 1] 
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)  # [batch, 1]
    return (
        dy_t
        - C * dy_xx
        + tf.exp(-x[:, 1:])
        * (tf.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * tf.sin(np.pi * x[:, 0:1]))
    )  # [batch, 1]

# Define exact solution
def func(x):  # x: [batch, 2]
    return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])  # [batch, 1]

# Define geometry and time domain
geom = dde.geometry.Interval(-1, 1)  # 1D spatial domain
timedomain = dde.geometry.TimeDomain(0, 1)  # Time domain [0,1]
geomtime = dde.geometry.GeometryXTime(geom, timedomain)  # Combined space-time domain

# Define boundary and initial conditions
bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

# Add observation points
observe_x = np.vstack((np.linspace(-1, 1, num=10), np.full((10), 1))).T  # [10, 2]
observe_y = dde.icbc.PointSetBC(observe_x, func(observe_x), component=0)  # [10, 1]

# Create the TimePDE problem
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic, observe_y],
    num_domain=40,
    num_boundary=20, 
    num_initial=10,
    anchors=observe_x,
    solution=func,
    num_test=10000,
)

# Define neural network architecture
layer_size = [2] + [32] * 3 + [1]  # Input: 2, Hidden: [32,32,32], Output: 1
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# Create and compile model
model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"], external_trainable_variables=C)

# Train model
variable = dde.callbacks.VariableValue(C, period=1000)
losshistory, train_state = model.train(iterations=50000, callbacks=[variable])

# Save and plot results
import os
save_path = "results/xde/"
os.makedirs(save_path,exist_ok=True)
dde.saveplot(losshistory, train_state, issave=True, isplot=True, save_path=save_path)

import datetime
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Plot loss history
plt.figure()
plt.semilogy(losshistory.steps, losshistory.loss_train, label="Training loss")
plt.semilogy(losshistory.steps, losshistory.loss_test, label="Testing loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(save_path,f"deep_xde_baseline_{date}.png"))
plt.show()
