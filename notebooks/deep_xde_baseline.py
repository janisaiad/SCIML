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
import os
from tqdm import tqdm
from datetime import datetime
# Define parameters
nx = 30  # Number of points in x direction
ny = 30  # Number of points in y direction
nt = 500  # Number of time steps
alpha = 0.05  # Diffusion coefficient

# Define geometry and time domain
geom = dde.geometry.Rectangle([-1, -1], [1, 1])  # 2D spatial domain
timedomain = dde.geometry.TimeDomain(0, 2)  # Time domain [0,2]
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Generate initial conditions like in generate_big_heat_fno.py
X, Y = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
mean_x, mean_y = np.random.uniform(0.3, 0.7, size=2)
var = np.random.uniform(0.01, 0.1)
initial_conditions = np.exp(-((X - mean_x)**2 + (Y - mean_y)**2)/(2*var))

def pde(x, y):  # x: [batch, 3], y: [batch, 1]
    dy_t = dde.grad.jacobian(y, x, i=0, j=2)  # Time derivative
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)  # Second x derivative  
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)  # Second y derivative
    return dy_t - alpha * (dy_xx + dy_yy)  # Heat equation

def func(x):  # Initial condition function
    return initial_conditions[
        np.searchsorted(np.linspace(-1, 1, nx), x[:, 0]),
        np.searchsorted(np.linspace(-1, 1, ny), x[:, 1])
    ][:, None]

# Define boundary and initial conditions
bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

# Create the TimePDE problem
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=2000,
    num_boundary=400,
    num_initial=400,
    solution=None,
    num_test=10000
)

# Define neural network architecture
layer_size = [3] + [64] * 4 + [1]  # Input: 3 (x,y,t), Hidden: [64,64,64,64], Output: 1
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# Create and compile model
model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])

# Train model
losshistory, train_state = model.train(iterations=50000)

# Save and plot results
save_path = "results/xde/"
os.makedirs(save_path, exist_ok=True)
dde.saveplot(losshistory, train_state, issave=True, isplot=True, save_path=save_path)

# Plot loss history
plt.figure()
plt.semilogy(losshistory.steps, losshistory.loss_train, label="Training loss")
plt.semilogy(losshistory.steps, losshistory.loss_test, label="Testing loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(save_path, f"deep_xde_baseline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"))
plt.show()
