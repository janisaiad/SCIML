import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot_heat_solution(sol:np.ndarray,x:np.ndarray,y:np.ndarray,t:np.ndarray,index:int):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, sol[len(t)//2 - 1].T, cmap=cm.viridis, alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Temperature')
    ax.set_title(f'Heat distribution at t={t[len(t)//2 - 1]}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(f"data/plots/heat/heat_solution_{index}.png")
    plt.close()
    
if __name__ == "__main__":
    for mu in range(40):
        sol = np.load(f"data/test_data/example_data/heat2d/mu_{mu}.npy")
        x = np.load(f"data/test_data/example_data/heat2d/xs_{mu}.npy")
        y = np.load(f"data/test_data/example_data/heat2d/ys_{mu}.npy")
        t = np.linspace(0, 1, sol.shape[0])
        plot_heat_solution(sol,x,y,t,mu)