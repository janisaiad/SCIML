import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import json

type_of_problem = "burgers2d"

def plot_heat_solution(sol:np.ndarray,x:np.ndarray,y:np.ndarray,t_interior:float,index:int):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, sol[t_interior].T, cmap=cm.viridis, alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Temperature')
    ax.set_title(f'Heat distribution at t={t_interior}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(f"data/plots/{type_of_problem}/heat_solution_{index}.png")
    plt.close()
    
if __name__ == "__main__":
    for index in range(40):
        sol = np.load(f"data/test_data/example_data/{type_of_problem}/sol_{index}.npy")
        matrix = np.load(f"data/test_data/example_data/{type_of_problem}/mu_{index}.npy")
        xs = np.load(f"data/test_data/example_data/{type_of_problem}/xs_{index}.npy") # it is a meshgrid of x,y,t
        X = np.unique(xs[:,0])
        Y = np.unique(xs[:,1])
        with open(f"data/test_data/example_data/{type_of_problem}/params.json", "r") as f:
            params = json.load(f)
        nt = params["nt"]
        plot_heat_solution(sol,X,Y,nt//2 - 1,index)