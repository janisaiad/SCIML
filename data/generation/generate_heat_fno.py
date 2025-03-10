import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

# simples way to solve heat with euler scheme, stable
def solve_heat_square(boundary_conditions:np.ndarray, initial_conditions:np.ndarray, t:np.ndarray, x:np.ndarray, y:np.ndarray, alpha:float=1.0)->np.ndarray:
    nx, ny = len(x), len(y)
    nt = len(t)
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    solution = np.zeros((nt, nx, ny))
    
    solution[0, :, :] = initial_conditions
    
    for n in tqdm(range(1, nt), desc="Solving heat equation"):
        dt = t[n] - t[n-1]
        
        solution[n] = solution[n-1].copy() # copy to not interfere
        
        for i in range(nx):
            for j in range(ny):
                ip1 = (i + 1) % nx
                im1 = (i - 1) % nx
                jp1 = (j + 1) % ny
                jm1 = (j - 1) % ny
                
                laplacian = (solution[n-1, ip1, j] - 2*solution[n-1, i, j] + solution[n-1, im1, j]) / dx**2 + \
                           (solution[n-1, i, jp1] - 2*solution[n-1, i, j] + solution[n-1, i, jm1]) / dy**2
                
                solution[n, i, j] = solution[n-1, i, j] + dt * alpha * laplacian
    
    return solution

def create_heat_data(n_mu:int,nt:int,nx:int,ny:int,alpha:float=1.0)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    freqs = np.linspace(0.5,1,n_mu)
    
    for i,freq in tqdm(enumerate(freqs),desc="Creating heat data"):
        X, Y = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny))
        initial_conditions = np.sin(2*np.pi*freq*X)*np.cos(2*np.pi*freq*Y) + \
                           np.cos(2*np.pi*freq*X)*np.sin(2*np.pi*freq*Y)
        
        t = np.linspace(0,1,nt)
        x = np.linspace(0,1,nx)
        y = np.linspace(0,1,ny)
        
        sol = solve_heat_square(None, initial_conditions,t,x,y,alpha)
        xs = np.linspace(0,1,nx)
        ys = np.linspace(0,1,ny)
        
        np.save(f"data/test_data/example_data_fno/heat2d/mu_{i}.npy",initial_conditions)
        
        X, Y, T = np.meshgrid(xs, ys, t)
        points = np.column_stack((X.ravel(), Y.ravel(), T.ravel()))
        np.save(f"data/test_data/example_data_fno/heat2d/xs_{i}.npy",points)
        np.save(f"data/test_data/example_data_fno/heat2d/sol_{i}.npy",sol)
        with open(f"data/test_data/example_data_fno/heat2d/params.json", "w") as f:
            json.dump({"freq": freq,"initial_conditions": "np.sin(2*np.pi*freq*X)*np.cos(2*np.pi*freq*Y) + np.cos(2*np.pi*freq*X)*np.sin(2*np.pi*freq*Y)", "nt": nt, "nx": nx, "ny": ny, "alpha": alpha}, f)

if __name__ == "__main__":
    create_heat_data(n_mu =40,nt=20,nx = 20,ny = 20,alpha=0.01)
    # be careful for the stability condition of the scheme : dt < dx^2/(4*alpha)