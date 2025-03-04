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
    
    solution[0, 0, :] = boundary_conditions[0]
    solution[0, :, 0] = boundary_conditions[1]
    solution[0, -1, :] = boundary_conditions[2]
    solution[0, :, -1] = boundary_conditions[3]
    
    for n in tqdm(range(1, nt), desc="Solving heat equation"):
        dt = t[n] - t[n-1]
        
        solution[n] = solution[n-1].copy()
        
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                laplacian = (solution[n-1, i+1, j] - 2*solution[n-1, i, j] + solution[n-1, i-1, j]) / dx**2 + \
                           (solution[n-1, i, j+1] - 2*solution[n-1, i, j] + solution[n-1, i, j-1]) / dy**2
                
                solution[n, i, j] = solution[n-1, i, j] + dt * alpha * laplacian
        
        solution[n, 0, :] = boundary_conditions[0]
        solution[n, :, 0] = boundary_conditions[1]
        solution[n, -1, :] = boundary_conditions[2]
        solution[n, :, -1] = boundary_conditions[3]
    
    return solution

def create_heat_data(n_mu:int,nt:int,nx:int,ny:int,alpha:float=1.0)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    mus = np.linspace(0.5,1,n_mu)
    
    for i,mu in tqdm(enumerate(mus),desc="Creating heat data"):
        boundary_conditions = np.array([
            np.cos(np.linspace(0,1,nx)*2*np.pi*mu),
            np.cos(np.linspace(0,1,ny)*2*np.pi*mu),
            np.cos(np.linspace(0,1,nx)*2*np.pi*mu),
            np.cos(np.linspace(0,1,ny)*2*np.pi*mu)
        ])
        
        initial_conditions = np.exp(-(np.linspace(0,1,nx)**2 + np.linspace(0,1,ny)**2))
        
        t = np.linspace(0,1,nt)
        x = np.linspace(0,1,nx)
        y = np.linspace(0,1,ny)
        
        sol = solve_heat_square(boundary_conditions,initial_conditions,t,x,y,alpha)
        xs = np.linspace(0,1,nx)
        ys = np.linspace(0,1,ny)
        
        np.save(f"data/test_data/example_data/heat/mu_{i}.npy",sol)
        np.save(f"data/test_data/example_data/heat/xs_{i}.npy",xs)
        np.save(f"data/test_data/example_data/heat/ys_{i}.npy",ys)
        with open(f"data/test_data/example_data/heat/params.json", "w") as f:
            json.dump({"mu": mu, "nt": nt, "nx": nx, "ny": ny, "alpha": alpha}, f)

if __name__ == "__main__":
    create_heat_data(n_mu =40,nt=20,nx = 20,ny = 20,alpha=0.01)
    # be careful for the stability condition of the scheme