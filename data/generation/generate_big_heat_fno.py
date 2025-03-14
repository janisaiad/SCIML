import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

def solve_heat_square(boundary_conditions:np.ndarray, initial_conditions:np.ndarray, t:np.ndarray, x:np.ndarray, y:np.ndarray, alpha:float=1.0)->np.ndarray:
    nx, ny = len(x), len(y)  # [scalar], [scalar]
    nt = len(t)  # [scalar]
    
    dx = x[1] - x[0]  # [scalar] 
    dy = y[1] - y[0]  # [scalar]
    
    solution = np.zeros((nx, ny, nt))  # [nx, ny, nt]
    solution[:, :, 0] = initial_conditions  # [nx, ny]
    
    for n in tqdm(range(1, nt), desc="Solving heat equation"):
        dt = t[n] - t[n-1]  # [scalar]
        solution[:, :, n] = solution[:, :, n-1].copy()  # [nx, ny]
        
        for i in range(nx):
            for j in range(ny):
                ip1 = (i + 1) % nx  # [scalar] # Periodic boundary conditions
                im1 = (i - 1) % nx  # [scalar] 
                jp1 = (j + 1) % ny  # [scalar]
                jm1 = (j - 1) % ny  # [scalar]
                
                laplacian = (solution[ip1, j, n-1] - 2*solution[i, j, n-1] + solution[im1, j, n-1]) / dx**2 + \
                           (solution[i, jp1, n-1] - 2*solution[i, j, n-1] + solution[i, jm1, n-1]) / dy**2  # [scalar]
                
                solution[i, j, n] = solution[i, j, n-1] + dt * alpha * laplacian  # [scalar]
    
    return solution  # [nx, ny, nt]

def create_gaussian_initial_condition(X:np.ndarray, Y:np.ndarray, mean_x:float, mean_y:float, variance:float)->np.ndarray:
    """Creates a 2D Gaussian initial condition"""
    return np.exp(-((X - mean_x)**2 + (Y - mean_y)**2)/(2*variance))

def create_big_heat_data(N:int=200, nx:int=100, ny:int=100, nt:int=50, alpha:float=0.01)->None:
    """Creates a large dataset of heat equation solutions with periodic boundary conditions and varied initial conditions
    
    Args:
        N: Number of samples (100 sinusoidal + 100 gaussian)
        nx: Number of points in x direction
        ny: Number of points in y direction 
        nt: Number of time steps
        alpha: Diffusion coefficient
    """
    # Initialize arrays
    mu = np.zeros((N, nx, ny, 1))  # [N, nx, ny, 1]
    sol = np.zeros((N, nx, ny, nt))  # [N, nx, ny, nt]
    xs = np.zeros((N, nx, ny, 2))  # [N, nx, ny, 2]
    
    # Grid setup
    X, Y = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny))  # [nx, ny], [nx, ny]
    t = np.linspace(0,1,nt)  # [nt]
    x = np.linspace(0,1,nx)  # [nx]
    y = np.linspace(0,1,ny)  # [ny]
    
    # First half: sinusoidal patterns with different frequencies
    N_sin = N//2
    freqs = np.linspace(0.5, 5, N_sin)  # [N_sin]
    
    for i, freq in tqdm(enumerate(freqs), desc="Creating sinusoidal patterns"):
        initial_conditions = np.sin(2*np.pi*freq*X)*np.cos(2*np.pi*freq*Y) + \
                           np.cos(2*np.pi*freq*X)*np.sin(2*np.pi*freq*Y)  # [nx, ny]
        
        mu[i, :, :, 0] = initial_conditions  # [nx, ny]
        sol[i] = solve_heat_square(None, initial_conditions, t, x, y, alpha)  # [nx, ny, nt]
        xs[i, :, :, :] = np.stack([X, Y], axis=-1)  # [nx, ny, 2]
    
    # Second half: Gaussian patterns with different means and variances
    means_x = np.random.uniform(0.2, 0.8, N-N_sin)  # Random centers
    means_y = np.random.uniform(0.2, 0.8, N-N_sin)
    variances = np.random.uniform(0.01, 0.2, N-N_sin)  # Different spreads
    
    for i, (mean_x, mean_y, var) in tqdm(enumerate(zip(means_x, means_y, variances)), desc="Creating Gaussian patterns"):
        initial_conditions = create_gaussian_initial_condition(X, Y, mean_x, mean_y, var)
        
        idx = i + N_sin
        mu[idx, :, :, 0] = initial_conditions  # [nx, ny]
        sol[idx] = solve_heat_square(None, initial_conditions, t, x, y, alpha)  # [nx, ny, nt]
        xs[idx, :, :, :] = np.stack([X, Y], axis=-1)  # [nx, ny, 2]

    # Save with compression due to large size
    np.savez_compressed("data/test_data/big_dataset_fno/data.npz",
                       mu=mu, sol=sol, xs=xs)
    
    params = {
        "n_mu": N,
        "n_sinusoidal": N_sin,
        "n_gaussian": N-N_sin,
        "nx": nx, 
        "ny": ny,
        "nt": nt,
        "alpha": alpha,
        "freq_range": [float(freqs[0]), float(freqs[-1])],
        "gaussian_variance_range": [float(min(variances)), float(max(variances))],
        "initial_conditions": [
            "np.sin(2*np.pi*freq*X)*np.cos(2*np.pi*freq*Y) + np.cos(2*np.pi*freq*X)*np.sin(2*np.pi*freq*Y)",
            "np.exp(-((X - mean_x)**2 + (Y - mean_y)**2)/(2*variance))"
        ],
        "boundary_conditions": "periodic"
    }
    with open("data/test_data/big_dataset_fno/params.json", "w") as f:
        json.dump(params, f)

if __name__ == "__main__":
    create_big_heat_data()
