import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os

def solve_burgers_square(initial_conditions:np.ndarray, t:np.ndarray, x:np.ndarray, y:np.ndarray, nu:float=0.01)->np.ndarray:
    """
    Solve 2D Burgers equation with periodic boundary conditions using a simple explicit scheme
    u_t + u*u_x + v*u_y = nu*(u_xx + u_yy)
    v_t + u*v_x + v*v_y = nu*(v_xx + v_yy)
    """
    nx, ny = len(x), len(y)
    nt = len(t)
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    u = np.zeros((nt, nx, ny))
    v = np.zeros((nt, nx, ny))
    
    # Set initial conditions
    u[0] = initial_conditions[0]
    v[0] = initial_conditions[1]
    
    for n in tqdm(range(1, nt), desc="Solving Burgers equation"):
        dt = t[n] - t[n-1]
        
        # Copy previous timestep
        u[n] = u[n-1].copy()
        v[n] = v[n-1].copy()
        
        for i in range(nx):
            for j in range(ny):
                # Periodic indices
                ip1 = (i + 1) % nx
                im1 = (i - 1) % nx
                jp1 = (j + 1) % ny
                jm1 = (j - 1) % ny
                
                # Compute spatial derivatives
                u_x = (u[n-1, ip1, j] - u[n-1, im1, j]) / (2*dx)
                u_y = (u[n-1, i, jp1] - u[n-1, i, jm1]) / (2*dy)
                u_xx = (u[n-1, ip1, j] - 2*u[n-1, i, j] + u[n-1, im1, j]) / dx**2
                u_yy = (u[n-1, i, jp1] - 2*u[n-1, i, j] + u[n-1, i, jm1]) / dy**2
                
                v_x = (v[n-1, ip1, j] - v[n-1, im1, j]) / (2*dx)
                v_y = (v[n-1, i, jp1] - v[n-1, i, jm1]) / (2*dy)
                v_xx = (v[n-1, ip1, j] - 2*v[n-1, i, j] + v[n-1, im1, j]) / dx**2
                v_yy = (v[n-1, i, jp1] - 2*v[n-1, i, j] + v[n-1, i, jm1]) / dy**2
                
                # Update u and v
                u[n, i, j] = u[n-1, i, j] - dt * (u[n-1, i, j]*u_x + v[n-1, i, j]*u_y - nu*(u_xx + u_yy))
                v[n, i, j] = v[n-1, i, j] - dt * (u[n-1, i, j]*v_x + v[n-1, i, j]*v_y - nu*(v_xx + v_yy))
    
    return np.stack([u, v])

def create_burgers_data(n_mu:int, nt:int, nx:int, ny:int, nu:float=0.01)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Generate data for different initial conditions"""
    
    # Create output directory if it doesn't exist
    os.makedirs("data/test_data/example_data/burgers2d", exist_ok=True)
    
    for i in tqdm(range(n_mu), desc="Creating Burgers data"):
        # Create random initial conditions with different frequencies
        kx = np.random.randint(1, 4)
        ky = np.random.randint(1, 4)
        
        x = np.linspace(0, 2*np.pi, nx)
        y = np.linspace(0, 2*np.pi, ny)
        X, Y = np.meshgrid(x, y)
        
        # Initial conditions: random combination of sine waves
        u0 = np.sin(kx*X) * np.cos(ky*Y)
        v0 = np.cos(kx*X) * np.sin(ky*Y)
        initial_conditions = np.array([u0, v0])
        
        t = np.linspace(0, 1, nt)
        
        # Solve Burgers equation
        sol = solve_burgers_square(initial_conditions, t, x, y, nu)
        
        # Save data
        X, Y, T = np.meshgrid(x, y, t)
        points = np.column_stack((X.ravel(), Y.ravel(), T.ravel()))
        
        np.save(f"data/test_data/example_data/burgers2d/mu_{i}.npy", initial_conditions)
        np.save(f"data/test_data/example_data/burgers2d/xs_{i}.npy", points)
        np.save(f"data/test_data/example_data/burgers2d/sol_{i}.npy", sol)
        
        with open(f"data/test_data/example_data/burgers2d/params.json", "w") as f:
            json.dump({
                "kx": int(kx),
                "ky": int(ky),
                "nt": nt,
                "nx": nx,
                "ny": ny,
                "nu": nu
            }, f)

if __name__ == "__main__":
    create_burgers_data(n_mu=40, nt=20, nx=32, ny=32, nu=0.01)
    # Note: For stability, ensure dt < min(dx,dy)^2/(4*nu) and dt < min(dx,dy)/max(|u|,|v|)
