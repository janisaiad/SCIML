import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os

def solve_burgers_square(initial_conditions:np.ndarray, t:np.ndarray, x:np.ndarray, y:np.ndarray)->np.ndarray:
    '''
    Solve 2D inviscid Burgers equation with periodic BC using Godunov scheme
    # Format: (2,nt,nx,ny) for (u,v) components
    # Equations:
    # u_t + u*u_x + v*u_y = 0
    # v_t + u*v_x + v*v_y = 0
    '''
    nx, ny = len(x), len(y)
    nt = len(t)
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    u = np.zeros((nt, nx, ny))
    v = np.zeros((nt, nx, ny))
    
    u[0] = initial_conditions[0]
    v[0] = initial_conditions[1]
    
    for n in tqdm(range(1, nt), desc="Solving Burgers equation"):
        dt = t[n] - t[n-1]
        
        # CFL condition
        max_speed = max(np.max(abs(u[n-1])), np.max(abs(v[n-1])))
        dt = min(dt, 0.5*min(dx,dy)/max_speed) if max_speed > 0 else dt
        
        for i in range(nx):
            for j in range(ny):
                # Periodic indices
                ip1 = (i + 1) % nx
                im1 = (i - 1) % nx
                jp1 = (j + 1) % ny
                jm1 = (j - 1) % ny
                
                # Godunov flux for u
                if u[n-1,i,j] >= 0:
                    flux_x_u = u[n-1,i,j]**2/2 - u[n-1,im1,j]**2/2
                else:
                    flux_x_u = u[n-1,ip1,j]**2/2 - u[n-1,i,j]**2/2
                    
                if v[n-1,i,j] >= 0:
                    flux_y_u = u[n-1,i,j]*v[n-1,i,j] - u[n-1,i,jm1]*v[n-1,i,jm1]
                else:
                    flux_y_u = u[n-1,i,jp1]*v[n-1,i,jp1] - u[n-1,i,j]*v[n-1,i,j]
                
                # Godunov flux for v
                if u[n-1,i,j] >= 0:
                    flux_x_v = u[n-1,i,j]*v[n-1,i,j] - u[n-1,im1,j]*v[n-1,im1,j]
                else:
                    flux_x_v = u[n-1,ip1,j]*v[n-1,ip1,j] - u[n-1,i,j]*v[n-1,i,j]
                    
                if v[n-1,i,j] >= 0:
                    flux_y_v = v[n-1,i,j]**2/2 - v[n-1,i,jm1]**2/2
                else:
                    flux_y_v = v[n-1,i,jp1]**2/2 - v[n-1,i,j]**2/2
                
                # Update
                u[n,i,j] = u[n-1,i,j] - dt/dx * flux_x_u - dt/dy * flux_y_u
                v[n,i,j] = v[n-1,i,j] - dt/dx * flux_x_v - dt/dy * flux_y_v
    
    return np.stack([u, v])

def create_burgers_data(n_mu:int, nt:int, nx:int, ny:int)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Generate data for different initial conditions with uniform transport velocities"""
    os.makedirs("data/test_data/example_data/burgers2d", exist_ok=True)
    
    for i in tqdm(range(n_mu), desc="Creating Burgers data"):
        # Random uniform velocities between -1 and 1
        u_speed = np.random.uniform(-1, 1)
        v_speed = np.random.uniform(-1, 1)
        
        x = np.linspace(0, 2*np.pi, nx)
        y = np.linspace(0, 2*np.pi, ny)
        X, Y = np.meshgrid(x, y)
        
        # Uniform velocity field - same speed at every point
        u0 = np.ones((nx, ny)) * u_speed
        v0 = np.ones((nx, ny)) * v_speed
        initial_conditions = np.array([u0, v0])
        
        t = np.linspace(0, 1, nt)
        sol = solve_burgers_square(initial_conditions, t, x, y)
        
        X, Y, T = np.meshgrid(x, y, t)
        points = np.column_stack((X.ravel(), Y.ravel(), T.ravel()))
        
        np.save(f"data/test_data/example_data/burgers2d/mu_{i}.npy", initial_conditions)
        np.save(f"data/test_data/example_data/burgers2d/xs_{i}.npy", points)
        np.save(f"data/test_data/example_data/burgers2d/sol_{i}.npy", sol)
        
        with open(f"data/test_data/example_data/burgers2d/params.json", "w") as f:
            json.dump({
                "u_speed": float(u_speed),
                "v_speed": float(v_speed),
                "nt": nt,
                "nx": nx,
                "ny": ny
            }, f)

if __name__ == "__main__":
    create_burgers_data(n_mu=40, nt=20, nx=32, ny=32)
