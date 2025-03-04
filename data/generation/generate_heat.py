import numpy as np
from tqdm import tqdm
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def solve_heat_square(boundary_conditions:np.ndarray, initial_conditions:np.ndarray, t:np.ndarray, x:np.ndarray, y:np.ndarray, alpha:float=1.0)->np.ndarray:
    # alpha is the thermal diffusivity coefficient
    
    nx, ny = len(x), len(y)
    nt = len(t)
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    solution = np.zeros((nt, nx, ny))
    
    solution[0, :, :] = initial_conditions # at time 0
    
    # Apply boundary conditions for the first time step
    solution[0, 0, :] = boundary_conditions[0, :]  # bottom
    solution[0, 0, :] = boundary_conditions[0, :]  # bottom
    solution[0, :, 0] = boundary_conditions[1, :]  # left
    solution[0, -1, :] = boundary_conditions[2, :]  # top
    solution[0, :, -1] = boundary_conditions[3, :]  # right
    
    
    # We'll use the implicit scheme: (I - dt*alpha*L)u^{n+1} = u^n
    # This is a very not well conditioned matrix, but it works for now, just prototyping
    # it is stable for any d
    # where L is the Laplacian operator
    
    n_interior = (nx-2) * (ny-2)
    
    
    main_diag = -2.0 * (1.0/dx**2 + 1.0/dy**2) * np.ones(n_interior)
    x_diag = 1.0/dx**2 * np.ones(n_interior-1)
    y_diag = 1.0/dy**2 * np.ones(n_interior-(nx-2))
    
    # boundary between rows
    for i in range(nx-3):
        x_diag[(i+1)*(ny-2)-1] = 0
    
    L = diags([main_diag, x_diag, x_diag, y_diag, y_diag], 
              [0, 1, -1, nx-2, -(nx-2)], 
              shape=(n_interior, n_interior))
    
    
    for n in tqdm(range(1, nt),desc="Solving heat equation"):
        dt = t[n] - t[n-1]
        A = csr_matrix(np.eye(n_interior) - dt * alpha * L)
        
        u_prev = solution[n-1, 1:-1, 1:-1].flatten() # we flatten because we want a 1D array for the linear system
        
        b = u_prev.copy() # we copy because
        
        for i in range(nx-2):
            for j in range(ny-2):
                idx = i * (ny-2) + j
                
                if j == 0:
                    b[idx] += dt * alpha * boundary_conditions[0, i+1] / dy**2
                if j == ny-3:
                    b[idx] += dt * alpha * boundary_conditions[2, i+1] / dy**2
                
                if i == 0:
                    b[idx] += dt * alpha * boundary_conditions[1, j+1] / dx**2
                
                if i == nx-3:
                    b[idx] += dt * alpha * boundary_conditions[3, j+1] / dx**2
        
        u_new = spsolve(A, b)
        
        solution[n, 1:-1, 1:-1] = u_new.reshape((nx-2, ny-2))
        
        solution[n, 0, :] = boundary_conditions[0, :]  # bottom
        solution[n, :, 0] = boundary_conditions[1, :]  # left
        solution[n, -1, :] = boundary_conditions[2, :]  # top
        solution[n, :, -1] = boundary_conditions[3, :]  # right
    
    return solution

def create_heat_data(nx:int,ny:int,nt:int,alpha:float=1.0)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    
    mus = np.random.rand(nt,nx,ny)
    
    for i,mu in tqdm(enumerate(mus),desc="Creating heat data"):
        boundary_conditions = np.sin(np.linspace(0,1,nx)*np.random.normal())
        initial_conditions = np.cos(np.linspace(0,1,nx)*np.random.normal())
        
        t = np.linspace(0,1,nt)
        x = np.linspace(0,1,nx)
        y = np.linspace(0,1,ny)
        
        sol = solve_heat_square(boundary_conditions,initial_conditions,t,x,y,alpha)
        xs = np.linspace(0,1,nx)
        ys = np.linspace(0,1,ny)
        
        
        np.save(f"data/heat/heat_{i}.npy",sol)
        np.save(f"data/heat/xs_{i}.npy",xs)
        np.save(f"data/heat/ys_{i}.npy",ys)
        
        
if __name__ == "__main__":
    create_heat_data(20,20,20,alpha=0.01)