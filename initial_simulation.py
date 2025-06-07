import torch
from utils import in_basin_A, in_basin_B
from MB_potential import V

def langevin_update(X, Z, Z_prev, dt, beta):
    """
    Helper function for Langevin dynamics update. It applies the gradient and random noise update 
    to the particle position X.
    
    Args:
        X (torch.Tensor): Current particle position.
        Z (torch.Tensor): Current random noise.
        Z_prev (torch.Tensor): Previous random noise.
        dt (float): Time step.
        beta (float): Parameter controlling the noise magnitude in Langevin dynamics.
        
    Returns:
        torch.Tensor: Updated particle position.
        torch.Tensor: Updated random noise for the next iteration.
    """
    # Compute the potential and gradients
    v = V(X)  # Compute potential V at current position X
    v.backward()  # Compute the gradients of the potential w.r.t. X
    dv = X.grad  # Gradient of potential
    
    # Update the particle position using Langevin dynamics formula
    X.data.sub_(dv * dt - torch.sqrt(torch.tensor(dt / (2 * beta))) * (Z + Z_prev))
    X = X.detach().clone().requires_grad_(True)  # Detach X from the computation graph to avoid tracking gradients

    # Return the updated particle position and random noise
    return X, Z

def simulation_start(N):
    """
    Initializes and simulates N particles in two basins (A and B) using Langevin dynamics.
    Each particle starts in a specific basin and is updated through Langevin dynamics until
    it exits the basin. The final positions of the particles in each basin are returned.

    Args:
        N (int): The number of particles to simulate for each basin.

    Returns:
        tuple: Two tensors containing the starting positions of the particles in basin A and basin B.
            - A_start: Tensor of shape (N, 2) representing the start positions of particles in basin A.
            - B_start: Tensor of shape (N, 2) representing the start positions of particles in basin B.
    """
    
    # Time step for Langevin dynamics
    dt = 0.001  
    Z = torch.randn(2)  # Initialize random noise for Langevin update
    Z_prev = torch.randn(2)  # Initialize previous random noise for Langevin update
    
    # Initialize tensors to store the particle positions for each basin
    A_start = torch.zeros((N, 2))  
    B_start = torch.zeros((N, 2))  

    # Simulating for basin A
    X = torch.tensor([-0.56, 1.44]).requires_grad_(True)  # Initial position in basin A
    centers = 0  # Counter for particles in basin A
    beta = 1  # Temperature parameter for Langevin dynamics (controls noise magnitude)

    while centers < N:
        X_prev = X.clone().detach()  # Save the previous position of the particle

        X, Z = langevin_update(X, Z, Z_prev, dt, beta)  # Update particle position

        # Check if the particle has moved out of basin A
        if in_basin_A(X_prev) and not in_basin_A(X):
            A_start[centers] = X.detach()  # Store the particle position when it exits basin A
            centers += 1  # Increment counter for particles in basin A

        # Update the noise for Langevin dynamics
        Z_prev = Z
        Z = torch.randn(2)  # New random noise

    # Simulate for basin B
    centers = 0  # Reset counter for basin B
    X = torch.tensor([0.63, 0.03]).requires_grad_(True)  # Initial position in basin B

    while centers < N:
        X_prev = X.clone().detach()  # Save the previous position of the particle

        X, Z = langevin_update(X, Z, Z_prev, dt, beta)  # Update particle position

        # Check if the particle has moved out of basin B
        if in_basin_B(X_prev) and not in_basin_B(X):
            B_start[centers] = X.detach()  # Store the particle position when it exits basin B
            centers += 1  # Increment counter for particles in basin B

        # Update the noise for Langevin dynamics
        Z_prev = Z
        Z = torch.randn(2)  # New random noise

    return A_start, B_start  # Return the final particle positions for both basins
