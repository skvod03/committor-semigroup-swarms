import torch
from MB_potential import V
from utils import in_basin_A, in_basin_B

def simulate_swarm(start):
    """
    Simulate a swarm of particles using Langevin dynamics.
    """
    k, tau, dt = 1000, 10, 0.001  # Time parameters
    beta = 1.0  # Assuming a value for beta

    # Initialize positions (X) from start, and repeat for k particles
    X = start.requires_grad_().unsqueeze(0).repeat(k, 1)  # Shape: (k, 2)
    Z = torch.randn((k, 2))

    for _ in range(tau):
        v = V(X)
        dv = torch.autograd.grad(outputs=v, inputs=X, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        X = X - dv * dt + torch.sqrt(torch.tensor(2 * dt / beta)) * Z

        if in_basin_A(X).any() or in_basin_B(X).any():
            break

        Z = torch.randn((k, 2))

    return X.detach()

def chain_A_step(model, max_coor, cur_A_end_coords):
    """
    Simulate a single step of the chain from Basin A to Basin B.
    """
    new_A_coords = simulate_swarm(max_coor)  # Simulate a new swarm step
    chain_A_done = in_basin_B(new_A_coords).any()  # Check if any point reaches Basin B

    cur_A_end_coords = torch.cat((cur_A_end_coords, new_A_coords), dim=0)
    A_commitors = torch.where(in_basin_A(cur_A_end_coords), torch.tensor(0.0), model(cur_A_end_coords))

    if torch.max(A_commitors) > model(max_coor.unsqueeze(0)).item():
        max_idx = torch.argmax(A_commitors, dim=0)
        max_coor = cur_A_end_coords[max_idx]
    
    return chain_A_done, max_coor, new_A_coords, cur_A_end_coords

def chain_B_step(model, min_coor, cur_B_end_coords):
    """
    Simulate a single step of the chain from Basin B to Basin A.
    """
    new_B_coords = simulate_swarm(min_coor)  # Simulate a new swarm step
    chain_B_done = in_basin_A(new_B_coords).any()  # Check if any point reaches Basin A

    cur_B_end_coords = torch.cat((cur_B_end_coords, new_B_coords), dim=0)
    B_committors = torch.where(in_basin_A(cur_B_end_coords), torch.tensor(0.0), model(cur_B_end_coords))

    if torch.min(B_committors) < model(min_coor.unsqueeze(0)).item():
        min_idx = torch.argmin(B_committors, dim=0)
        min_coor = cur_B_end_coords[min_idx]
    
    return chain_B_done, min_coor, new_B_coords, cur_B_end_coords

