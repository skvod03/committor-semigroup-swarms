import torch
from utils import in_basin_A, in_basin_B

def objective(committors_at_start, committors_at_end):
    """
    Compute the objective function for committor training using a symmetrized loss function.
    
    Args:
        committors_at_start (torch.Tensor): Committor values for the start configurations.
        committors_at_end (torch.Tensor): Committor values for the end configurations.
        
    Returns:
        torch.Tensor: The computed loss.
    """
    first = torch.mean((torch.log(committors_at_start) - torch.log(torch.mean(committors_at_end, dim=1))) ** 2)
    second = torch.mean((torch.log(1 - committors_at_start) - torch.log(torch.mean(1 - committors_at_end, dim=1))) ** 2)
    return (first + second) / 2

def learn(model, start_swarm, end_swarm, optimizer):
    """
    Perform one step of training on the committor model using stochastic gradient descent (SGD).
    
    Args:
        model (torch.nn.Module): The neural network model for the committor function.
        start_swarm (torch.Tensor): Start configurations of the swarm.
        end_swarm (torch.Tensor): End configurations of the swarm.
        optimizer (torch.optim.Optimizer): The optimizer for gradient updates.
        
    Returns:
        torch.Tensor: The computed loss after the optimization step.
    """
    k = 1000
    in_A = in_basin_A(end_swarm)  
    in_B = in_basin_B(end_swarm)  
    committors_at_end = torch.where(in_A, torch.tensor(0.0),  torch.where(in_B, torch.tensor(1.0),  model(end_swarm))).view(end_swarm.size(0) // k, k).detach()
    #print(torch.mean(committors_at_end, dim=1))
    # Training loop
    for _ in range(100):  # Example number of epochs
        optimizer.zero_grad()
        # Forward pass: Get committor output for start_swarm
        committors_at_start = model(start_swarm)
        # Compute the custom objective (loss)
        loss = objective(committors_at_start, committors_at_end)

        # Backpropagation and optimization
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters
    return loss
