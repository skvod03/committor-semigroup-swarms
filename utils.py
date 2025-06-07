import torch

def create_grid(x_min, x_max, y_min, y_max, resolution=100):
    """
    Create a grid of points for visualization.
    
    Args:
        x_min, x_max, y_min, y_max (float): Grid boundaries.
        resolution (int): Number of points in the grid.
        
    Returns:
        tuple: X grid, Y grid, and position tensors.
    """
    x_vals = torch.linspace(x_min, x_max, resolution)
    y_vals = torch.linspace(y_min, y_max, resolution)
    X, Y = torch.meshgrid(x_vals, y_vals)
    positions = torch.stack([X.ravel(), Y.ravel()], dim=1)
    return X, Y, positions

def in_basin_A(x):
    center_A = torch.tensor([-0.56, 1.44])
    # If input is 1D, treat it as a single point
    if x.dim() == 1:
        return torch.linalg.norm(x - center_A) < 0.1
    # If input is 2D, apply the check for each point (row) and see if any are in the basin
    elif x.dim() == 2:
        return (torch.linalg.norm(x - center_A, dim=1) < 0.1)

def in_basin_B(x):
    center_B = torch.tensor([0.63, 0.03])
    # If input is 1D, treat it as a single point
    if x.dim() == 1:
        return torch.linalg.norm(x - center_B) < 0.1
    # If input is 2D, apply the check for each point (row) and see if any are in the basin
    elif x.dim() == 2:
        return (torch.linalg.norm(x - center_B, dim=1) < 0.1)
