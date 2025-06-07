import matplotlib.pyplot as plt
import torch
import numpy as np

def create_grid(x_min, x_max, y_min, y_max, resolution=100):
    """
    Create a grid of points for visualization within specified bounds.

    This function generates a grid of evenly spaced points in the specified range for both the x and y axes. 
    The resulting grid can be used to visualize functions such as the MB potential and committor function 
    across a 2D plane.

    Args:
        x_min (float): Minimum x-coordinate for the grid.
        x_max (float): Maximum x-coordinate for the grid.
        y_min (float): Minimum y-coordinate for the grid.
        y_max (float): Maximum y-coordinate for the grid.
        resolution (int): Number of grid points along each axis. The grid will have resolution x resolution points.

    Returns:
        tuple: A tuple containing:
            - X (tensor): 2D tensor representing x-coordinates of the grid.
            - Y (tensor): 2D tensor representing y-coordinates of the grid.
            - positions (tensor): A 2D tensor of grid points stacked column-wise, where each row represents a grid point (x, y).
    """
    # Create x and y coordinate values
    x_vals = torch.linspace(x_min, x_max, resolution)
    y_vals = torch.linspace(y_min, y_max, resolution)
    
    # Create a mesh grid from x and y values
    X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')  # Explicitly use 'ij' indexing

    
    # Stack the x and y values into a tensor of positions
    positions = torch.stack([X.ravel(), Y.ravel()], dim=1)
    
    return X, Y, positions

def plot_end_points_with_committor(A_end_coords, B_end_coords, V_func, committor_model):
    """
    Plot the committor function and the MB potential along with the start and end points in a 2D space.

    This function generates a contour plot that shows:
    - The MB potential across the 2D grid of points.
    - The committor values, which represent the probability of transitioning from Basin A to Basin B.
    - The start and end points of simulations, with points in Basin A and Basin B shown in different colors.

    Args:
        A_end_coords (torch.Tensor): The coordinates of the end points in Basin A.
        B_end_coords (torch.Tensor): The coordinates of the end points in Basin B.
        V_func (function): A function representing the MB potential to compute at each grid point.
        committor_model (torch.nn.Module): A trained model to calculate committor values at each grid point.

    Returns:
        None: This function generates and displays a plot using Matplotlib.
    """
    # Create grid for plotting the MB potential and committor function
    X_grid, Y_grid, grid_positions = create_grid(-2, 1.5, -0.5, 2, resolution=100)
    
    # Compute the MB potential at each grid point
    V_vals = V_func(grid_positions).detach().numpy().reshape(X_grid.shape)
    
    # Compute the committor at each grid point
    committor_vals = committor_model(grid_positions).detach().numpy().reshape(X_grid.shape)
    
    # Convert the A_end and B_end coordinates to NumPy for plotting
    A_end_np = A_end_coords.detach().numpy()
    B_end_np = B_end_coords.detach().numpy()
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot the MB potential as a contour map
    contour_potential = plt.contourf(X_grid, Y_grid, V_vals, levels=np.linspace(-15, 0, 25), cmap="viridis")
    plt.colorbar(contour_potential, label="MB Potential V(x)")
    
    # Plot the committor function as another contour plot (ranging 0 to 1)
    contour_committor = plt.contour(X_grid, Y_grid, committor_vals, levels=np.linspace(0, 1, 15), cmap='gray', alpha=0.5)
    plt.colorbar(contour_committor, label="Committor Value")
    
    # Plot the A_end points (red) and B_end points (blue)
    plt.scatter(A_end_np[:, 0], A_end_np[:, 1], color='red', label='A_start Points', s=2, alpha=0.1)
    plt.scatter(B_end_np[:, 0], B_end_np[:, 1], color='blue', label='B_start Points', s=2, alpha=0.1)

    plt.title("A_start and B_start Points with Committor Function on MB Potential")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
