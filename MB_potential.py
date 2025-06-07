import torch

def V(x):
    """
    Multi-dimensional potential function. For simplicity, we sum Gaussian potentials centered at different positions.
    
    Args:
        x (torch.Tensor): Input coordinates, either 1D or 2D tensor.

    Returns:
        torch.Tensor: Potential energy at the given position(s).
    """
    # Constants for the Gaussian functions
    A = torch.tensor([-20, -10, -17, 1.5])
    a = torch.tensor([-1, -1, -6.5, 0.7])
    b = torch.tensor([0, 0, 11, 0.6])
    c = torch.tensor([-10, -10, -6.5, 0.7])
    x0 = torch.tensor([1, 0, -0.5, -1])
    y0 = torch.tensor([0, 0.5, 1.5, 1])

    def _gau(x, idx):
        """
        Compute a multidimensional Gaussian for each index.
        
        Args:
            x (torch.Tensor): The input coordinates.
            idx (int): Index of the current Gaussian function.

        Returns:
            torch.Tensor: The value of the Gaussian at the given position.
        """
        x_0 = x[..., 0]
        x_1 = x[..., 1]
        return A[idx] * torch.exp(a[idx] * torch.square(x_0 - x0[idx]) +
                                  b[idx] * (x_0 - x0[idx]) * (x_1 - y0[idx]) +
                                  c[idx] * torch.square(x_1 - y0[idx]))

    return _gau(x, 0) + _gau(x, 1) + _gau(x, 2) + _gau(x, 3)

def langevin_update(X, V, dt, Z, beta):
    """
    Perform one Langevin dynamics update step.
    
    Args:
        X (torch.Tensor): Current position of particles (batch of positions).
        V (function): Potential function to compute the potential at a given position.
        dt (float): Time step.
        Z (torch.Tensor): Random noise added to the dynamics.
        beta (float): Temperature parameter for Langevin dynamics.
        
    Returns:
        torch.Tensor: Updated positions after Langevin dynamics update.
    """
    v = V(X)  # Evaluate potential
    v.backward()  # Compute gradients of potential
    dv = X.grad  # Get gradient w.r.t X
    
    X.data.sub_(dv * dt - torch.sqrt(torch.tensor(dt / (2 * beta))) * (Z))  # Langevin update
    X = X.detach().clone().requires_grad_(True)  # Detach and reattach for the next iteration

    return X, Z
