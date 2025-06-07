import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from committor_nn import CommittorNN
from initial_simulation import simulation_start
from committor_update import learn
from transition_simulation import chain_A_step, chain_B_step
from plotting import plot_end_points_with_committor
from MB_potential import V




# Initialize the model
model = CommittorNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def launch_chains(model, optimizer, N):
    """
    Launch the sampling process for both Basin A and Basin B.
    The function runs simulations from both basins, alternates between
    Basin A and Basin B, and updates the model using learned committor values.
    
    Args:
        model (torch.nn.Module): The trained committor model.
        N (int): The number of particles to simulate for both basins.
    """
    i = 1  # Counter for iterations
    # Initialize the starting points for both basins
    A_start, B_start = simulation_start(N)  
    print("Starting points initialization complete.")

    # Separate the start points for Basin A and Basin B
    A_start, max_coor = A_start[:-1], A_start[-1]
    B_start, min_coor = B_start[:-1], B_start[-1]
    
    # Flags to track when each chain has finished
    chain_A_done = False
    chain_B_done = False
    
    # Containers to store the start and end coordinates
    all_start_coords = torch.empty((0, 2))
    all_end_coords = torch.empty((0, 2))
    
    # Current end coordinates for both chains
    cur_A_end_coords, cur_B_end_coords = max_coor.unsqueeze(0), min_coor.unsqueeze(0)

    # Main simulation loop: alternates between Basin A and Basin B
    while A_start.numel() and B_start.numel():
        # Process Basin A chain
        if chain_A_done:
            print("Chain A finished.")
            A_start, max_coor = A_start[:-1], A_start[-1]
            cur_A_end_coords = max_coor.unsqueeze(0)
        
        all_start_coords = torch.cat((all_start_coords, max_coor.unsqueeze(0)), dim=0) 
        chain_A_done, max_coor, new_A_coords, cur_A_end_coords = chain_A_step(model, max_coor, cur_A_end_coords)
        all_end_coords = torch.cat((all_end_coords, new_A_coords), dim=0) 

        # Process Basin B chain
        if chain_B_done:
            print("Chain B finished.")
            B_start, min_coor = B_start[:-1], B_start[-1]
            cur_B_end_coords = min_coor.unsqueeze(0)
        
        all_start_coords = torch.cat((all_start_coords, min_coor.unsqueeze(0)), dim=0) 
        chain_B_done, min_coor, new_B_coords, cur_B_end_coords = chain_B_step(model, min_coor, cur_B_end_coords)
        all_end_coords = torch.cat((all_end_coords, new_B_coords), dim=0) 

        # Update the model with the new data
        loss = learn(model, all_start_coords, all_end_coords, optimizer)

        # Every 50 steps, print the progress and plot the results
        if i % 50 == 0:
            print(f"Iteration {i}: Start coordinates size = {all_start_coords.size()}")
            plot_end_points_with_committor(all_start_coords[1::2], all_start_coords[0::2], V, model)
            print(f"Loss at iteration {i}: {loss.item()}")
        
        i += 1  # Increment the iteration counter


def main(N):
    """
    Main function to run the simulation, training, and rate estimation.
    
    Args:
        N (int): The number of particles to simulate.
    """
    
    # Initialize the model
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    model = CommittorNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    launch_chains(model, optimizer, N)



if __name__ == "__main__":
    # Start the simulation with 100 particles
    main(100)
