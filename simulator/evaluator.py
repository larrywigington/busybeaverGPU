import numpy as np
from numba import cuda
from simulator.simulator_gpu import simulate_batch

def evaluate_batch(machine_arrays, max_steps=10000, tape_size=1000):
    """
    Host-side function to launch the GPU kernel.
    machine_arrays: list of transition arrays (one per machine)
    """

    num_machines = len(machine_arrays)

    # Allocate device arrays
    transitions = np.array(machine_arrays, dtype=np.int32)
    tapes = np.zeros((num_machines, tape_size), dtype=np.int32)
    heads = np.full((num_machines,), tape_size // 2, dtype=np.int32)  # Start head at center
    states = np.zeros((num_machines,), dtype=np.int32)
    halts = np.zeros((num_machines,), dtype=np.bool_)

    # Transfer to device
    d_transitions = cuda.to_device(transitions)
    d_tapes = cuda.to_device(tapes)
    d_heads = cuda.to_device(heads)
    d_states = cuda.to_device(states)
    d_halts = cuda.to_device(halts)

    threads_per_block = 128
    blocks_per_grid = (num_machines + threads_per_block - 1) // threads_per_block

    # Launch kernel
    simulate_batch[blocks_per_grid, threads_per_block](d_transitions, d_tapes, d_heads, d_states, d_halts, max_steps)

    # Copy results back
    return d_halts.copy_to_host()
