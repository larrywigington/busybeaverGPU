from numba import cuda
import numpy as np

@cuda.jit
def simulate_batch(transitions, tapes, heads, states, halts, max_steps):
    """
    GPU Kernel to simulate a batch of Turing machines.
    Each thread simulates one machine.
    """
    idx = cuda.grid(1)
    if idx >= tapes.shape[0]:
        return

    head = heads[idx]
    state = states[idx]
    halted = False

    for step in range(max_steps):
        symbol = tapes[idx, head]

        trans_idx = state * 2 + symbol
        new_symbol, dir_bit, new_state = transitions[idx, trans_idx]

        if new_symbol == -1:
            halted = True
            break

        # Write symbol
        tapes[idx, head] = new_symbol

        # Move head
        if dir_bit == 0:
            head -= 1
        else:
            head += 1

        # Bounds check
        if head < 0 or head >= tapes.shape[1]:
            halted = True
            break

        # Update state
        state = new_state

    heads[idx] = head
    states[idx] = state
    halts[idx] = halted
