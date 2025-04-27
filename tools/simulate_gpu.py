import argparse
import json
from pathlib import Path
import cupy as cp
from numba import cuda
import numpy as np

# === GPU Kernel ===
@cuda.jit
def simulate_machines_kernel(rulesets, num_transitions, tape_size, max_steps, steps_taken, halted_flags):
    idx = cuda.grid(1)
    total_threads = cuda.gridsize(1)

    if idx >= rulesets.shape[0]:
        return

    # Local variables
    tape = cuda.local.array(512, dtype=cp.int32)  # Fixed tape size
    head = tape_size // 2
    state = 0
    steps = 0
    halted = False

    # Initialize tape to zeros
    for i in range(tape_size):
        tape[i] = 0

    while steps < max_steps:
        symbol = tape[head]
        rule_idx = state * 2 + symbol

        new_symbol = rulesets[idx, rule_idx, 0]
        move_dir = rulesets[idx, rule_idx, 1]
        new_state = rulesets[idx, rule_idx, 2]

        if new_symbol == -1:
            halted = True
            break

        # Apply transition
        tape[head] = new_symbol
        head += -1 if move_dir == 0 else 1

        # Boundary check (basic wrap-around for now)
        if head < 0:
            head = 0
        elif head >= tape_size:
            head = tape_size - 1

        state = new_state
        steps += 1

    steps_taken[idx] = steps
    halted_flags[idx] = 1 if halted else 0

# === Main Driver ===
def simulate_gpu(case_folder, batch_size=4096, max_steps=1000000, tape_size=512):
    case_folder = Path("rulesets") / case_folder
    block_root = case_folder / "blocks"
    index_file = case_folder / "index.jsonl"

    results_folder = Path("results") / case_folder.name
    results_folder.mkdir(parents=True, exist_ok=True)
    results_file = results_folder / "simulation_results.jsonl"

    machines = []
    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            machines.append(json.loads(line))

    print(f"[INFO] Loaded {len(machines):,} machines.")

    batch = []
    batch_ids = []
    count = 0

    for machine in machines:
        ruleset_hash = machine["ruleset_hash"]
        block_path = block_root / ruleset_hash[:2] / ruleset_hash[2:4] / f"{ruleset_hash}.json"

        with open(block_path, "r", encoding="utf-8") as f:
            rules = json.load(f)

        # Ensure rules are full matrix (even if missing dummy entries)
        batch.append(rules)
        batch_ids.append(machine["machine_id"])

        if len(batch) >= batch_size:
            run_batch(batch, batch_ids, tape_size, max_steps, results_file)
            batch = []
            batch_ids = []

    # Final batch
    if batch:
        run_batch(batch, batch_ids, tape_size, max_steps, results_file)

    print(f"[INFO] Simulation complete. Results saved to {results_file}")

def run_batch(batch, batch_ids, tape_size, max_steps, results_file):
    num_machines = len(batch)
    num_transitions = len(batch[0])

    # Copy rulesets to GPU
    rulesets_gpu = cp.full((num_machines, num_transitions, 3), -1, dtype=cp.int32)
    for i, ruleset in enumerate(batch):
        for j, transition in enumerate(ruleset):
            rulesets_gpu[i, j, 0] = transition[0]
            rulesets_gpu[i, j, 1] = transition[1]
            rulesets_gpu[i, j, 2] = transition[2]

    steps_taken = cp.zeros(num_machines, dtype=cp.int32)
    halted_flags = cp.zeros(num_machines, dtype=cp.int32)

    threads_per_block = 256
    blocks_per_grid = (num_machines + (threads_per_block - 1)) // threads_per_block

    simulate_machines_kernel[blocks_per_grid, threads_per_block](
        rulesets_gpu, num_transitions, tape_size, max_steps, steps_taken, halted_flags
    )

    steps_cpu = cp.asnumpy(steps_taken)
    halted_cpu = cp.asnumpy(halted_flags)

    with open(results_file, "a", encoding="utf-8") as f:
        for idx in range(num_machines):
            entry = {
                "machine_id": batch_ids[idx],
                "steps_taken": int(steps_cpu[idx]),
                "halted": bool(halted_cpu[idx])
            }
            f.write(json.dumps(entry) + "\n")

    print(f"[INFO] Simulated batch of {num_machines:,} machines.")

# === CLI ===
def main():
    parser = argparse.ArgumentParser(description="Simulate Busy Beaver Machines (GPU)")
    parser.add_argument("--case", required=True, help="Case folder, e.g., s2_k2")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size for simulation")
    parser.add_argument("--max_steps", type=int, default=1000000, help="Max steps before timeout")
    parser.add_argument("--tape_size", type=int, default=512, help="Tape size (cells)")
    args = parser.parse_args()

    simulate_gpu(args.case, args.batch_size, args.max_steps, args.tape_size)

if __name__ == "__main__":
    main()
