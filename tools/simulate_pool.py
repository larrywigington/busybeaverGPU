# tools/simulate_pool.py

import argparse
import json
import os
from pathlib import Path
import numpy as np
import cupy as cp
from numba import cuda
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# === GPU Simulation Kernel ===
@cuda.jit
def simulate_one_machine_kernel(ruleset, tape, steps_taken, halted_flag, tape_size, max_steps):
    head = tape_size // 2
    state = 0
    steps = 0

    for i in range(tape_size):
        tape[i] = 0

    while steps < max_steps:
        symbol = tape[head]
        rule_idx = state * 2 + symbol

        new_symbol = ruleset[rule_idx, 0]
        move_dir = ruleset[rule_idx, 1]
        new_state = ruleset[rule_idx, 2]

        if new_symbol == -1:
            halted_flag[0] = 1
            break

        tape[head] = new_symbol
        head += -1 if move_dir == 0 else 1

        if head < 0:
            head = 0
        elif head >= tape_size:
            head = tape_size - 1

        state = new_state
        steps += 1

    steps_taken[0] = steps

# === CPU Simulation ===
def simulate_single_cpu(rules, max_steps=1000000, tape_size=512):
    tape = [0] * tape_size
    head = tape_size // 2
    state = 0
    steps = 0

    while steps < max_steps:
        symbol = tape[head]
        rule_idx = state * 2 + symbol

        new_symbol, move_dir, new_state = rules[rule_idx]

        if new_symbol == -1:
            return steps, True

        tape[head] = new_symbol
        head += -1 if move_dir == 0 else 1

        if head < 0:
            head = 0
        elif head >= tape_size:
            head = tape_size - 1

        state = new_state
        steps += 1

    return steps, False

# === GPU Simulation ===
def simulate_single_gpu(rules, max_steps=1000000, tape_size=512):
    rules_gpu = cp.asarray(rules, dtype=cp.int32)
    tape_gpu = cp.zeros(tape_size, dtype=cp.int32)
    steps_taken = cp.zeros(1, dtype=cp.int32)
    halted_flag = cp.zeros(1, dtype=cp.int32)

    simulate_one_machine_kernel[1, 1](rules_gpu, tape_gpu, steps_taken, halted_flag, tape_size, max_steps)

    steps = int(steps_taken.get()[0])
    halted = bool(halted_flag.get()[0])

    return steps, halted

# === Promotion for Long-Runners ===
def promote_long_runner(machine_id, pool_file="pools/long_runners.txt"):
    Path(pool_file).parent.mkdir(parents=True, exist_ok=True)
    with open(pool_file, "a", encoding="utf-8") as f:
        f.write(machine_id + "\n")

# === Utility Loaders ===
def load_machine_pool(machine_pool_file):
    with open(machine_pool_file, "r", encoding="utf-8") as f:
        machines = [line.strip() for line in f if line.strip()]
    return machines

def load_checkpoint(checkpoint_path):
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        return checkpoint.get("completed", [])
    return []

def save_checkpoint(completed, checkpoint_path):
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump({"completed": completed}, f, indent=4)

def load_ruleset(case_folder, machine_id):
    case_folder = Path("rulesets") / case_folder
    index_file = case_folder / "index.jsonl"
    block_root = case_folder / "blocks"

    target = None
    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry["machine_id"] == machine_id:
                target = entry
                break

    if target is None:
        raise ValueError(f"Machine ID {machine_id} not found in {index_file}")

    ruleset_hash = target["ruleset_hash"]
    block_path = block_root / ruleset_hash[:2] / ruleset_hash[2:4] / f"{ruleset_hash}.json"

    with open(block_path, "r", encoding="utf-8") as f:
        rules = json.load(f)

    return rules

def console_message(msg):
    print(f"[{Path(os.getcwd()).name}] {msg}")

# === Main Simulation Runner ===
def simulate_pool(machine_pool_file, case_folder, output_name, batch_size=4096, max_steps=1000000, tape_size=512, use_gpu=False):
    pool_name = Path(machine_pool_file).stem
    results_folder = Path("results") / pool_name
    results_folder.mkdir(parents=True, exist_ok=True)
    results_file = results_folder / f"{output_name}.jsonl"
    checkpoint_file = results_folder / f"{output_name}_checkpoint.json"

    all_machines = load_machine_pool(machine_pool_file)
    completed = load_checkpoint(checkpoint_file)

    pending_machines = [m for m in all_machines if m not in completed]
    console_message(f"Loaded {len(all_machines):,} total machines. {len(pending_machines):,} pending.")

    results_fh = open(results_file, "a", encoding="utf-8")

    for batch_start in range(0, len(pending_machines), batch_size):
        batch = pending_machines[batch_start:batch_start + batch_size]
        console_message(f"Processing batch {batch_start // batch_size + 1} with {len(batch):,} machines...")

        with Progress(
                SpinnerColumn(),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TextColumn("[progress.completed]/[progress.total] Machines"),
                TimeElapsedColumn()
        ) as progress:

            task = progress.add_task("[cyan]Simulating...", total=len(batch))

            batch_results = []  # <--- buffer

            for machine_id in batch:
                try:
                    rules = load_ruleset(case_folder, machine_id)

                    if use_gpu:
                        steps, halted = simulate_single_gpu(rules, max_steps=max_steps, tape_size=tape_size)
                    else:
                        steps, halted = simulate_single_cpu(rules, max_steps=max_steps, tape_size=tape_size)

                    entry = {
                        "machine_id": machine_id,
                        "steps_taken": steps,
                        "halted": halted
                    }
                    batch_results.append(entry)
                    completed.append(machine_id)

                    # === Auto-Promote Long Runners ===
                    if not halted and steps >= max_steps:
                        promote_long_runner(machine_id)

                except Exception as e:
                    console_message(f"[WARNING] Failed to simulate {machine_id}: {e}")

                progress.update(task, advance=1)

            # === BULK WRITE once per batch ===
            for entry in batch_results:
                results_fh.write(json.dumps(entry) + "\n")
            results_fh.flush()

            save_checkpoint(completed, checkpoint_file)
            console_message(f"[INFO] Batch completed. Checkpoint saved.")

    results_fh.close()
    console_message("[SUCCESS] All machines simulated. Results saved.")


# === CLI ===
def main():
    parser = argparse.ArgumentParser(description="Simulate a pool of Busy Beaver machines with checkpointing and optional GPU acceleration.")
    parser.add_argument("--pool", required=True, help="Path to machine pool file (one ID per line)")
    parser.add_argument("--case", required=True, help="Case folder where machines are stored (e.g., s2_k2)")
    parser.add_argument("--output", default="results", help="Output result file name (default: results)")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size per save/checkpoint")
    parser.add_argument("--max_steps", type=int, default=1000000, help="Maximum steps before timeout")
    parser.add_argument("--tape_size", type=int, default=512, help="Tape size (cells)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    args = parser.parse_args()

    simulate_pool(
        args.pool,
        args.case,
        args.output,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        tape_size=args.tape_size,
        use_gpu=args.gpu
    )

if __name__ == "__main__":
    main()
