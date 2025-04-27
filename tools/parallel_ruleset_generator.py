import json
import os
import hashlib
from itertools import product, islice
from datetime import datetime
from pathlib import Path
import argparse
import multiprocessing
from functools import partial

import numpy as np
import cupy as cp
from numba import cuda

# === TRANSITION OPTION MAP ===
def generate_transition_options(num_states, num_symbols):
    """Build list of all possible transitions including halt (-1)."""
    options = []
    for new_symbol in range(num_symbols):
        for direction in ['L', 'R']:
            for new_state in range(num_states):
                dir_bit = 0 if direction == 0 else 1
                options.append((new_symbol, dir_bit, new_state))
    options.append(None)  # Represent halting as None
    return options

def hash_ruleset(rules):
    """Hash a serialized ruleset deterministically."""
    rules_json = json.dumps(rules, sort_keys=True)
    return hashlib.sha256(rules_json.encode('utf-8')).hexdigest()

def save_block(block_root, ruleset_hash, rules):
    """Save ruleset to a hashed block path if it doesn't exist."""
    subfolder = block_root / ruleset_hash[:2] / ruleset_hash[2:4]
    subfolder.mkdir(parents=True, exist_ok=True)
    block_path = subfolder / f"{ruleset_hash}.json"
    if not block_path.exists():
        with open(block_path, "w", encoding="utf-8") as f:
            json.dump(rules, f)
    return str(block_path)

def save_machine_index(index_file, machine_entry):
    """Append a machine entry to the index file."""
    index_file.parent.mkdir(parents=True, exist_ok=True)
    with open(index_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(machine_entry) + "\n")

# === CPU VERSION ===
def worker_generate(chunk_id, start_idx, choices_chunk, transition_list, case_folder):
    """Worker function to generate a chunk of rule sets."""
    block_root = case_folder / "blocks"
    index_file = case_folder / "index.jsonl"

    machine_counter = start_idx

    seen_hashes = {}

    for transition_choice in choices_chunk:
        rules = [transition_list[choice_idx] for choice_idx in transition_choice]

        has_halt = any(transition == [-1, 0, -1] for transition in rules)
        if not has_halt:
            continue

        ruleset_hash = hash_ruleset(rules)
        is_canonical = ruleset_hash not in seen_hashes

        if is_canonical:
            save_block(block_root, ruleset_hash, rules)
            seen_hashes[ruleset_hash] = f"TM_{machine_counter:06d}"

        machine_entry = {
            "machine_id": f"TM_{machine_counter:06d}",
            "ruleset_hash": ruleset_hash,
            "is_canonical": is_canonical,
            "timestamp": datetime.now().isoformat()
        }
        save_machine_index(index_file, machine_entry)

        machine_counter += 1

def generate_rulesets_cpu(num_states, num_symbols, num_workers):
    """Main CPU parallel rule set generation."""
    case_folder = Path(f"rulesets/s{num_states}_k{num_symbols}")
    block_root = case_folder / "blocks"
    index_file = case_folder / "index.jsonl"

    block_root.mkdir(parents=True, exist_ok=True)
    index_file.parent.mkdir(parents=True, exist_ok=True)

    num_transitions = num_states * num_symbols
    transition_options = generate_transition_options(num_states, num_symbols)
    num_transition_options = len(transition_options)

    total_combinations = num_transition_options ** num_transitions
    print(f"[INFO] Preparing to generate {total_combinations:,} rule sets...")

    # Pre-build serialized transitions
    prebuilt_transitions = []
    for option in transition_options:
        if option is None:
            prebuilt_transitions.append([-1, 0, -1])
        else:
            prebuilt_transitions.append(list(option))

    all_choices = list(product(range(num_transition_options), repeat=num_transitions))
    chunk_size = len(all_choices) // num_workers + 1

    print(f"[INFO] Using {num_workers} CPU cores...")
    print(f"[INFO] Each core will process ~{chunk_size:,} rule sets.")

    chunks = [
        (i, i * chunk_size, all_choices[i * chunk_size:(i + 1) * chunk_size])
        for i in range(num_workers)
    ]

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(partial(worker_generate, transition_list=prebuilt_transitions, case_folder=case_folder), chunks)

    print("[INFO] CPU parallel generation completed.")

# === GPU VERSION ===
@cuda.jit
def generate_rulesets_kernel(num_transitions, transition_options, out_rulesets, validity_flags):
    idx = cuda.grid(1)
    total_threads = cuda.gridsize(1)

    # Calculate the transition choice vector for this idx
    choice = idx
    base = transition_options.shape[0]

    ruleset = cuda.local.array(shape=(32, 3), dtype=cp.int32)  # Max 32 transitions

    has_halt = False
    for t in range(num_transitions):
        choice_idx = choice % base
        choice = choice // base
        ruleset[t, 0] = transition_options[choice_idx, 0]
        ruleset[t, 1] = transition_options[choice_idx, 1]
        ruleset[t, 2] = transition_options[choice_idx, 2]
        if ruleset[t, 0] == -1:
            has_halt = True

    if has_halt:
        for t in range(num_transitions):
            out_rulesets[idx, t, 0] = ruleset[t, 0]
            out_rulesets[idx, t, 1] = ruleset[t, 1]
            out_rulesets[idx, t, 2] = ruleset[t, 2]
        validity_flags[idx] = 1
    else:
        validity_flags[idx] = 0

def generate_rulesets_gpu(num_states, num_symbols):
    """Main GPU rule set generation."""
    case_folder = Path(f"rulesets/s{num_states}_k{num_symbols}")
    block_root = case_folder / "blocks"
    index_file = case_folder / "index.jsonl"

    block_root.mkdir(parents=True, exist_ok=True)
    index_file.parent.mkdir(parents=True, exist_ok=True)

    num_transitions = num_states * num_symbols
    transition_options = generate_transition_options(num_states, num_symbols)
    num_transition_options = len(transition_options)

    total_combinations = num_transition_options ** num_transitions
    print(f"[INFO] Preparing to generate {total_combinations:,} rule sets on GPU...")

    transition_arr = []
    for option in transition_options:
        if option is None:
            transition_arr.append([-1, 0, -1])
        else:
            transition_arr.append(list(option))

    transition_arr = cp.asarray(transition_arr, dtype=cp.int32)

    num_threads = total_combinations
    threads_per_block = 256
    blocks_per_grid = (num_threads + (threads_per_block - 1)) // threads_per_block

    print(f"[INFO] Launching {blocks_per_grid} blocks of {threads_per_block} threads each.")

    out_rulesets = cp.full((num_threads, num_transitions, 3), -2, dtype=cp.int32)
    validity_flags = cp.zeros(num_threads, dtype=cp.int32)

    generate_rulesets_kernel[blocks_per_grid, threads_per_block](
        num_transitions, transition_arr, out_rulesets, validity_flags
    )

    # Transfer results back to CPU
    out_rulesets_cpu = cp.asnumpy(out_rulesets)
    validity_flags_cpu = cp.asnumpy(validity_flags)

    print(f"[INFO] Filtering valid machines...")
    valid_indices = np.nonzero(validity_flags_cpu)[0]
    print(f"[INFO] Found {len(valid_indices):,} valid machines.")

    # Save valid rulesets
    seen_hashes = {}
    for idx, machine_idx in enumerate(valid_indices):
        rules = out_rulesets_cpu[machine_idx].tolist()
        rules = [r for r in rules if r[0] != -2]  # Clean padding

        ruleset_hash = hash_ruleset(rules)
        is_canonical = ruleset_hash not in seen_hashes

        if is_canonical:
            save_block(block_root, ruleset_hash, rules)
            seen_hashes[ruleset_hash] = f"TM_{idx:06d}"

        machine_entry = {
            "machine_id": f"TM_{idx:06d}",
            "ruleset_hash": ruleset_hash,
            "is_canonical": is_canonical,
            "timestamp": datetime.now().isoformat()
        }
        save_machine_index(index_file, machine_entry)

    print(f"[INFO] GPU rule generation completed.")

# === CLI DRIVER ===
def main():
    parser = argparse.ArgumentParser(description="Parallel Busy Beaver RuleSet Generator (CPU or GPU)")

    parser.add_argument("--states", type=int, default=3, help="Number of machine states (default=3)")
    parser.add_argument("--symbols", type=int, default=2, help="Number of tape symbols (default=2)")
    parser.add_argument("--cpu_cores", type=int, help="Number of CPU cores to use (mutually exclusive with GPU)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU parallel generation (mutually exclusive with CPU cores)")

    args = parser.parse_args()

    # === Safety Checks ===
    if args.cpu_cores and args.gpu:
        raise ValueError("You cannot specify both --cpu_cores and --gpu. Choose one.")

    if args.gpu:
        generate_rulesets_gpu(args.states, args.symbols)
    else:
        num_workers = args.cpu_cores if args.cpu_cores else 1
        generate_rulesets_cpu(args.states, args.symbols, num_workers)

if __name__ == "__main__":
    main()
