import json
import os
import hashlib
from itertools import product
from datetime import datetime
from pathlib import Path
import argparse

# === CONFIGURABLE ===
BLOCK_ROOT = Path("rulesets/blocks")
INDEX_FILE = Path("rulesets/index.jsonl")


# === TRANSITION OPTION MAP ===
def generate_transition_options(num_states, num_symbols):
    """Build list of all possible transitions including halt (-1)."""
    options = []
    for new_symbol in range(num_symbols):
        for direction in ['L', 'R']:
            for new_state in range(num_states):
                dir_bit = 0 if direction == 'L' else 1
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
            json.dump(rules, f, indent=2)
    return str(block_path)

def save_machine_index(index_file, machine_entry):
    """Append a machine entry to the index file."""
    index_file.parent.mkdir(parents=True, exist_ok=True)
    with open(index_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(machine_entry) + "\n")

def generate_all_rulesets(num_states=3, num_symbols=2):
    """Generate all combinatorial rulesets, organized by (states, symbols) case."""

    # === Setup Folder Structure ===
    case_folder = Path(f"rulesets/s{num_states}_k{num_symbols}")
    block_root = case_folder / "blocks"
    index_file = case_folder / "index.jsonl"

    block_root.mkdir(parents=True, exist_ok=True)
    index_file.parent.mkdir(parents=True, exist_ok=True)

    # === Compute Options ===
    num_transitions = num_states * num_symbols
    transition_options = generate_transition_options(num_states, num_symbols)

    # Pre-build serialized transitions (lists)
    prebuilt_transitions = []
    for option in transition_options:
        if option is None:
            prebuilt_transitions.append([-1, 0, -1])  # Halting
        else:
            prebuilt_transitions.append(list(option))

    num_transition_options = len(transition_options)

    seen_hashes = dict()
    machine_counter = 0
    skipped_counter = 0

    total_combinations = num_transition_options ** num_transitions
    print(f"[INFO] Preparing to generate {total_combinations:,} possible rule sets...")

    # === Main Generation Loop ===
    for transition_choice in product(range(num_transition_options), repeat=num_transitions):
        rules = [prebuilt_transitions[choice_idx] for choice_idx in transition_choice]

        # === New Halt Check ===
        has_halt = any(transition == [-1, 0, -1] for transition in rules)
        if not has_halt:
            skipped_counter += 1
            continue  # Skip machines with no halting transitions

        ruleset_hash = hash_ruleset(rules)

        is_canonical = ruleset_hash not in seen_hashes
        if is_canonical:
            save_block(block_root, ruleset_hash, rules)
            seen_hashes[ruleset_hash] = f"TM_{machine_counter:06d}"

        machine_entry = {
            "machine_id": f"TM_{machine_counter:06d}",
            "states": num_states,
            "symbols": num_symbols,
            "ruleset_hash": ruleset_hash,
            "is_canonical": is_canonical,
            "timestamp": datetime.now().isoformat()
        }
        save_machine_index(index_file, machine_entry)

        machine_counter += 1

        if machine_counter % 10000 == 0:
            print(f"[INFO] Generated {machine_counter:,} machines so far...")

    print(f"[INFO] Finished generating {machine_counter:,} machines.")
    print(f"[INFO] Skipped {skipped_counter:,} rule sets without any halting transitions.")


# === CLI WRAPPER ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Busy Beaver RuleSet Generator (Hash-Based Storage)")

    parser.add_argument("--states", type=int, default=3,
                        help="Number of machine states (default=3)")
    parser.add_argument("--symbols", type=int, default=2,
                        help="Number of tape symbols (default=2)")

    args = parser.parse_args()

    generate_all_rulesets(num_states=args.states, num_symbols=args.symbols)
