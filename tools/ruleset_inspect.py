import json
import argparse
from pathlib import Path

def load_machine_index(index_file):
    """Load index.jsonl into a dictionary mapping machine_id -> entry."""
    machine_map = {}
    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            machine_map[entry["machine_id"]] = entry
    return machine_map

def load_ruleset(block_root, ruleset_hash):
    """Load a ruleset JSON file given its hash."""
    block_path = block_root / ruleset_hash[:2] / ruleset_hash[2:4] / f"{ruleset_hash}.json"
    if not block_path.exists():
        raise FileNotFoundError(f"Ruleset block {block_path} not found.")
    with open(block_path, "r", encoding="utf-8") as f:
        return json.load(f)


def pretty_print_ruleset(rules, num_states, num_symbols):
    """Pretty print the ruleset in a state x symbol table with clean Busy Beaver notation."""
    # === Terminal Human-Readable Table ===
    print("\n=== Transition Table ===")
    header = [" "] + [f"{i}" for i in range(num_symbols)]
    print("\t".join(header))

    idx = 0
    transition_table = []

    for state in range(num_states):
        state_letter = chr(ord('A') + state)
        row = [f"State {state_letter}"]
        latex_row = [state_letter]
        for symbol in range(num_symbols):
            transition = rules[idx]
            if transition == [-1, 0, -1]:
                action = "HALT"
                latex_action = "HALT"
            else:
                write_symbol, dir_bit, next_state = transition
                move_dir = "L" if dir_bit == 0 else "R"
                next_state_letter = chr(ord('A') + next_state)
                action = f"{write_symbol}{move_dir}{next_state_letter}"
                latex_action = action  # Directly use compact format
            row.append(action)
            latex_row.append(latex_action)
            idx += 1
        print("\t".join(row))
        transition_table.append(latex_row)

    # === LaTeX Table Output ===
    print("\n=== LaTeX Table ===")
    print(r"\begin{array}{c|" + "c" * num_symbols + "}")
    print("State/Symbol & " + " & ".join([f"\\text{{{i}}}" for i in range(num_symbols)]) + r" \\ \hline")
    for latex_row in transition_table:
        print(" & ".join(latex_row) + r" \\")
    print(r"\end{array}")

def main():
    parser = argparse.ArgumentParser(description="Busy Beaver Ruleset Inspector")
    parser.add_argument("--case", required=True, help="Case folder, e.g., s2_k2")
    parser.add_argument("--machine_id", help="Machine ID to inspect, e.g., TM_000123")
    parser.add_argument("--hash", help="Ruleset hash to inspect directly")
    args = parser.parse_args()

    case_folder = Path("rulesets") / args.case
    block_root = case_folder / "blocks"
    index_file = case_folder / "index.jsonl"

    if args.machine_id:
        machine_map = load_machine_index(index_file)
        if args.machine_id not in machine_map:
            raise ValueError(f"Machine ID {args.machine_id} not found in {index_file}")
        entry = machine_map[args.machine_id]
        ruleset_hash = entry["ruleset_hash"]
        is_canonical = entry["is_canonical"]
        print(f"[INFO] Machine {args.machine_id}")
        print(f"  States: {entry['states']}")
        print(f"  Symbols: {entry['symbols']}")
        print(f"  Ruleset Hash: {ruleset_hash}")
        print(f"  Canonical: {is_canonical}")
    elif args.hash:
        ruleset_hash = args.hash
        print(f"[INFO] Direct hash lookup: {ruleset_hash}")
    else:
        raise ValueError("You must specify either --machine_id or --hash.")

    # Load and pretty-print the ruleset
    rules = load_ruleset(block_root, ruleset_hash)
    print("\n=== Ruleset ===")
    pretty_print_ruleset(rules, entry["states"], entry["symbols"])

if __name__ == "__main__":
    main()
