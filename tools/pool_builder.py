# tools/pool_builder.py

import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table

console = Console()

def load_case_machines(case_folder):
    """Load all machine IDs from a case."""
    index_path = Path("rulesets") / case_folder / "index.jsonl"
    if not index_path.exists():
        console.print(f"[red]Index file not found: {index_path}[/red]")
        exit(1)

    machines = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            machines.append(entry["machine_id"])

    return machines

def build_pool(case_folder, output_path):
    machines = load_case_machines(case_folder)
    console.print(f"[green]Loaded {len(machines):,} machines from {case_folder}.[/green]")

    selected_machines = []

    while True:
        console.print("\n[bold cyan]Machine Pool Builder Menu[/bold cyan]")
        console.print("[1] List some machines")
        console.print("[2] Add a machine by ID")
        console.print("[3] Add all machines")
        console.print("[4] Save and exit")
        console.print("[5] Cancel and exit without saving")

        choice = Prompt.ask("\nChoose an option", choices=["1", "2", "3", "4", "5"], default="5")

        if choice == "1":
            table = Table(title="Available Machines (sample 10)")
            table.add_column("Machine ID")
            for mid in machines[:10]:
                table.add_row(mid)
            console.print(table)

        elif choice == "2":
            mid = Prompt.ask("Enter machine ID exactly (e.g., TM_000123)")
            if mid in machines:
                if mid not in selected_machines:
                    selected_machines.append(mid)
                    console.print(f"[green]Added {mid}.[/green]")
                else:
                    console.print(f"[yellow]{mid} already in pool.[/yellow]")
            else:
                console.print(f"[red]Machine ID {mid} not found![/red]")

        elif choice == "3":
            confirm = Confirm.ask("Are you sure you want to add ALL machines?", default=False)
            if confirm:
                selected_machines.extend(machines)
                selected_machines = list(set(selected_machines))  # Remove duplicates
                console.print(f"[green]Added all {len(machines):,} machines.[/green]")

        elif choice == "4":
            if selected_machines:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    for mid in selected_machines:
                        f.write(mid + "\n")
                console.print(f"[green]Saved pool with {len(selected_machines):,} machines to {output_path}.[/green]")
            else:
                console.print("[yellow]No machines selected, nothing saved.[/yellow]")
            break

        elif choice == "5":
            console.print("[red]Canceled. No pool saved.[/red]")
            break

def main():
    parser = argparse.ArgumentParser(description="Interactive Pool Builder for Busy Beaver Machines")
    parser.add_argument("--case", required=True, help="Case folder (e.g., s2_k2)")
    parser.add_argument("--output", required=True, help="Path to save pool file (e.g., pools/custom_pool.txt)")
    args = parser.parse_args()

    build_pool(args.case, args.output)

if __name__ == "__main__":
    main()
