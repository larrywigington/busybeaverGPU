# app.py

import argparse
import json
from pathlib import Path

from rich import print
from rich.console import Console
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table

from config.config_loader import load_config
from tools.parallel_ruleset_generator import generate_rulesets_cpu, generate_rulesets_gpu
from tools.simulate_gpu import simulate_gpu
from tools.simulate_pool import simulate_pool

console = Console()

# === Utilities ===
def load_runtime_config():
    config_path = Path("config/runtime_config.json")
    if not config_path.exists():
        console.print("[red]Error: runtime_config.json not found![/red]")
        exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_runtime_config(config):
    config_path = Path("config/runtime_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    console.print("[green]Configuration updated successfully.[/green]")

def show_main_menu():
    console.print("\n[bold cyan]Busy Beaver GPU Solver[/bold cyan]")
    console.print("[1] Generate Rule Sets")
    console.print("[2] Simulate Rule Sets")
    console.print("[3] Simulate Machine Pool")    # <-- New!
    console.print("[4] Inspect/Print Rule Sets")
    console.print("[5] Edit Config")
    console.print("[6] Exit")


def handle_generate(config):
    console.print("\n[bold]Generate Rule Sets[/bold]")

    use_gpu = Confirm.ask("Use GPU for generation?", default=config.get("use_gpu", True))
    states = IntPrompt.ask("Number of States", default=config.get("states", 3))
    symbols = IntPrompt.ask("Number of Symbols", default=config.get("symbols", 2))

    if use_gpu:
        console.print(f"[cyan]Launching GPU generation for {states} states, {symbols} symbols...[/cyan]")
        generate_rulesets_gpu(states, symbols)
    else:
        cpu_cores = IntPrompt.ask("Number of CPU Cores", default=config.get("cpu_cores", 4))
        console.print(f"[cyan]Launching CPU generation with {cpu_cores} cores...[/cyan]")
        generate_rulesets_cpu(states, symbols, cpu_cores)

    # After generation, auto-create a pool
    pool_dir = Path("pools")
    pool_dir.mkdir(exist_ok=True)

    case_name = f"s{states}_k{symbols}"
    index_file = Path("rulesets") / case_name / "index.jsonl"
    pool_file = pool_dir / f"{case_name}.txt"

    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f_in, open(pool_file, "w", encoding="utf-8") as f_out:
            for line in f_in:
                entry = json.loads(line)
                f_out.write(entry["machine_id"] + "\n")

        console.print(f"[green]Auto-created machine pool at {pool_file}[/green]")
    else:
        console.print(f"[red]Warning: Index file not found, pool not created automatically.[/red]")


def handle_simulate(config):
    console.print("\n[bold]Simulate Rule Sets[/bold]")

    states = IntPrompt.ask("States of case to simulate", default=config.get("states", 3))
    symbols = IntPrompt.ask("Symbols of case to simulate", default=config.get("symbols", 2))
    batch_size = IntPrompt.ask("Batch Size", default=config.get("batch_size", 4096))
    max_steps = IntPrompt.ask("Max Steps", default=config.get("max_steps", 1000000))
    tape_size = IntPrompt.ask("Tape Size", default=config.get("tape_size", 512))

    case_folder = f"s{states}_k{symbols}"

    console.print(f"[cyan]Simulating case {case_folder}...[/cyan]")
    simulate_gpu(case_folder, batch_size=batch_size, max_steps=max_steps, tape_size=tape_size)
    console.print("[green]Simulation completed![/green]")

def handle_inspect():
    console.print("\n[bold yellow]Rule Set Inspection coming soon![/bold yellow]")
    console.print("[gray](You can run tools/ruleset_inspect.py manually for now.)[/gray]")

def handle_edit_config(config):
    console.print("\n[bold]Edit Configuration[/bold]")

    states = IntPrompt.ask("Number of States", default=config.get("states", 3))
    symbols = IntPrompt.ask("Number of Symbols", default=config.get("symbols", 2))
    use_gpu = Confirm.ask("Use GPU?", default=config.get("use_gpu", True))
    cpu_cores = IntPrompt.ask("Number of CPU Cores", default=config.get("cpu_cores", 4))
    batch_size = IntPrompt.ask("Batch Size", default=config.get("batch_size", 4096))
    max_steps = IntPrompt.ask("Max Steps", default=config.get("max_steps", 1000000))
    tape_size = IntPrompt.ask("Tape Size", default=config.get("tape_size", 512))


    config.update({
        "states": states,
        "symbols": symbols,
        "use_gpu": use_gpu,
        "cpu_cores": cpu_cores,
        "batch_size": batch_size,
        "max_steps": max_steps,
        "tape_size": tape_size
    })

    save_runtime_config(config)

def handle_simulate_pool():
    console.print("\n[bold]Simulate Machine Pool[/bold]")

    pools = detect_pools()

    if not pools:
        console.print("[red]No pools found. Please create a pool first.[/red]")
        return

    console.print("\n[bold cyan]Available Pools:[/bold cyan]\n")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Index", justify="center")
    table.add_column("Pool Name", justify="center")
    table.add_column("Status", justify="center")

    for idx, (pool_name, status) in enumerate(pools):
        color = {
            "Available": "cyan",
            "In Progress": "yellow",
            "Completed": "green"
        }.get(status, "white")
        table.add_row(str(idx), pool_name, f"[{color}]{status}[/{color}]")

    console.print(table)

    idx_choice = IntPrompt.ask("\nChoose a pool by Index")
    if idx_choice < 0 or idx_choice >= len(pools):
        console.print("[red]Invalid choice. Exiting.[/red]")
        return

    selected_pool, _ = pools[idx_choice]

    batch_size = IntPrompt.ask("Batch Size", default=4096)
    max_steps = IntPrompt.ask("Max Steps", default=1000000)
    tape_size = IntPrompt.ask("Tape Size", default=512)
    use_gpu = Confirm.ask("Use GPU for simulation?", default=False)

    pool_path = Path("pools") / f"{selected_pool}.txt"

    case_folder = Prompt.ask("Case folder (e.g., s2_k2)", default="s2_k2")  # Could improve to detect automatically later

    console.print(f"[cyan]Simulating pool {selected_pool}...[/cyan]")
    simulate_pool(
        str(pool_path),
        case_folder,
        "results",
        batch_size=batch_size,
        max_steps=max_steps,
        tape_size=tape_size,
        use_gpu=use_gpu
    )

    console.print("[green]Machine pool simulation completed![/green]")

def detect_pools():
    pool_dir = Path("pools")
    results_dir = Path("results")

    if not pool_dir.exists():
        pool_dir.mkdir(parents=True, exist_ok=True)

    pools = []
    for pool_file in pool_dir.glob("*.txt"):
        pool_name = pool_file.stem
        checkpoint = results_dir / pool_name / "results_checkpoint.json"
        results_file = results_dir / pool_name / "results.jsonl"

        if not checkpoint.exists() and not results_file.exists():
            status = "Available"
        elif checkpoint.exists():
            with open(checkpoint, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            completed_machines = checkpoint_data.get("completed", [])
            machine_list = load_machine_pool(pool_file)
            if len(completed_machines) >= len(machine_list):
                status = "Completed"
            else:
                status = "In Progress"
        else:
            status = "In Progress"

        pools.append((pool_name, status))

    return pools

def load_machine_pool(pool_file):
    with open(pool_file, "r", encoding="utf-8") as f:
        machines = [line.strip() for line in f if line.strip()]
    return machines


def interactive_main():
    config = load_runtime_config()

    while True:
        show_main_menu()
        choice = Prompt.ask("\nChoose an option", choices=["1", "2", "3", "4", "5", "6"], default="6")

        if choice == "1":
            handle_generate(config)
        elif choice == "2":
            handle_simulate(config)
        elif choice == "3":
            handle_simulate_pool()
        elif choice == "4":
            handle_inspect()
        elif choice == "5":
            handle_edit_config(config)
            config = load_runtime_config()
        elif choice == "6":
            console.print("[bold green]Goodbye![/bold green]")
            break

# === CLI Mode for Automation ===
def cli_main(args):
    config = load_runtime_config()

    if args.generate:
        if args.gpu:
            generate_rulesets_gpu(config["states"], config["symbols"])
        else:
            generate_rulesets_cpu(config["states"], config["symbols"], config.get("cpu_cores", 1))
    if args.simulate:
        case_folder = f"s{config['states']}_k{config['symbols']}"
        simulate_gpu(case_folder, batch_size=config.get("batch_size", 4096), max_steps=config.get("max_steps", 1000000), tape_size=config.get("tape_size", 512))

def main():
    parser = argparse.ArgumentParser(description="Busy Beaver GPU Solver Application")
    parser.add_argument("--generate", action="store_true", help="Generate rule sets immediately")
    parser.add_argument("--simulate", action="store_true", help="Simulate rule sets immediately")
    parser.add_argument("--gpu", action="store_true", help="Force GPU mode (only for generation)")
    args = parser.parse_args()

    if args.generate or args.simulate:
        cli_main(args)
    else:
        interactive_main()

if __name__ == "__main__":
    main()
