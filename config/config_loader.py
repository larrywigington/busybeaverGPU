import json
import os
from datetime import datetime

DEFAULT_CONFIG = {
    "max_steps": 1_000_000,
    "state_size": 3,
    "num_symbols": 2,
    "log_frequency": 100,
    "save_partial_results": True,
    "enable_checkpointing": True,
    "checkpoint_interval": 3600,
    "rl_guided_mode": False,
    "rl_model_path": "heuristics/model.pkl",
    "throttle_enabled": True,
    "throttle_schedule": {
        "start_hour": 9,
        "end_hour": 18,
        "days_active": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    },
    "output_directory": "logs/",
    "log_file_prefix": "busybeaver_",
    "pending_file": "logs/pending.json",
    "skipped_file": "logs/skipped.json"
}

def load_config(path="config/runtime_config.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r") as f:
        user_config = json.load(f)

    # Merge defaults with overrides
    config = DEFAULT_CONFIG.copy()
    config.update(user_config)

    # Validate output directory
    os.makedirs(config["output_directory"], exist_ok=True)

    # Print config summary (optional)
    print(f"[{datetime.now()}] Loaded config:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    return config
