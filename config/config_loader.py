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

# Expected types for validation
CONFIG_SCHEMA = {
    "max_steps": int,
    "state_size": int,
    "num_symbols": int,
    "log_frequency": int,
    "save_partial_results": bool,
    "enable_checkpointing": bool,
    "checkpoint_interval": int,
    "rl_guided_mode": bool,
    "rl_model_path": str,
    "throttle_enabled": bool,
    "throttle_schedule": dict,
    "output_directory": str,
    "log_file_prefix": str,
    "pending_file": str,
    "skipped_file": str
}

def validate_config(config):
    for key, expected_type in CONFIG_SCHEMA.items():
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
        if not isinstance(config[key], expected_type):
            raise TypeError(f"Config key '{key}' expected {expected_type}, got {type(config[key])}.")

    # Special check inside throttle_schedule
    throttle = config["throttle_schedule"]
    if not all(k in throttle for k in ["start_hour", "end_hour", "days_active"]):
        raise ValueError("Throttle schedule must contain 'start_hour', 'end_hour', and 'days_active'.")

def load_config(path="config/runtime_config.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r") as f:
        user_config = json.load(f)

    # Merge defaults with overrides
    config = DEFAULT_CONFIG.copy()
    config.update(user_config)

    # Validate schema
    validate_config(config)

    # Validate output directory
    os.makedirs(config["output_directory"], exist_ok=True)

    # Print config summary (optional)
    print(f"[{datetime.now()}] Loaded config:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    return config
