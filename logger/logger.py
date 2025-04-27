import json
import os
from datetime import datetime, timezone

class JSONLogger:
    def __init__(self, output_directory="logs/", log_file_prefix="busybeaver_"):
        self.output_directory = output_directory
        self.log_file_prefix = log_file_prefix
        os.makedirs(self.output_directory, exist_ok=True)
        self.today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.current_log = self._get_log_filename()

    def _get_log_filename(self):
        filename = f"{self.log_file_prefix}{self.today}.jsonl"  # JSON lines format
        return os.path.join(self.output_directory, filename)

    def _log_to_file(self, filename, entries):
        path = os.path.join(self.output_directory, filename)
        with open(path, "a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def log(self, entry: dict):
        """Log a single entry to the main busybeaver log."""
        with open(self.current_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def log_batch(self, entries: list):
        """Log a batch of entries to the main busybeaver log."""
        with open(self.current_log, "a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def rotate(self):
        """Force start a new main log file."""
        self.current_log = self._get_log_filename()

    def log_summary(self, entries: list):
        """Log summary info (halted or not, steps, etc.)"""
        filename = f"{self.log_file_prefix}{self.today}.jsonl"
        self._log_to_file(filename, entries)

    def log_halting(self, entries: list):
        """Log full rulesets for halting machines."""
        filename = f"halting_{self.today}.jsonl"
        self._log_to_file(filename, entries)

    def log_non_halting(self, entries: list):
        """Log full rulesets for non-halting machines."""
        filename = f"non_halting_{self.today}.jsonl"
        self._log_to_file(filename, entries)
