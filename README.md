# 🧠 BusyBeaver-GPU: GPU-Accelerated Search for Busy Beaver Candidates

A Python-based framework for discovering Busy Beaver Turing machines using GPU acceleration and guided heuristics. Built for portability, scalability, and persistent search with optional Docker deployment.

---

## 🚀 Project Goals

- Discover and verify Busy Beaver machines with maximum halting steps or maximum number of 1s on the tape.
- Leverage **Numba/CUDA** or **CuPy** to massively parallelize Turing machine simulations.
- Introduce **reinforcement learning** as a heuristic engine for prioritizing high-potential machine configurations.
- Support background, scheduled, or full-speed execution depending on system load or schedule.

---

## 🧰 Key Features

- **Portable Architecture**: Runs on Windows (via Docker or WSL2), with future scalability to Linux clusters or AWS EC2.
- **GPU Acceleration**: Designed for modern NVIDIA GPUs (tested on RTX 4080 Super).
- **Simulation Engine**: Parallel machine evaluator with halting detection and step limits.
- **Logging System**: JSON-based logs of machine ID, steps, final tape, and output class.
- **Guided Search**: Future implementation of RL model to prioritize promising machines.
- **Manual CLI Control**: Start, pause, resume, or stop the background engine with live progress metrics.
- **Throttling Mode**: Full-speed compute during off-hours, reduced usage during set active hours.
- **Checkpointing & Restarting**: Save current progress to disk and resume from where you left off.

---

## 📁 File Structure (WIP)

```text
busybeaver_gpu/
├── main.py                 # Entry point and CLI
├── simulator/
│   ├── turing_machine.py  # Turing machine data structures
│   ├── simulator_gpu.py   # GPU simulation kernels (Numba/CuPy)
│   └── evaluator.py       # Evaluation logic and halting criteria
├── heuristics/
│   └── rl_model.py        # RL engine for machine ranking (future)
├── logger/
│   └── logger.py          # JSON logging utilities
├── config/
│   └── runtime_config.json # Configurable throttle hours, max steps, etc.
├── dashboard/             # Optional web dashboard UI (future)
└── Dockerfile             # Containerized runtime environment
```

---

## 📊 Planned Dashboard (Optional)

- Live metrics (machines/sec, halting %)
- Top machines (steps, 1s, final state)
- Progress timeline (log of total simulated)
- Toggle for throttle mode and manual override
- Alert/log for non-halting but interesting patterns

---

## 📋 Logging Format (JSON)

Each simulation writes to a daily log file:

```json
{
  "machine_id": "TM_00023814",
  "states": 4,
  "halted": true,
  "steps": 234902,
  "ones_written": 37,
  "tape_checksum": "a94f...",
  "final_state": "q3",
  "timestamp": "2025-04-23T16:35:12Z"
}
```

We also maintain a `pending.json` list of machines not yet evaluated and `skipped.json` for timeouts or exceptions.

---

## 🧪 Getting Started

### System Requirements
- Windows 11, WSL2 enabled OR Docker Desktop
- NVIDIA GPU (Compute Capability 7.5+ recommended)
- Python 3.9+

### Installation
```bash
git clone https://github.com/your-username/busybeaver-gpu.git
cd busybeaver-gpu
docker build -t busybeaver .
```

### Run (Docker)
```bash
docker run --gpus all -v $(pwd):/app busybeaver python main.py
```

### Run (WSL2)
```bash
python main.py
```

---

## 🔧 Configuration

Edit `config/runtime_config.json` to change:
- `max_steps`: cutoff threshold
- `throttle_schedule`: `{ "start_hour": 9, "end_hour": 18 }`
- `log_frequency`: machines per batch
- `rl_guided_mode`: true/false

---

## 🧠 Roadmap

- [x] Enumerate and simulate 3-state machines
- [x] JSON-based result logging
- [ ] Scale to 4-5 states
- [ ] RL-guided heuristics for promising machines
- [ ] Web dashboard for monitoring
- [ ] AWS compatibility for cluster mode

---

## 🧵 License
MIT

---

## 🙌 Contributing
PRs welcome! Start with the `simulator/` or `logger/` modules, and help us scale to n=6 and beyond!

