import socket
import time
import json
import os

HOST = "127.0.0.1"
PORT = 12000  # must match the orchestrator socket configured in simulation.py


def load_config(path="simu-config.json"):
    """Load simulation parameters from a JSON configuration file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = json.load(f)
    print(f"[Config] Loaded: {cfg}")
    return cfg


def connect_to_sim():
    """Establish a persistent connection with the BlueSky orchestrator socket."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    banner = s.recv(1024).decode(errors="ignore").strip()
    print("[BlueSky]", banner)
    return s


def send(sock, cmd, wait=0.2):
    """Send a command to BlueSky and return the response."""
    sock.sendall((cmd + "\n").encode("utf-8"))
    time.sleep(wait)
    try:
        data = sock.recv(4096)
        resp = data.decode(errors="ignore").strip()
    except Exception:
        resp = ""
    print(f"[BlueSky] {resp}")
    return resp


def main():
    # --- Load configuration parameters ---
    cfg = load_config()
    sim_time = cfg.get("simulation_time_sec", 60)
    dt = cfg.get("incremental_time_sec", 5)
    scen = cfg.get("scenario_file_name", "data/scenario/DEMO/demo-scenario.scn")

    print("[Orchestrator] Connecting to BlueSky orchestrator socket...")
    sock = connect_to_sim()
    time.sleep(1)

    # Enable external control mode (simulation time is advanced manually)
    send(sock, "EXT ON")

    # Load the selected scenario
    send(sock, f"IC {scen}")

    # Start simulation
    send(sock, "OP")

    # Calculate the number of simulation steps
    steps = int(sim_time / dt)
    print(f"[Orchestrator] Running {steps} steps of {dt}s (total {sim_time}s).")

    # Run the simulation in fixed time steps
    for i in range(steps):
        send(sock, f"STEP {dt}")
        print(f"[Orchestrator] Step {i+1}/{steps} executed.")
        time.sleep(0.5)

    # Pause and check final status
    send(sock, "HOLD")
    send(sock, "STATUS")

    sock.close()
    print("[Orchestrator] Done.")


if __name__ == "__main__":
    main()
