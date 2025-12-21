import socket
import time
import json
import os
import struct
from datetime import datetime

from json_utils import extract_json_from_buffer
from csv_utils import init_csv, append_rows

# =========================
# Configuration
# =========================

HOST = "127.0.0.1"
PORT = 12000

RECV_WINDOW_SEC = 0.5
RECV_CHUNK_SIZE = 65536
SLEEP_SLICE = 0.02

# Directories
base_dir = os.getcwd()
log_dir = os.path.join(os.path.dirname(base_dir), "log")
err_path = os.path.join(log_dir, "orchestrator_errors.log")


# =========================
# Logging
# =========================

def log_error(msg: str):
    os.makedirs(log_dir, exist_ok=True)
    with open(err_path, "a") as ef:
        ef.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")


# =========================
# Utilities
# =========================

def load_config(path="simu-config.json"):
    with open(path, "r") as f:
        cfg = json.load(f)
    print(f"[Config] Loaded: {cfg}")
    return cfg


def safe_close_socket(sock: socket.socket | None):
    """Ensure TCP socket is fully released (no PID hang)."""
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    finally:
        try:
            sock.close()
        except OSError:
            pass


# =========================
# Socket Communication
# =========================

def connect() -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Force immediate socket close (avoid TIME_WAIT / CLOSE_WAIT)
    s.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_LINGER,
        struct.pack("ii", 1, 0)
    )

    s.settimeout(0.2)
    s.connect((HOST, PORT))

    try:
        banner = s.recv(1024).decode(errors="ignore").strip()
        if banner:
            print("[BlueSky]", banner)
    except Exception:
        pass

    return s


def send(sock: socket.socket, cmd: str):
    sock.sendall((cmd + "\n").encode("utf-8"))


def recv_for_window(sock: socket.socket, window_sec=RECV_WINDOW_SEC) -> str:
    """Receive data for a limited time window."""
    end_t = time.time() + window_sec
    chunks = []

    while time.time() < end_t:
        try:
            data = sock.recv(RECV_CHUNK_SIZE)
            if not data:
                break
            chunks.append(data.decode(errors="ignore"))
            if len(data) < RECV_CHUNK_SIZE:
                time.sleep(SLEEP_SLICE)
        except (socket.timeout, BlockingIOError):
            time.sleep(SLEEP_SLICE)

    return "".join(chunks)


# =========================
# Data Processing
# =========================

def convert_json_to_csv(objs, csv_path, step, buffer, chunk):
    parsed = 0

    for obj in objs:
        try:
            data = json.loads(obj)
            simt = data.get("simt", 0.0)
            utc = data.get("utc", "")

            aircraft_data = []
            for ac in data.get("aircraft", []):
                ac["simt"] = simt
                ac["utc"] = utc
                aircraft_data.append(ac)

            if aircraft_data:
                append_rows(csv_path, step, aircraft_data)
                parsed += 1

        except json.JSONDecodeError as e:
            log_error(f"JSON decode error at step {step}: {e}")
        except Exception as e:
            log_error(f"Unexpected parse error at step {step}: {e}")

    if parsed == 0 and ("{" not in buffer):
        frag = (chunk or buffer)[:200].replace("\n", " ")
        log_error(f"No JSON parsed at step {step} | head: {frag}")


# =========================
# Main Orchestrator
# =========================

def main():
    cfg = load_config()

    sim_time = cfg.get("simulation_time_sec", 240)
    dt = cfg.get("incremental_time_sec", 5)
    scen = cfg.get("scenario_file_name", "DEMO/demo-scenario.scn")

    sock = None

    try:
        print("[Orchestrator] Connecting to BlueSky...")
        sock = connect()

        send(sock, "EXT ON"); recv_for_window(sock, 0.2)
        send(sock, f"IC {scen}"); recv_for_window(sock, 0.4)
        send(sock, "OP"); recv_for_window(sock, 0.2)

        steps = int(sim_time / dt)
        print(f"[Orchestrator] Running {steps} steps of {dt}s")

        csv_path = init_csv(log_dir)
        buffer = ""

        for step in range(1, steps + 1):
            send(sock, f"STEP {dt}")
            recv_for_window(sock, 0.15)

            send(sock, "STATE")
            chunk = recv_for_window(sock, RECV_WINDOW_SEC)

            if chunk:
                buffer += chunk

            objs, buffer = extract_json_from_buffer(buffer)
            convert_json_to_csv(objs, csv_path, step, buffer, chunk)

            print(f"[Orchestrator] Step {step}/{steps}")
            time.sleep(0.15)

        # Graceful stop
        try:
            send(sock, "HOLD")
            recv_for_window(sock, 0.2)
        except Exception:
            pass

        print(f"[Orchestrator] Done.")
        print(f"[Orchestrator] CSV: {csv_path}")
        print(f"[Orchestrator] Errors: {err_path}")

    except Exception as e:
        log_error(f"Fatal error: {e}")
        raise

    finally:
        safe_close_socket(sock)
        print("[Orchestrator] Socket closed cleanly.")


# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    main()
