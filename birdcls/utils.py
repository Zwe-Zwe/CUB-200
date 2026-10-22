"""Small helpers: reproducibility, device, GPU safety checks, logging, JSON."""
import contextlib
import json
import logging
import os
import random
import sys

import numpy as np
import torch

from . import config as C


def set_seed(seed=C.SEED):
    """Fix every random number generator we use, so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """Use the GPU when one is available; everything also runs (slowly) on the CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def gpu_memory_report():
    """Free / total GPU memory in GiB, e.g. to spot another process already using the GPU."""
    if not torch.cuda.is_available():
        return "no CUDA GPU"
    free, total = torch.cuda.mem_get_info()          # in bytes; 2**30 bytes = 1 GiB
    return f"{torch.cuda.get_device_name(0)}: {free / 2**30:.2f} GiB free of {total / 2**30:.2f} GiB"


def gpu_is_busy(min_free_fraction=0.7):
    """True if another process already holds a large share of the GPU (training would likely OOM)."""
    if not torch.cuda.is_available():
        return False
    free, total = torch.cuda.mem_get_info()
    return free / total < min_free_fraction


@contextlib.contextmanager
def run_lock(lock_file):
    """Prevent two processes from training the same run at once.

    Writes this process id to lock_file; refuses to start if the file names a process that is still
    alive. A lock left behind by a crashed process (dead PID) is simply replaced.
    """
    if lock_file.exists():
        try:
            pid = int(lock_file.read_text().strip())
            os.kill(pid, 0)                       # signal 0 = "does this process exist?"
            raise RuntimeError(f"{lock_file.parent.name} is already being trained by process {pid}. "
                               "Wait for it to finish (or stop it) before starting another.")
        except (ValueError, ProcessLookupError):
            pass                                  # stale lock from a crashed run
    lock_file.write_text(str(os.getpid()))
    try:
        yield
    finally:
        lock_file.unlink(missing_ok=True)


def get_logger(name, log_file=None):
    """Logger that prints to the console and (optionally) appends to a log file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()                       # avoid duplicate lines when a run is started twice
    logger.propagate = False                      # do not also print through the root logger
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))
    for h in handlers:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def save_json(obj, path):
    """Write a dict (e.g. the training history) as readable JSON."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def load_json(path):
    """Read a JSON file written by save_json."""
    with open(path) as f:
        return json.load(f)
