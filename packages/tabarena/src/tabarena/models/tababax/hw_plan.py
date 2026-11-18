"""Hardware budget for TabAbax context sampling.

The eval script publishes the official machine, not the host's real disk and RAM,
through three environment variables. Sampling decisions read those first so an H20
with 1 TB of RAM still behaves like the 96 GB GPU / 180 GB RAM / 128 GB disk box.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# fp32 column-embedding width used by TabAbax / TabICL: one token is 128 floats.
BYTES_PER_TOKEN = 128 * 4
CLS_TOKENS = 4

# The number below is the one fp32 buffer, not process RSS.
# On the official box a ~94 GiB buffer already showed ~159 GiB RSS, and the host
# guard rejects anon_rss + this allocation above the RAM cap. Leaving 45% of RAM
# for that resident overhead keeps the buffer near the size that has actually fit.
# Disk is the fallback placement, so the buffer also has to fit in the disk budget.
# A flat 200k-row cap is not used: maps at 200k rows is still ~102 GiB, which is
# the same edge that killed the uncapped 149–595 GiB allocations.
GPU_TENSOR_FRAC = 0.50
RAM_TENSOR_FRAC = 0.55
DISK_TENSOR_FRAC = 0.85
# Training must not consume the whole buffer: predict concatenates context and query.
TRAIN_SHARE = 0.80


@dataclass(frozen=True)
class HardwareBudget:
    gpu_bytes: int
    ram_bytes: int
    disk_bytes: int
    capacity_bytes: int
    train_budget_bytes: int
    source: str

    @property
    def query_floor_bytes(self) -> int:
        return self.capacity_bytes - self.train_budget_bytes

    @staticmethod
    def from_totals(
        gpu_bytes: int,
        ram_bytes: int,
        disk_bytes: int,
        source: str,
    ) -> "HardwareBudget":
        ram_room = _usable(ram_bytes, RAM_TENSOR_FRAC)
        disk_room = _usable(disk_bytes, DISK_TENSOR_FRAC)
        gpu_room = _usable(gpu_bytes, GPU_TENSOR_FRAC)
        host = [room for room in (ram_room, disk_room) if room > 0]
        # RAM and disk must both be able to hold the buffer: the library prefers
        # CPU, and the guard aborts that placement instead of quietly moving it to disk.
        host_room = min(host) if host else 0
        capacity = max(host_room, gpu_room, BYTES_PER_TOKEN)
        return HardwareBudget(
            gpu_bytes=gpu_bytes,
            ram_bytes=ram_bytes,
            disk_bytes=disk_bytes,
            capacity_bytes=capacity,
            train_budget_bytes=max(BYTES_PER_TOKEN, int(capacity * TRAIN_SHARE)),
            source=source,
        )


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _machine_gpu_bytes() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        return 0
    return 0


def _machine_ram_bytes() -> int:
    """Physical RAM, or the cgroup limit when that is tighter (containers)."""
    total = 0
    try:
        import psutil

        total = int(psutil.virtual_memory().total)
    except Exception:
        total = 0
    limit = _cgroup_memory_max()
    if limit is not None and (total <= 0 or limit < total):
        return limit
    return total


def _cgroup_memory_max() -> int | None:
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            text = open(path, encoding="ascii").read().strip()
        except OSError:
            continue
        if text.isdigit():
            value = int(text)
            # cgroup v1 uses a near-2^63 sentinel for "unlimited".
            if 0 < value < 1 << 60:
                return value
    return None


def _machine_disk_bytes() -> int:
    """Size of the filesystem that would hold an offload file, not the free space."""
    import shutil
    import tempfile

    directory = os.environ.get("TABARENA_DISK_PATH", "").strip() or tempfile.gettempdir()
    try:
        return int(shutil.disk_usage(directory).total)
    except OSError:
        return int(shutil.disk_usage(tempfile.gettempdir()).total)


def read_hardware() -> HardwareBudget:
    """Env limits win when the eval script publishes them; otherwise read this machine.

    Unset ``TABARENA_GPU_LIMIT_BYTES`` / ``TABARENA_MEMORY_LIMIT_BYTES`` /
    ``TABARENA_DISK_LIMIT_BYTES`` and the same planner uses the physical GPU,
    RAM (or cgroup limit) and the offload filesystem size.
    """
    gpu_env = _env_int("TABARENA_GPU_LIMIT_BYTES")
    ram_env = _env_int("TABARENA_MEMORY_LIMIT_BYTES")
    disk_env = _env_int("TABARENA_DISK_LIMIT_BYTES")
    gpu = _machine_gpu_bytes() if gpu_env is None else gpu_env
    ram = _machine_ram_bytes() if ram_env is None else ram_env
    disk = _machine_disk_bytes() if disk_env is None else disk_env
    used_env = gpu_env is not None or ram_env is not None or disk_env is not None
    source = "env" if used_env and gpu_env is not None and ram_env is not None and disk_env is not None else (
        "env+machine" if used_env else "machine"
    )
    return HardwareBudget.from_totals(int(gpu or 0), int(ram or 0), int(disk or 0), source)


# One entry per fit. The eval script copies this list into task_status.jsonl.
_RUNS: list[dict] = []


def begin_strategy(hardware: HardwareBudget, predicted: dict) -> None:
    """Record the plan chosen from hardware + the post-preprocess table, before fit."""
    _RUNS.append(
        {
            "source": hardware.source,
            "gpu_gib": round(hardware.gpu_bytes / 1024**3, 2),
            "ram_gib": round(hardware.ram_bytes / 1024**3, 2),
            "disk_gib": round(hardware.disk_bytes / 1024**3, 2),
            "capacity_gib": round(hardware.capacity_bytes / 1024**3, 2),
            "train_budget_gib": round(hardware.train_budget_bytes / 1024**3, 2),
            "predicted": predicted,
            "actual": {},
        }
    )


def update_actual(**fields) -> None:
    """Fill in what this fit/predict actually did. Later predict calls overwrite chunk fields."""
    if not _RUNS:
        _RUNS.append({"source": None, "predicted": None, "actual": {}})
    actual = _RUNS[-1]["actual"]
    for key, value in fields.items():
        if isinstance(value, float):
            value = round(value, 4)
        actual[key] = value


def take_strategies() -> list[dict]:
    """Return every strategy recorded since the last take, then clear the buffer."""
    runs = list(_RUNS)
    _RUNS.clear()
    return runs


def _usable(total: int, frac: float) -> int:
    if total <= 0:
        return 0
    return int(total * frac)


def bytes_per_row(n_cols: int) -> int:
    return (int(n_cols) + CLS_TOKENS) * BYTES_PER_TOKEN


def member_row_index(n_rows: int, n_estimators: int, kept_rows: int, seed: int):
    """Kumo's per-member context index, shape ``[n_estimators, kept_rows]``.

    ``None`` when every row already fits. Otherwise each member takes the next slice of
    concatenated random permutations, so rows are spread across the ensemble instead of
    one shared subset. A slice that crosses two permutations may repeat a row.
    """
    import numpy as np

    n_estimators = max(1, int(n_estimators))
    kept_rows = int(kept_rows)
    if kept_rows >= n_rows or kept_rows <= 0:
        return None
    num_repeats = -(-n_estimators * kept_rows // n_rows)
    rng = np.random.default_rng(int(seed))
    perm = np.concatenate([rng.permutation(n_rows) for _ in range(num_repeats)])
    return perm[: n_estimators * kept_rows].reshape(n_estimators, kept_rows)


def plan_context(n_rows: int, n_cols: int, n_estimators: int, budget: HardwareBudget) -> dict:
    """Drop rows first. Drop columns only when even one full-width row exceeds the train budget."""
    n_rows = int(n_rows)
    n_cols = int(n_cols)
    bpr = bytes_per_row(n_cols)
    max_rows = budget.train_budget_bytes // bpr if bpr else n_rows
    if max_rows >= n_rows:
        kept_rows, kept_cols = n_rows, n_cols
        row_sample = col_sample = False
    elif max_rows >= 1:
        kept_rows, kept_cols = int(max_rows), n_cols
        row_sample, col_sample = True, False
    else:
        # One row of every column is already larger than the train budget.
        max_tokens = max(CLS_TOKENS + 1, budget.train_budget_bytes // BYTES_PER_TOKEN)
        kept_cols = max(1, min(n_cols, max_tokens - CLS_TOKENS))
        kept_rows = min(n_rows, max(1, budget.train_budget_bytes // bytes_per_row(kept_cols)))
        row_sample = kept_rows < n_rows
        col_sample = kept_cols < n_cols
    return {
        "row_sample": row_sample,
        "col_sample": col_sample,
        "kept_rows": int(kept_rows),
        "kept_cols": int(kept_cols),
        "bytes_per_row": bytes_per_row(kept_cols),
    }


def plan_query(n_context: int, n_cols: int, n_test: int, budget: HardwareBudget) -> dict:
    """Chunk the test set only when context + the full test set exceeds the one buffer."""
    bpr = bytes_per_row(n_cols)
    train_bytes = int(n_context) * bpr
    slack = budget.capacity_bytes - train_bytes
    n_test = max(0, int(n_test))
    if n_test == 0 or n_test * bpr <= slack:
        return {
            "chunk": False,
            "chunk_rows": n_test,
            "peak_bytes": train_bytes + n_test * bpr,
        }
    chunk_rows = max(1, slack // bpr) if slack > 0 else 1
    return {
        "chunk": True,
        "chunk_rows": int(chunk_rows),
        "peak_bytes": train_bytes + int(chunk_rows) * bpr,
    }


def can_allocate(peak_bytes: int, budget: HardwareBudget) -> bool:
    return int(peak_bytes) <= budget.capacity_bytes
