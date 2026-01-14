"""
Utilities for reproducibility and controlled randomness.

This module centralizes all seeding logic for:
- Python random
- NumPy
- PyTorch (CPU + CUDA)
- DataLoader workers

Import and call `seed_everything` once at the experiment entrypoint.
"""

import random
import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """
    Seed all relevant random number generators.

    Parameters
    ----------
    seed : int
        Base random seed.
    deterministic : bool, optional
        If True, enforce deterministic CUDA behavior.
        This may reduce performance and raise errors for unsupported ops.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """
    Initialize DataLoader worker seeds.

    Ensures reproducible randomness when using num_workers > 0.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """
    Create a torch.Generator with a fixed seed.

    Useful for:
    - DataLoader shuffling
    - random_split
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
