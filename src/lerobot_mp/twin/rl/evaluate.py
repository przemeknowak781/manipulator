"""Ewaluacja polityki na CPU - w zwyklym MuJoCo, nie w tym, w ktorym sie uczyla.

Polityka uczona w MuJoCo Warp (float32, GPU) jedzie tu w MuJoCo na CPU
(float64), z randomizacja albo bez. To najtanszy test przenoszenia: jesli
polityka nie przezywa zmiany silnika fizyki, na pewno nie przezyje zmiany
na prawdziwe ramie.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..workspace import Workspace
from .env import TwinEnv
from .policy import Policy
from .randomize import Dynamics, Randomization


def evaluate(policy: Policy, episodes: int = 50, *, randomization: Randomization | None = None,
             workspace: Workspace | None = None, seed: int = 10_000, on_frame=None) -> dict[str, Any]:
    """`randomization=None` - ramie nominalne: dynamika z `workspace.dynamics` (identyfikacja), bez rozrzutu.

    Wczesniej None znaczylo model Menagerie, wiec "CPU bez randomizacji" po identyfikacji
    nie mowilo nic o zmierzonym ramieniu. W wyniku `randomized` mowi, czy dynamika byla
    losowana naprawde, a `centre` - wokol czego.
    """
    task = policy.task
    rand = randomization if randomization is not None else \
        Randomization.nominal(Dynamics.from_dict(workspace.dynamics) if workspace is not None else None)
    env = TwinEnv(task, workspace=workspace, randomization=rand, render_mode="rgb_array" if on_frame else None)
    succ, final, ever = [], [], []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            hit = False
            for _ in range(task.episode_steps):
                obs, _, term, trunc, info = env.step(policy.act(obs))
                hit |= info["success"]
                if on_frame is not None:
                    on_frame(env.render())
                if term or trunc:
                    break
            succ.append(float(info["success"]))
            ever.append(float(hit))
            final.append(info["distance"] if task.name == "reach" else info["height"])
    finally:
        env.close()
    out = {"episodes": episodes, "success": float(np.mean(succ)), "ever_success": float(np.mean(ever)),
           "randomized": rand.randomized, "centre": rand.centre.source}
    key = "final_distance_mm" if task.name == "reach" else "final_height_mm"
    out[key] = float(np.median(final) * 1000)
    return out
