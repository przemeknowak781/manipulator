"""Uczenie ze wzmocnieniem na blizniaku: zadania, srodowiska CPU i GPU, PPO, polityki.

Import rejestruje srodowiska Gymnasium:

    LeRobotMP/TwinReach-v0   dojazd TCP do punktu (cel losowany z osiagalnych)
    LeRobotMP/TwinLift-v0    chwyt i podniesienie kostki 3 cm
"""

from __future__ import annotations

import gymnasium as gym

for _id, _task in (("LeRobotMP/TwinReach-v0", "reach"), ("LeRobotMP/TwinLift-v0", "lift")):
    if _id not in gym.registry:
        gym.register(id=_id, entry_point="lerobot_mp.twin.rl.env:TwinEnv", kwargs={"task": _task})
