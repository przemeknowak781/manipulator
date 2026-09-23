"""Geometria szczek SO-101 wzgledem TCP (site gripperframe): gdzie sa czubki przy roznym otwarciu."""

from __future__ import annotations

import numpy as np

from lerobot_mp.twin.kinematics import RobotKinematics, inverse, pose
from lerobot_mp.twin.robots import SO101

kin = RobotKinematics(SO101)
m, d = kin.model, kin.data
tips = [m.geom(n).id for n in SO101.fingertips]
print("TCP axes (site frame): approach=x, closing=z")
for g in (0, 10, 26, 50, 75, 100):
    kin._apply(kin.to_q({**SO101.home, "gripper": g}))
    T_site = pose(d.site_xmat[kin.site_id].reshape(3, 3), d.site_xpos[kin.site_id])
    Ti = inverse(T_site)
    fixed = Ti[:3, :3] @ d.geom_xpos[tips[0]] + Ti[:3, 3]
    moving = Ti[:3, :3] @ d.geom_xpos[tips[1]] + Ti[:3, 3]
    print(f"gripper={g:5.1f}  fixed_tip[mm]={np.round(1000 * fixed, 1)}  moving_tip[mm]={np.round(1000 * moving, 1)}"
          f"  gap={1000 * np.linalg.norm(fixed - moving):.1f}")
