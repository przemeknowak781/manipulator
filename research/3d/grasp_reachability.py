"""Gdzie SO-101 moze chwycic z gory (podejscie pionowe) i z boku (podejscie poziome, promieniowe)?

Siatka punktow w ukladzie podstawy; IK projektu z zadana orientacja TCP.
Wynik: blad pozycji i orientacji. Uruchamiac interpreterem projektu z -B.
"""

from __future__ import annotations

import json
import time

import numpy as np

from lerobot_mp.twin.kinematics import RobotKinematics
from lerobot_mp.twin.robots import SO101

kin = RobotKinematics(SO101)
print("zakresy [deg]:", {j: (round(float(np.degrees(lo)), 1), round(float(np.degrees(hi)), 1))
                         for j, lo, hi in zip(SO101.joints, kin.lo, kin.hi)})
home_tcp = kin.tcp(SO101.home)
print("TCP w home [m]:", np.round(home_tcp[:3, 3], 3))
print("osie w home: podejscie, zamykanie:", [np.round(a, 2) for a in kin.tool_axes(SO101.home)])


def R_from(approach, closing):
    approach = np.asarray(approach, float) / np.linalg.norm(approach)
    closing = np.asarray(closing, float) / np.linalg.norm(closing)
    return np.column_stack([approach, np.cross(closing, approach), closing])


rows = []
t0 = time.perf_counter()
for r in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
    for z in (0.02, 0.05, 0.10):
        for phi_deg in (0.0, 45.0):
            phi = np.radians(phi_deg)
            radial = np.array([np.cos(phi), np.sin(phi), 0.0])
            p = r * radial + np.array([0, 0, z])
            tang = np.array([-np.sin(phi), np.cos(phi), 0.0])
            res = {}
            # z gory, szczeki zamykaja sie wzdluz promienia albo w poprzek
            for name, (a, c) in {
                "top_close_tangent": ((0, 0, -1), tang),
                "top_close_radial": ((0, 0, -1), radial),
                "side_radial": (radial, (0, 0, 1)),
                "side_radial_roll90": (radial, tang),
                "side_tangent": (tang, (0, 0, 1)),  # podejscie z boku, NIE promieniowe
            }.items():
                s = kin.ik(p, R_from(a, c), seed={**SO101.home, "gripper": 100.0})
                res[name] = (bool(s.ok), round(1000 * s.pos_err, 1), round(float(np.degrees(s.rot_err)), 1))
            rows.append({"r": r, "z": z, "phi": phi_deg, **res})
dt = time.perf_counter() - t0

n_ik = len(rows) * 5
print(f"{n_ik} wywolan IK w {dt:.1f} s -> {1000 * dt / n_ik:.0f} ms/IK")
print("kolumny: (ok, blad pozycji mm, blad obrotu deg)")
for row in rows:
    print(row)
with open("grasp_reachability_out.json", "w") as f:
    json.dump({"ms_per_ik": 1000 * dt / n_ik, "rows": rows}, f, indent=1)
