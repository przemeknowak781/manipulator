"""Randomizacja dziedziny: jak bardzo symulacja moze sie roznic od ramienia na biurku.

Zakresy to mnozniki wokol wartosci NOMINALNYCH modelu. Nominal nie musi byc
modelem z MuJoCo Menagerie: `Workspace.dynamics` trzyma wynik identyfikacji
serw zmierzony na prawdziwym ramieniu (`rl.sysid`), a `Randomization.around`
stawia zakresy wokol niego - wtedy randomizacja pokrywa niepewnosc pomiaru,
a nie zgadywanie. To jest "kalibracja treningu": polityka uczy sie na
rozrzucie, w ktorym na pewno lezy prawdziwe ramie.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np


@dataclass
class Dynamics:
    """Zidentyfikowana dynamika ramienia - mnozniki wzgledem modelu Menagerie."""

    kp: float = 1.0                 # wzmocnienie pozycyjne serw
    damping: float = 1.0            # tlumienie w stawach
    armature: float = 1.0           # bezwladnosc wirnika
    frictionloss: float = 1.0       # tarcie suche
    #: Opoznienie od rozkazu do ruchu serwa [takty polityki].
    delay: float = 0.0
    #: Skad te liczby: "menagerie" (nominal) albo opis pomiaru.
    source: str = "menagerie"
    #: Blad dopasowania identyfikacji [st.], jesli byla.
    fit_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any] | None) -> Dynamics:
        return Dynamics(**(data or {}))


@dataclass
class Randomization:
    kp: tuple[float, float] = (0.85, 1.15)
    damping: tuple[float, float] = (0.75, 1.3)
    armature: tuple[float, float] = (0.75, 1.3)
    frictionloss: tuple[float, float] = (0.6, 1.5)
    cube_mass: tuple[float, float] = (0.6, 1.6)
    cube_friction: tuple[float, float] = (0.7, 1.3)
    #: Szum pomiaru katow [st.] - enkoder serwa ma 0,088 st. na tik.
    obs_noise_deg: float = 0.3
    #: Opoznienie akcji: losowane z {0 .. max_delay} taktow na epizod.
    max_delay: int = 1
    #: Percepcja kostki tak, jak ja daje `perception` z kamer, a nie fizyka: polozenie
    #: sprzed {0 .. cube_delay} taktow, odswiezane co {1 .. cube_period} taktow, z szumem
    #: [m] i z obrotem zlozonym do +-45 st. (kostka ma symetrie 90 st.). Zmierzone:
    #: polityka uczona na prawdzie z 10 Hz / 150 ms percepcji podnosila 7 z 20 kostek.
    cube_delay: int = 3
    cube_period: int = 3
    cube_noise: float = 0.002
    fold_yaw: bool = True
    #: Srodek zakresow - nominal z identyfikacji, domyslnie model Menagerie.
    centre: Dynamics = field(default_factory=Dynamics)

    @staticmethod
    def none() -> Randomization:
        """Bez randomizacji - do testow i do porownan deterministycznych."""
        one = (1.0, 1.0)
        return Randomization(one, one, one, one, one, one, 0.0, 0, 0, 1, 0.0, False)

    @staticmethod
    def around(dyn: Dynamics, spread: float = 1.0) -> Randomization:
        """Zakresy wokol zmierzonej dynamiki; `spread` skaluje ich szerokosc."""
        base = Randomization()

        def widen(r):
            return (1.0 - (1.0 - r[0]) * spread, 1.0 + (r[1] - 1.0) * spread)

        out = replace(base, kp=widen(base.kp), damping=widen(base.damping), armature=widen(base.armature),
                      frictionloss=widen(base.frictionloss), centre=dyn)
        out.max_delay = int(np.ceil(dyn.delay)) + (1 if spread > 0 else 0)
        return out

    def sample(self, rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
        """Mnozniki dla `n` swiatow (juz przemnozone przez srodek)."""
        c = self.centre
        return {
            "kp": c.kp * rng.uniform(*self.kp, n),
            "damping": c.damping * rng.uniform(*self.damping, n),
            "armature": c.armature * rng.uniform(*self.armature, n),
            "frictionloss": c.frictionloss * rng.uniform(*self.frictionloss, n),
            "cube_mass": rng.uniform(*self.cube_mass, n),
            "cube_friction": rng.uniform(*self.cube_friction, n),
            "delay": rng.integers(0, self.max_delay + 1, n),
            "cube_delay": rng.integers(0, self.cube_delay + 1, n),
            "cube_period": rng.integers(1, max(1, self.cube_period) + 1, n),
        }
