"""Wspolny interfejs backendow robota.

Cala reszta aplikacji rozmawia z robotem wylacznie przez ten interfejs, wiec
symulator i prawdziwe ramie sa w pelni wymienne - ten sam kod sterowania
dziala w obu przypadkach.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RobotInfo:
    """Opis podlaczonego robota (do HUD i logow)."""

    name: str
    description: str = ""
    simulated: bool = True


class RobotBackend(ABC):
    """Minimalny interfejs: polacz, czytaj pozycje, wysylaj pozycje, rozlacz."""

    info: RobotInfo

    @abstractmethod
    def connect(self) -> None:
        """Nawiazuje polaczenie (dla symulatora - inicjalizuje stan)."""

    @abstractmethod
    def disconnect(self) -> None:
        """Zamyka polaczenie; powinno byc bezpieczne do wielokrotnego wywolania."""

    @abstractmethod
    def read_joints(self) -> dict[str, float]:
        """Zwraca aktualne pozycje stawow: {nazwa_stawu: wartosc}."""

    @abstractmethod
    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        """Wysyla zadane pozycje stawow. Zwraca to, co faktycznie poszlo do robota."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        ...

    def step(self, dt: float) -> None:  # noqa: B027 - domyslnie nic nie robi
        """Krok symulacji. Prawdziwy robot ignoruje (rzeczywistosc liczy sie sama)."""

    def __enter__(self) -> "RobotBackend":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.disconnect()
