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
        """Wysyla zadane pozycje stawow. Zwraca to, co faktycznie poszlo do robota.

        Moze zwrocic mniej stawow niz dostal - albo nic, gdy backend wstrzymal
        wysylke (np. `feetech` przy zerwanym laczu, zeby rozkazy nie czekaly
        w kolejce sieci i nie dojechaly potem do serw seria). Wolajacy zostaje
        wtedy przy poprzednim rozkazie.
        """

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        ...

    @property
    def link_silent(self) -> bool:
        """Ostatni `read_joints` nie dostal ZADNEJ swiezej odpowiedzi - oddal stare pozycje.

        Blizniak nie liczy wtedy takiego odczytu jako pomiaru i wstrzymuje nadzor.
        Backend bez lacza (symulator) - zawsze False. Domyslnie czyta `_link_silent`,
        ktore ustawiaja `feetech` i `lerobot` (i atrapy w testach).
        """
        return bool(getattr(self, "_link_silent", False))

    def faults(self) -> list[str]:
        """Usterki sprzetu widziane w ostatnich odpowiedziach - czytelne zdania po polsku.

        Pusta lista = wszystko w porzadku. Serwo w ochronie (przeciazenie,
        przegrzanie, napiecie) zwalnia moment, ale dalej grzecznie odpowiada,
        wiec bez tego wyglada na zdrowe, a reszta ramienia jedzie dalej.
        Nigdy nie rzuca wyjatkiem - wola to petla sterowania co cykl.
        """
        return []

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """Twarde limity samego sprzetu w jednostkach aplikacji: {staw: (min, max)}.

        Tylko stawy, ktore sprzet naprawde ogranicza (np. limity kata w EEPROM
        serw). Wyzsze warstwy zawezaja nimi swoje zakresy, zeby rozkaz nigdy
        nie byl przycinany po cichu przez serwo.
        """
        return {}

    def gripper_ticks(self) -> tuple[float, float, float] | None:
        """(zamkniety, otwarty, zero) chwytaka w surowych tikach serwa - to, na co TEN backend
        naprawde przelicza skale aplikacji: 0 = zamkniety, 100 = otwarty, zero = zero stawu.

        Blizniak liczy kat szczeki z tikow konfiguracji `feetech`. Backend, ktory
        mapuje 0..100 inaczej (LeRobot - po swoim zakresie z kalibracji), dawal
        polityce i sledzeniu kostki inny kat szczeki niz prawdziwy, a nikt tego
        nie widzial. None = backend nie ma tikow (symulator) albo ich nie zna.
        """
        return None

    def calibration_warnings(self) -> list[str]:
        """Niefatalne zastrzezenia do kalibracji ramienia (zdania dla panelu). Nigdy nie rzuca.

        Np. `lerobot` liczy stopnie od srodka zakresu z kalibracji, a blizniak od
        tiku `center_ticks` - ramie jedzie, ale katy sa przesuniete wzgledem modelu.
        """
        return []

    def step(self, dt: float) -> None:  # noqa: B027 - domyslnie nic nie robi
        """Krok symulacji. Prawdziwy robot ignoruje (rzeczywistosc liczy sie sama)."""

    def __enter__(self) -> "RobotBackend":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.disconnect()
