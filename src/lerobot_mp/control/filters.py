"""Filtry sygnalu: One-Euro, EMA, ograniczenie predkosci, rozwijanie katow.

Sledzenie dloni jest z natury drgajace. Zwykly filtr dolnoprzepustowy tlumi
drzenie, ale wprowadza opoznienie odczuwalne przy szybkim ruchu. Filtr
One-Euro rozwiazuje ten kompromis: przy wolnym ruchu filtruje mocno, przy
szybkim - prawie wcale.

Zrodlo: Casiez, Roussel, Vogel, "1e Filter" (CHI 2012).
"""

from __future__ import annotations

import math

TWO_PI = 2.0 * math.pi


def _alpha(cutoff: float, dt: float) -> float:
    """Wspolczynnik filtru dolnoprzepustowego dla zadanej czestotliwosci granicznej."""
    tau = 1.0 / (TWO_PI * cutoff)
    return 1.0 / (1.0 + tau / dt)


class ExponentialFilter:
    """Prosty filtr wykladniczy (EMA).

    `smoothing` w [0, 1): 0 = brak filtracji, wartosci blizej 1 = mocniejsze
    wygladzanie (i wieksze opoznienie).
    """

    def __init__(self, smoothing: float = 0.5):
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing musi byc w przedziale [0, 1)")
        self.smoothing = smoothing
        self._value: float | None = None

    def __call__(self, value: float) -> float:
        if self._value is None:
            self._value = value
        else:
            self._value = self.smoothing * self._value + (1.0 - self.smoothing) * value
        return self._value

    @property
    def value(self) -> float | None:
        return self._value

    def reset(self, value: float | None = None) -> None:
        self._value = value


class OneEuroFilter:
    """Adaptacyjny filtr One-Euro dla jednego skalarnego sygnalu.

    Args:
        min_cutoff: czestotliwosc graniczna przy zerowej predkosci [Hz].
            Mniej = gladziej, ale z wiekszym opoznieniem.
        beta: jak mocno filtr "odpuszcza" przy szybkim ruchu.
            Wiecej = mniejsze opoznienie przy gwaltownych ruchach.
        d_cutoff: czestotliwosc graniczna dla estymaty pochodnej.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("min_cutoff i d_cutoff musza byc dodatnie")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0

    def __call__(self, value: float, dt: float) -> float:
        if dt <= 0.0:
            # Brak uplywu czasu - nie ma czego filtrowac.
            return self._x_prev if self._x_prev is not None else value
        if self._x_prev is None:
            self._x_prev = value
            self._dx_prev = 0.0
            return value

        # 1. Estymata pochodnej, sama wygladzona.
        dx = (value - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # 2. Czestotliwosc graniczna rosnie wraz z predkoscia sygnalu.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * value + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0


class RateLimiter:
    """Ogranicza szybkosc zmian sygnalu (slew rate) - twardy limit bezpieczenstwa."""

    def __init__(self, max_rate: float, value: float = 0.0):
        if max_rate <= 0:
            raise ValueError("max_rate musi byc dodatnie")
        self.max_rate = max_rate
        self._value = value

    def __call__(self, target: float, dt: float) -> float:
        max_step = self.max_rate * max(dt, 0.0)
        delta = target - self._value
        if delta > max_step:
            delta = max_step
        elif delta < -max_step:
            delta = -max_step
        self._value += delta
        return self._value

    @property
    def value(self) -> float:
        return self._value

    def reset(self, value: float) -> None:
        self._value = value


class AngleUnwrapper:
    """Rozwija kat z zakresu (-pi, pi] w ciagly sygnal.

    Bez tego obrot nadgarstka przez granice +-180 stopni powodowalby skok
    zadanej pozycji o pelny obrot - i gwaltowny ruch serwa.
    """

    def __init__(self, period: float = TWO_PI):
        self.period = period
        self._prev_raw: float | None = None
        self._offset = 0.0

    def __call__(self, angle: float) -> float:
        if self._prev_raw is None:
            self._prev_raw = angle
            return angle

        delta = angle - self._prev_raw
        half = self.period / 2.0
        if delta > half:
            self._offset -= self.period
        elif delta < -half:
            self._offset += self.period
        self._prev_raw = angle
        return angle + self._offset

    def reset(self) -> None:
        self._prev_raw = None
        self._offset = 0.0
