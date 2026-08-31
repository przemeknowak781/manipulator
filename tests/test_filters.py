"""Filtry sygnalu."""

from __future__ import annotations

import math
import random

import pytest

from lerobot_mp.control.filters import (
    AngleUnwrapper,
    ExponentialFilter,
    OneEuroFilter,
    RateLimiter,
)

DT = 1.0 / 30.0


def test_one_euro_passes_first_sample_through():
    assert OneEuroFilter()(3.0, DT) == 3.0


def test_one_euro_suppresses_noise_around_constant():
    """Przy nieruchomym sygnale filtr ma tlumic drzenie."""
    rng = random.Random(7)
    flt = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    raw, filtered = [], []
    for _ in range(200):
        value = 1.0 + rng.gauss(0.0, 0.1)
        raw.append(value)
        filtered.append(flt(value, DT))

    def spread(values):
        tail = values[50:]
        mean = sum(tail) / len(tail)
        return sum((v - mean) ** 2 for v in tail) / len(tail)

    assert spread(filtered) < spread(raw) / 5.0
    assert filtered[-1] == pytest.approx(1.0, abs=0.06)


def test_one_euro_lags_less_when_beta_is_larger():
    """Sedno One-Euro: im szybszy ruch, tym slabsza filtracja.

    Sprawdzamy to wprost - ten sam sygnal narastajacy, dwa rozne `beta`.
    Filtr z wiekszym `beta` musi byc blizej prawdy.
    """

    def lag(beta: float) -> float:
        flt = OneEuroFilter(min_cutoff=1.0, beta=beta)
        value = 0.0
        for _ in range(60):
            value += 0.05
            out = flt(value, DT)
        return value - out

    fast, slow = lag(beta=0.5), lag(beta=0.0)
    assert 0.0 < fast < slow
    assert fast < 0.15  # opoznienie rzedu 50 ms przy 1,5 jednostki/s


def test_one_euro_ignores_nonpositive_dt():
    flt = OneEuroFilter()
    flt(1.0, DT)
    assert flt(99.0, 0.0) == pytest.approx(1.0)


def test_one_euro_rejects_bad_parameters():
    with pytest.raises(ValueError):
        OneEuroFilter(min_cutoff=0.0)


def test_exponential_filter_converges():
    flt = ExponentialFilter(0.5)
    assert flt(10.0) == 10.0  # pierwsza probka bez opoznienia
    for _ in range(50):
        out = flt(0.0)
    assert out == pytest.approx(0.0, abs=1e-6)


def test_exponential_filter_rejects_bad_smoothing():
    with pytest.raises(ValueError):
        ExponentialFilter(1.0)


def test_rate_limiter_caps_step():
    limiter = RateLimiter(max_rate=10.0, value=0.0)
    assert limiter(100.0, 0.1) == pytest.approx(1.0)
    assert limiter(100.0, 0.1) == pytest.approx(2.0)


def test_rate_limiter_is_symmetric():
    limiter = RateLimiter(max_rate=10.0, value=0.0)
    assert limiter(-100.0, 0.1) == pytest.approx(-1.0)


def test_rate_limiter_passes_small_changes():
    limiter = RateLimiter(max_rate=10.0, value=0.0)
    assert limiter(0.5, 1.0) == pytest.approx(0.5)


def test_angle_unwrapper_removes_jumps():
    """Obrot przez granice +-pi nie moze dawac skoku o pelny obrot."""
    unwrap = AngleUnwrapper()
    values = [unwrap(a) for a in (3.0, 3.1, -3.1, -3.0)]
    assert all(abs(b - a) < 0.5 for a, b in zip(values, values[1:]))
    assert values[-1] == pytest.approx(3.0 + 4 * 0.0413, abs=0.2)


def test_angle_unwrapper_is_continuous_over_full_turn():
    unwrap = AngleUnwrapper()
    previous = None
    for step in range(200):
        raw = math.atan2(math.sin(step * 0.1), math.cos(step * 0.1))
        value = unwrap(raw)
        if previous is not None:
            assert abs(value - previous) < 0.5
        previous = value
    assert previous == pytest.approx(199 * 0.1, abs=1e-6)


def group_delay(flt: OneEuroFilter, freq: float, amplitude: float, seconds: float = 20.0) -> float:
    """Opoznienie grupowe filtru dla sinusa o zadanej czestotliwosci [s].

    Faze wyjscia wyznaczamy dopasowaniem sinusa i cosinusa metoda najmniejszych
    kwadratow, a nie szukaniem maksimum na siatce - dzieki temu wynik nie zalezy
    od kroku probkowania.
    """
    omega = 2.0 * math.pi * freq
    times, values = [], []
    for step in range(int(seconds / DT)):
        t = step * DT
        out = flt(amplitude * math.sin(omega * t), DT)
        if t > 4.0:  # rozbieg filtru nie nalezy do pomiaru
            times.append(t)
            values.append(out)

    sin_energy = sum(math.sin(omega * t) ** 2 for t in times)
    cos_energy = sum(math.cos(omega * t) ** 2 for t in times)
    sin_part = sum(math.sin(omega * t) * v for t, v in zip(times, values)) / sin_energy
    cos_part = sum(math.cos(omega * t) * v for t, v in zip(times, values)) / cos_energy
    return -math.atan2(cos_part, sin_part) / omega


def one_euro(section) -> OneEuroFilter:
    return OneEuroFilter(section.min_cutoff, section.beta, section.d_cutoff)


def test_configured_hand_filters_lag_less_than_a_tenth_of_a_second():
    """Regresja na NASTAWY, nie na kod filtru.

    Za mala `beta` zamienia One-Euro w zwykly filtr dolnoprzepustowy o stalym
    `min_cutoff`. Filtr dalej "dziala" - tylko doklada ponad 140 ms opoznienia
    przy zwyklym machnieciu reka, czego nie widac inaczej niz mierzac faze.
    Amplitudy odpowiadaja typowemu ruchowi: pol kadru, zmiana rozmiaru dloni
    o cwierc, obrot nadgarstka o ok. 35 stopni.
    """
    from lerobot_mp.config import load_config

    filters = load_config().filters
    assert group_delay(one_euro(filters.position), freq=0.5, amplitude=0.08) < 0.080
    assert group_delay(one_euro(filters.scale), freq=0.5, amplitude=0.03) < 0.080
    assert group_delay(one_euro(filters.angle), freq=0.5, amplitude=0.6) < 0.080


def test_beta_too_small_for_the_unit_is_what_makes_the_filter_lag():
    """Kontrola dla testu wyzej: o zwloce decyduje `beta`, a nie `min_cutoff`.

    Ta sama czestotliwosc graniczna, tylko `beta` sprzed pomiaru - opoznienie
    rosnie prawie dwukrotnie.
    """
    lagging = OneEuroFilter(min_cutoff=1.0, beta=0.02)
    assert group_delay(lagging, freq=0.5, amplitude=0.08) > 0.120
