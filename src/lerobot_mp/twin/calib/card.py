"""Karta kalibracyjna: geometria, poza w szczekach i arkusz do druku.

Jedyny znacznik w calym systemie, i to tylko na czas kalibracji. Cienki
prostokat sciska sie za jeden koniec w szczekach, wiec jego normalna to os
zamykania, a reszta sterczy za koncowkami wzdluz podejscia. Na wystajacym koncu
jest po jednym tagu 36h11 na kazdej stronie, plecami do siebie - ktorakolwiek
strona zwroci sie do kamery, da sie ja odczytac. Ta sama idea co w
galaxeo-manipulators `pour_scene.CARD`, z wymiarami dla mniejszego chwytaka.

Uklad karty (jak w galaxeo): x na zewnatrz wzdluz podejscia, y w poprzek karty,
z to normalna, poczatek w srodku taga w plaszczyznie srodkowej karty.

Arkusz do druku to pasek z oboma tagami i linia zgiecia NA KONCU karty:
zgiecie staje sie krawedzia wystajaca za szczeki, oba wolne konce laduja razem
w szczekach, a drugi tag po zlozeniu jest obrocony o 180 stopni wokol osi
zgiecia (y) - dokladnie jak zaklada `tag_poses`. Zlozenie w innym miejscu
wymagaloby dopasowywania obu srodkow "na oko".
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .tags import MARKER_FRACTION, render


@dataclass(frozen=True)
class Card:
    #: Bok CZARNEGO kwadratu taga [m]. Drukarki skaluja, a ta liczba ustala skale
    #: calej kalibracji - po wydruku ZMIERZ ja linijka i wpisz prawdziwa.
    tag_size: float = 0.050
    #: Ile karty sterczy za koncowkami szczek [m]; srodek taga jest w polowie.
    out: float = 0.070
    #: Szerokosc karty [m] - musi zmiescic tag z bialym marginesem.
    width: float = 0.070
    #: Dlugosc karty [m]: `out` na zewnatrz, reszta miedzy szczekami.
    length: float = 0.110
    #: Grubosc (tektura + dwie warstwy papieru) [m].
    thickness: float = 0.002
    ids: tuple[int, int] = (0, 1)

    @property
    def plate(self) -> float:
        """Bok calej plytki taga razem z bialym marginesem [m]."""
        return self.tag_size / MARKER_FRACTION

    def tag_poses(self) -> dict[int, np.ndarray]:
        """{id taga: 4x4 poza taga w ukladzie karty} - dokladna geometria wydruku.

        Ten sam srodek w plaszczyznie karty, przeciwne normalne, odleglosc
        rowna grubosci. Solver nie placi za druga strone zadnym parametrem.
        """
        front, back = np.eye(4), np.eye(4)
        front[2, 3] = self.thickness / 2
        back[:3, :3] = np.diag([-1.0, 1.0, -1.0])     # 180 stopni wokol osi zgiecia (y)
        back[2, 3] = -self.thickness / 2
        return {self.ids[0]: front, self.ids[1]: back}

    def nominal(self, pinch_in_tcp: np.ndarray) -> np.ndarray:
        """Nominalna poza karty w ukladzie TCP, gdy wlozono ja "mniej wiecej prosto".

        `pinch_in_tcp` to punkt miedzy zamknietymi czubkami szczek (patrz
        `pinch_point`). Osie karty pokrywaja sie z osiami TCP, bo w `robots`
        TCP ma x wzdluz podejscia i z wzdluz zamykania - tak jak karta.
        To tylko punkt startowy: prawdziwa poze wyznacza solver.
        """
        T = np.eye(4)
        T[:3, 3] = np.asarray(pinch_in_tcp, float) + np.array([self.out / 2, 0.0, 0.0])
        return T

    # ------------------------------------------------------------------ druk
    def sheet(self, dpi: int = 300) -> np.ndarray:
        """Arkusz A4 w poziomie (RGB) z paskiem do wyciecia i zlozenia.

        W poziomie, bo pasek ma dwie dlugosci karty (220 mm przy domyslnej),
        a A4 w pionie ma 210 mm szerokosci.
        """
        page_w, page_h = 297.0, 210.0
        mm = dpi / 25.4
        page = np.full((int(round(page_h * mm)), int(round(page_w * mm)), 3), 255, np.uint8)
        L, W = self.length * 1000, self.width * 1000
        if 2 * L > page_w - 20 or W > page_h - 100:
            raise ValueError(f"karta {2 * L:.0f} x {W:.0f} mm nie miesci sie na arkuszu A4")

        # Pasek 2L x W wysrodkowany; zgiecie w polowie jest koncem karty.
        x0 = (page_w - 2 * L) / 2
        y0 = 40.0
        fold = x0 + L

        def px(v: float) -> int:
            return int(round(v * mm))

        cv2.rectangle(page, (px(x0), px(y0)), (px(x0 + 2 * L), px(y0 + W)), (0, 0, 0), 1)
        for y in np.arange(y0, y0 + W, 3.0):                  # linia zgiecia: kreskowana
            cv2.line(page, (px(fold), px(y)), (px(fold), px(min(y + 1.5, y0 + W))), (120, 120, 120), 1)

        side = self.plate * 1000
        for tag_id, centre in ((self.ids[0], fold - self.out * 500), (self.ids[1], fold + self.out * 500)):
            n = px(side)
            img = render(tag_id, n - n % 10)
            top, left = px(y0 + W / 2) - img.shape[0] // 2, px(centre) - img.shape[1] // 2
            page[top:top + img.shape[0], left:left + img.shape[1]] = img[..., None]

        lines = [
            "Karta kalibracyjna - lerobot-mp twin",
            f"1. Drukuj w skali 100% (bez dopasowania). Bok czarnego kwadratu ma miec {self.tag_size * 1000:.1f} mm.",
            "2. ZMIERZ go linijka i wpisz zmierzona wartosc w panelu kalibracji.",
            "3. Wytnij pasek po ciaglej linii, zegnij po kreskowanej, tagami na zewnatrz.",
            "4. Wklej w srodek sztywna tekture. Zgiecie to koniec karty.",
            f"5. Scisnij w szczekach wolny koniec; ok. {self.out * 1000:.0f} mm ma wystawac.",
            "   Dokladna poza w dloni nie ma znaczenia - kalibracja ja wyznacza.",
        ]
        for k, text in enumerate(lines):
            cv2.putText(page, text, (px(12), px(y0 + W + 18 + 7 * k)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55 * dpi / 150 if k == 0 else 0.42 * dpi / 150, (0, 0, 0), max(1, dpi // 150),
                        cv2.LINE_AA)
        return page


def pinch_point(kin) -> np.ndarray:
    """Punkt miedzy zamknietymi czubkami szczek, w ukladzie TCP [m].

    Liczony z modelu, a nie wpisany: srodek geomow czubkow przy chwytaku na 0.
    """
    spec = kin.spec
    if not all(spec.fingertips):
        return np.zeros(3)
    joints = dict(spec.home)
    if spec.gripper:
        joints[spec.gripper] = 0.0
    T_tcp = kin.tcp(joints)                         # liczy tez kinematyke calego modelu
    d, m = kin.data, kin.model
    base_inv = np.linalg.inv(_body_pose(d, kin.base_id))
    tips = [(base_inv @ np.r_[d.geom_xpos[m.geom(kin.prefix + g).id], 1.0])[:3] for g in spec.fingertips]
    mid = (tips[0] + tips[1]) / 2
    return T_tcp[:3, :3].T @ (mid - T_tcp[:3, 3])


def _body_pose(d, body_id: int) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = d.xmat[body_id].reshape(3, 3)
    T[:3, 3] = d.xpos[body_id]
    return T


def perturb(T: np.ndarray, rng: np.random.Generator, dpos: float = 0.012,
            drot: float = np.radians(10.0), extreme: bool = False) -> np.ndarray:
    """Poza, w ktorej karta NAPRAWDE wyladowala: nominalna razy przypadkowe
    przesuniecie do `dpos` i obrot do `drot` na kazdej osi - jak w galaxeo
    `sample_card_pose`. Tylko do symulacji; solver tego nie widzi."""
    def draw(lim):
        if not extreme:
            return rng.uniform(-lim, lim, 3)
        return np.sign(rng.uniform(-1, 1, 3)) * lim * rng.uniform(0.95, 1.0, 3)

    d = np.eye(4)
    d[:3, :3], _ = cv2.Rodrigues(draw(drot))
    d[:3, 3] = draw(dpos)
    return T @ d
