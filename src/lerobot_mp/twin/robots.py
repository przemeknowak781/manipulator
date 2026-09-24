"""Rejestr ramion: jeden opis na model, wspolny dla symulacji, UI i kalibracji.

Opis mowi tylko to, czego nie da sie wyczytac z samego MJCF: ktore stawy sa
ramieniem w kolejnosci aplikacji, ktory z nich jest chwytakiem, gdzie jest
punkt narzedzia (TCP) i na jakim ciele jedzie karta kalibracyjna. Reszta -
zakresy, masy, siatki - siedzi w MJCF i jest czytana z niego, a nie powielana.

Jednostki aplikacji sa jednostkami LeRobota: stawy w stopniach, chwytak w skali
0..100. MJCF liczy w radianach, a chwytak to w nim kat szczeki. Przeliczenie
jest w `kinematics.RobotKinematics` i tylko tam - dla chwytaka przez te same tiki
serwa (`RobotConfig.gripper_closed_ticks..gripper_open_ticks`), co backend
`feetech`, zeby 0..100 znaczylo ten sam kat szczeki w blizniaku i na ramieniu.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Korzen repozytorium - `src/lerobot_mp/twin/robots.py` -> trzy poziomy wyzej.
REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_asset(path: str | Path) -> Path:
    """Sciezka zasobu wzgledem biezacego katalogu albo korzenia repozytorium.

    Ta sama kolejnosc co w podgladzie 3D: najpierw katalog, z ktorego
    uruchomiono aplikacje, potem repozytorium - zeby dzialalo i z klonu,
    i z instalacji `pip install -e .`.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    for root in (Path.cwd(), REPO_ROOT):
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return REPO_ROOT / candidate


@dataclass(frozen=True)
class RobotSpec:
    """Wszystko, czego blizniak potrzebuje o ramieniu, poza samym MJCF."""

    name: str
    title: str
    #: Plik MJCF samego ramienia (bez stolu i swiatla) - scena dokleja reszte.
    mjcf: str
    #: Stawy w kolejnosci aplikacji. Nazwy jak w MJCF.
    joints: tuple[str, ...]
    #: Ktory ze stawow jest chwytakiem (jednostki aplikacji 0..100); None = brak.
    gripper: str | None
    #: Site punktu narzedzia - to jego poze liczy IK i do niego celuje UI.
    tcp_site: str
    #: Cialo, na ktorym jedzie karta kalibracyjna trzymana w szczekach.
    hand_body: str
    #: Cialo podstawy - uklad, w ktorym podajemy pozy kamer i celow.
    base_body: str
    #: Poza spoczynkowa w jednostkach aplikacji.
    home: dict[str, float] = field(default_factory=dict)
    #: Kierunek podejscia (od nadgarstka przez koncowki szczek) i os zamykania
    #: szczek, oba w ukladzie `tcp_site`. Z nich karta kalibracyjna wylicza swoja
    #: nominalna poze: sterczy wzdluz podejscia, a jej normalna to os zamykania.
    #: Sprawdza je test na geometrii modelu, a nie wiara w konwencje autora MJCF.
    tcp_approach: tuple[float, float, float] = (1.0, 0.0, 0.0)
    tcp_closing: tuple[float, float, float] = (0.0, 0.0, 1.0)
    #: Geomy czubkow obu szczek. Srodek miedzy nimi przy zamknietym chwytaku
    #: to miejsce, w ktorym szczeki trzymaja karte kalibracyjna.
    fingertips: tuple[str, str] = ("", "")
    #: Staw obracajacy narzedzie wokol kierunku podejscia. Fala kalibracyjna
    #: dobiera go analitycznie, zeby karta patrzyla na kamere.
    roll_joint: str | None = None
    #: Zakresy [jednostki aplikacji], z ktorych fala losuje pozostale stawy -
    #: wezsze niz mechaniczne, zeby karta krazyla nad stolem, a nie za plecami.
    wave_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: Kamery zamontowane na ramieniu (nazwy kamer w MJCF).
    onboard_cameras: tuple[str, ...] = ()
    #: Ciala obu szczek (stala, ruchoma) - czujniki kontaktu chwytu w zadaniach RL.
    jaw_bodies: tuple[str, str] = ("", "")

    @property
    def mjcf_path(self) -> Path:
        return resolve_asset(self.mjcf)

    @property
    def arm_joints(self) -> tuple[str, ...]:
        return tuple(j for j in self.joints if j != self.gripper)


SO101 = RobotSpec(
    name="so101",
    title="SO-101 (LeRobot)",
    mjcf="assets/robots/so101/so101.xml",
    joints=("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"),
    gripper="gripper",
    # `gripperframe` lezy miedzy szczekami, przy koncowkach - tam, gdzie chwyta.
    tcp_site="gripperframe",
    hand_body="gripper",
    base_body="base",
    home={
        "shoulder_pan": 0.0,
        "shoulder_lift": -23.3,
        "elbow_flex": 46.4,
        "wrist_flex": 3.3,
        "wrist_roll": 0.0,
        "gripper": 35.0,
    },
    fingertips=("fixed_jaw_sph_tip1", "moving_jaw_sph_tip1"),
    roll_joint="wrist_roll",
    wave_ranges={
        "shoulder_pan": (-75.0, 75.0),
        "shoulder_lift": (-80.0, 30.0),
        "elbow_flex": (-30.0, 90.0),
        "wrist_flex": (-70.0, 80.0),
    },
    onboard_cameras=("wrist_cam",),
    jaw_bodies=("gripper", "moving_jaw_so101_v1"),
)

REGISTRY: dict[str, RobotSpec] = {SO101.name: SO101}


def get_spec(name: str) -> RobotSpec:
    try:
        return REGISTRY[name.lower()]
    except KeyError as exc:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Nieznane ramie {name!r}. Dostepne: {known}") from exc
