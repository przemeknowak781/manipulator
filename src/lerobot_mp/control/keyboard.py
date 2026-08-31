"""Sterowanie ramieniem z klawiatury - bez kamery i bez sledzenia dloni.

Klawisze prowadza *koncowke chwytaka po przestrzeni*, a katy stawow liczy ta
sama odwrotna kinematyka, co w trybie `ik`. Dzieki temu jeden klawisz to jeden
zrozumialy ruch, a nie "obroc trzeci staw o pare stopni".

Uklad osi jest zaczepiony w BIEZACYM kierunku ramienia, a nie w ukladzie
swiata: `W` zawsze wysuwa chwytak *dalej od podstawy*, a `A`/`D` przesuwaja go
*w bok* wzgledem tego samego kierunku. Gdyby osie byly swiatowe, to samo `W`
raz wysuwalo by ramie, a raz prowadzilo je bokiem - zaleznie od tego, gdzie
akurat patrzy. Przy okazji ruch promieniowy trzyma sie pierscienia zasiegu,
zamiast z niego uciekac.

Klawisz trzymany wciskiem generuje powtorzenia z autopowtarzania systemu, a
nie zdarzenie "puszczono". Dlatego wcisniecie nie robi kroku, tylko *nadaje
osi predkosc na krotka chwile* (`hold_timeout`) - kolejne powtorzenia ja
przedluzaja, a brak powtorzen wygasza ruch sam. Efekt: plynna jazda przy
trzymaniu i natychmiastowe zatrzymanie po puszczeniu, bez skoku na starcie.
"""

from __future__ import annotations

import math

from ..config import AppConfig, JOINT_NAMES
from .kinematics import ArmKinematics
from .mapping import ControlOutput

#: Kody strzalek roznia sie miedzy backendami okien OpenCV, wiec trzymamy
#: wszystkie znane warianty: Windows (highgui), GTK/Qt oraz Cocoa.
ARROW_LEFT = (2424832, 65361, 63234)
ARROW_UP = (2490368, 65362, 63232)
ARROW_RIGHT = (2555904, 65363, 63235)
ARROW_DOWN = (2621440, 65364, 63233)

#: Os -> (nazwa, zwrot). Nazwy osi sa tez kluczami predkosci w konfiguracji.
DEFAULT_BINDINGS: dict[object, tuple[str, float]] = {
    # WSAD prowadzi chwytak po plaszczyznie widzianej z boku, a glebokosc
    # siedzi na Q/E - jak w wiekszosci sterowan przestrzennych.
    "w": ("height", 1.0),
    "s": ("height", -1.0),
    "a": ("lateral", -1.0),
    "d": ("lateral", 1.0),
    "e": ("reach", 1.0),
    "q": ("reach", -1.0),
    ARROW_LEFT: ("roll", -1.0),
    ARROW_RIGHT: ("roll", 1.0),
    ARROW_DOWN: ("grip", -1.0),
    ARROW_UP: ("grip", 1.0),
}

AXES = ("reach", "lateral", "height", "roll", "grip")


def _normalise(key: int) -> object | None:
    """Zamienia kod klawisza na klucz mapowania: znak albo krotke strzalki."""
    if key in (-1, 255):
        return None
    for arrow in (ARROW_LEFT, ARROW_UP, ARROW_RIGHT, ARROW_DOWN):
        if key in arrow:
            return arrow
    if 32 <= key < 127:
        return chr(key).lower()
    return None


class KeyboardPilot:
    """Prowadzi koncowke chwytaka klawiszami; zwraca `ControlOutput` jak mapper dloni."""

    def __init__(self, cfg: AppConfig, bindings: dict[object, tuple[str, float]] | None = None):
        self.cfg = cfg
        self.kinematics = ArmKinematics(cfg.geometry)
        self.bindings = dict(DEFAULT_BINDINGS if bindings is None else bindings)

        #: Zwrot ruchu na osi (-1, 0, 1) i chwila, do ktorej jest wazny.
        self._axis: dict[str, float] = {name: 0.0 for name in AXES}
        self._until: dict[str, float] = {name: 0.0 for name in AXES}
        self._clock = 0.0

        self._point: tuple[float, float, float] | None = None
        self._pitch = 0.0
        self._roll = 0.0
        self._grip = 0.0
        self._seeded = False

    # ------------------------------------------------------------- zaczepienie
    @property
    def seeded(self) -> bool:
        return self._seeded

    def release(self) -> None:
        """Zapomina zaczepienie - kolejny krok zacznie od zmierzonej pozy."""
        self._seeded = False
        for name in AXES:
            self._axis[name] = 0.0

    def seed(self, joints: dict[str, float]) -> None:
        """Ustawia cel na *zmierzona* poze robota, zeby nie skoczyl przy zalaczeniu."""
        get = lambda name: float(joints.get(name, self.cfg.safety.home.get(name, 0.0)))  # noqa: E731
        pan, lift = get("shoulder_pan"), get("shoulder_lift")
        elbow, wrist = get("elbow_flex"), get("wrist_flex")

        self._point = self.kinematics.forward(pan, lift, elbow, wrist)
        # Kat narzedzia zostaje staly przez cala jazde, wiec musi od razu
        # wpasc w zakres, w ktorym IK ma sensowne rozwiazania - inaczej caly
        # ruch odbywa sie w zle uwarunkowanym rogu przestrzeni.
        ws = self.cfg.workspace
        pitch = self.kinematics.tool_pitch(lift, elbow, wrist)
        self._pitch = min(max(pitch, ws.pitch_min_deg), ws.pitch_max_deg)
        self._roll = get("wrist_roll")
        self._grip = get("gripper")
        self._seeded = True

    # ---------------------------------------------------------------- klawisze
    def press(self, key: int) -> bool:
        """Przyjmuje kod klawisza. Zwraca True, gdy klawisz nalezal do sterowania."""
        binding = self.bindings.get(_normalise(key)) if key is not None else None
        if binding is None:
            return False
        axis, direction = binding
        self._axis[axis] = direction
        self._until[axis] = self._clock + self.cfg.keyboard.hold_timeout
        return True

    def _active(self, axis: str) -> float:
        """Zwrot osi, o ile wcisniecie jeszcze nie wygaslo."""
        if self._clock > self._until[axis]:
            self._axis[axis] = 0.0
        return self._axis[axis]

    # ------------------------------------------------------------------- krok
    def update(self, dt: float, engaged: bool, measured: dict[str, float]) -> ControlOutput:
        """Liczy zadane pozycje stawow po `dt` sekundach trzymania klawiszy."""
        self._clock += max(dt, 0.0)

        if not engaged:
            # Puszczamy zaczepienie, zeby po ponownym zalaczeniu ruszyc z tego
            # miejsca, w ktorym ramie *stoi*, a nie z zapamietanego sprzed pauzy.
            self.release()
            return ControlOutput(
                targets=None,
                engaged=False,
                hand_present=True,
                reason="sterowanie wylaczone",
            )

        if not self._seeded or self._point is None:
            self.seed(measured)

        kc = self.cfg.keyboard
        reach = self._active("reach") * kc.move_speed * dt
        lateral = self._active("lateral") * kc.move_speed * dt
        height = self._active("height") * kc.move_speed * dt

        x, y, z = self._point  # type: ignore[misc]

        # Kierunek "do przodu" to kierunek od osi obrotu podstawy do koncowki,
        # rzutowany na poziom. Przy celu praktycznie na osi nie ma czego rzutowac,
        # wiec bierzemy os X - dowolny wybor jest tam rownie dobry.
        dx, dy = x - self.cfg.geometry.pan_axis_x, y
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            fx, fy = 1.0, 0.0
        else:
            fx, fy = dx / norm, dy / norm

        x += fx * reach - fy * lateral
        y += fy * reach + fx * lateral
        z += height

        # Ten sam pierscien, ktorego pilnuje tryb `ik` dla dloni. Bez niego cel
        # dochodzi do samej krawedzi zasiegu, gdzie IK jest zle uwarunkowana:
        # kilkanascie centymetrow ruchu koncowki kosztuje wtedy ~90 stopni na
        # barku i ~115 na lokciu, limity predkosci stawow nasycaja sie i ramie
        # miota sie w pionie zamiast wychylic na zewnatrz.
        ws = self.cfg.workspace
        axis_x = self.cfg.geometry.pan_axis_x
        radius = math.hypot(x - axis_x, y)
        if radius > 1e-6:
            bounded = min(max(radius, ws.radius_min), ws.radius_max)
            if bounded != radius:
                scale = bounded / radius
                x = axis_x + (x - axis_x) * scale
                y *= scale
        self._point = (x, y, z)

        result = self.kinematics.inverse(x, y, z, self._pitch)
        if result.clamped:
            # Bez tego trzymanie klawisza poza zasiegiem "nakreca" zapamietany
            # punkt gdzies daleko i powrot trwalby tyle, ile trwalo wyjscie.
            self._point = result.reached

        targets = result.as_dict()
        roll = self._roll + self._active("roll") * kc.roll_speed * dt
        grip = self._grip + self._active("grip") * kc.gripper_speed * dt
        self._roll = self._clamp("wrist_roll", roll)
        self._grip = self._clamp("gripper", grip)
        targets["wrist_roll"] = self._roll
        targets["gripper"] = self._grip

        return ControlOutput(
            targets={name: targets[name] for name in JOINT_NAMES if name in targets},
            engaged=True,
            # Nadzor pilnuje, czy operator jest obecny. Przy klawiaturze dowodem
            # obecnosci jest sama klawiatura, wiec watchdog dloni nie ma tu czego
            # pilnowac - inaczej zamrazalby ruch po 0,4 s w kazdym trybie bez kamery.
            hand_present=True,
            ee_target=self._point,
            ik=result,
        )

    def _clamp(self, joint: str, value: float) -> float:
        jc = self.cfg.joint(joint)
        return min(max(value, jc.min), jc.max)

    # ------------------------------------------------------------------- opis
    def legend(self) -> str:
        """Krotka podpowiedz na HUD."""
        return "WS gora-dol  AD bok  QE tyl-przod  <- -> obrot  gora/dol chwytak"
