"""Mapowanie ruchu dloni na zadane pozycje stawow.

Dwa tryby:

* ``direct`` - kazda os dloni steruje jednym stawem. Nie wymaga znajomosci
  wymiarow ramienia, wiec dziala poprawnie przy dowolnej kalibracji. Domyslny.
* ``ik``     - pozycja dloni wyznacza punkt w przestrzeni, a katy stawow
  liczy odwrotna kinematyka. Ruch jest bardziej "kartezjanski", ale wymaga
  w miare poprawnej geometrii w konfiguracji.

W obu trybach domyslnie pracujemy *wzglednie*: w chwili zalaczenia sprzegla
zapamietujemy pozycje dloni i pozycje robota, a potem dodajemy tylko roznice.
Dzieki temu robot nigdy nie "przeskakuje" po zalaczeniu, a dlon mozna
przelozyc w wygodne miejsce (jak podniesienie myszy z podkladki).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import AppConfig, JOINT_NAMES
from ..vision.features import HandFeatures
from .filters import AngleUnwrapper, ExponentialFilter, OneEuroFilter
from .kinematics import ArmKinematics, IKResult


@dataclass
class ControlOutput:
    """Wynik mapowania dla jednej klatki."""

    #: Zadane pozycje stawow (przed nadzorem bezpieczenstwa) albo None.
    targets: dict[str, float] | None = None
    engaged: bool = False
    hand_present: bool = False
    #: Wygladzone cechy dloni (do rysowania na HUD).
    features: HandFeatures = field(default_factory=HandFeatures.absent)
    #: Zadany punkt koncowki [m] - tylko w trybie `ik`.
    ee_target: tuple[float, float, float] | None = None
    ik: IKResult | None = None
    #: Powod, dla ktorego nie ma celu (np. "brak dloni").
    reason: str = ""


def _apply_deadzone(value: float, deadzone: float) -> float:
    """Martwa strefa bez skoku: odejmuje prog zamiast zerowac odcinkami."""
    if deadzone <= 0.0:
        return value
    if abs(value) <= deadzone:
        return 0.0
    return value - math.copysign(deadzone, value)


class HandToJointMapper:
    """Przetwarza cechy dloni na zadane pozycje stawow (z filtracja i sprzeglem)."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.kinematics = ArmKinematics(cfg.geometry)

        f = cfg.filters
        self._fx = OneEuroFilter(f.position.min_cutoff, f.position.beta, f.position.d_cutoff)
        self._fy = OneEuroFilter(f.position.min_cutoff, f.position.beta, f.position.d_cutoff)
        self._fscale = OneEuroFilter(f.scale.min_cutoff, f.scale.beta, f.scale.d_cutoff)
        self._froll = OneEuroFilter(f.angle.min_cutoff, f.angle.beta, f.angle.d_cutoff)
        self._fpitch = OneEuroFilter(f.angle.min_cutoff, f.angle.beta, f.angle.d_cutoff)
        self._gripper_ema = ExponentialFilter(f.gripper_ema)
        self._roll_unwrap = AngleUnwrapper()

        self._engaged = cfg.clutch.engaged_on_start
        self._key_engaged = cfg.clutch.engaged_on_start
        self._pending_gesture: bool | None = None
        self._pending_since = 0.0
        self._elapsed = 0.0

        self._anchor: HandFeatures | None = None
        self._anchor_joints: dict[str, float] = {}
        self._anchor_ee: tuple[float, float, float] | None = None
        self._prev_features: HandFeatures | None = None

        # Progi chwytaka - mozna je nadpisac kalibracja "na zywo".
        self.pinch_open = cfg.mapping.pinch_open
        self.pinch_closed = cfg.mapping.pinch_closed

    # --------------------------------------------------------------- sprzeglo
    @property
    def engaged(self) -> bool:
        return self._engaged

    @property
    def anchored(self) -> bool:
        return self._anchor is not None

    def toggle_key_clutch(self) -> bool:
        """Przelacza sprzeglo klawiszem (spacja). Zwraca nowy stan."""
        self._key_engaged = not self._key_engaged
        if not self._key_engaged:
            self.release_anchor()
        return self._key_engaged

    def set_key_clutch(self, engaged: bool) -> None:
        self._key_engaged = engaged
        if not engaged:
            self.release_anchor()

    def release_anchor(self) -> None:
        """Zrywa zaczepienie - kolejne zalaczenie zacznie od nowej pozycji dloni."""
        self._anchor = None
        self._anchor_joints = {}
        self._anchor_ee = None

    def _update_clutch(self, features: HandFeatures, dt: float) -> bool:
        """Wyznacza stan sprzegla wg trybu z konfiguracji (z debouncingiem)."""
        mode = self.cfg.clutch.mode.lower()
        self._elapsed += dt

        if mode == "always":
            desired = features.present
        elif mode == "key":
            desired = self._key_engaged and features.present
        elif mode == "gesture":
            # Zwiniete trzy ostatnie palce = pauza. Klawisz dziala jako
            # dodatkowy wylacznik (spacja moze zablokowac sterowanie calkiem).
            desired = features.present and not features.curled and self._key_engaged
        else:
            raise ValueError(f"Nieznany tryb sprzegla: {self.cfg.clutch.mode!r}")

        # Debounce: stan zmienia sie dopiero, gdy utrzyma sie przez chwile.
        if desired != self._engaged:
            if self._pending_gesture != desired:
                self._pending_gesture = desired
                self._pending_since = self._elapsed
            elif self._elapsed - self._pending_since >= self.cfg.clutch.debounce_s:
                self._engaged = desired
                self._pending_gesture = None
                if not desired:
                    self.release_anchor()
        else:
            self._pending_gesture = None

        return self._engaged

    # ---------------------------------------------------------------- filtracja
    def _smooth(self, features: HandFeatures, dt: float) -> HandFeatures:
        if not features.present:
            return features
        roll = self._roll_unwrap(features.roll)
        return HandFeatures(
            present=True,
            x=self._fx(features.x, dt),
            y=self._fy(features.y, dt),
            scale=self._fscale(features.scale, dt),
            roll=self._froll(roll, dt),
            pitch=self._fpitch(features.pitch, dt),
            pinch=self._gripper_ema(features.pinch),
            extension=features.extension,
            curled=features.curled,
            handedness=features.handedness,
            score=features.score,
            palm_px=features.palm_px,
        )

    def _reset_filters(self) -> None:
        for flt in (self._fx, self._fy, self._fscale, self._froll, self._fpitch):
            flt.reset()
        self._roll_unwrap.reset()
        self._gripper_ema.reset()

    # ------------------------------------------------------------------- krok
    def update(
        self,
        features: HandFeatures,
        joints_now: dict[str, float],
        dt: float,
    ) -> ControlOutput:
        """Liczy zadane pozycje stawow dla biezacej klatki.

        Args:
            features: surowe cechy dloni z `extract_features` (moze byc "absent").
            joints_now: aktualnie zadane/zmierzone pozycje stawow robota.
            dt: czas od poprzedniej klatki [s].
        """
        if not features.present:
            self._reset_filters()
            self._update_clutch(features, dt)
            self.release_anchor()
            self._prev_features = None
            return ControlOutput(
                targets=None, engaged=False, hand_present=False, reason="brak dloni"
            )

        smooth = self._smooth(features, dt)
        self._prev_features = smooth
        engaged = self._update_clutch(smooth, dt)

        if not engaged:
            return ControlOutput(
                targets=None,
                engaged=False,
                hand_present=True,
                features=smooth,
                reason=self._clutch_reason(smooth),
            )

        if self._anchor is None:
            self._capture_anchor(smooth, joints_now)

        if self.cfg.mapping.mode.lower() == "ik":
            targets, ee, ik = self._map_ik(smooth)
        else:
            targets, ee, ik = self._map_direct(smooth), None, None

        targets["gripper"] = self._map_gripper(smooth)
        return ControlOutput(
            targets=targets,
            engaged=True,
            hand_present=True,
            features=smooth,
            ee_target=ee,
            ik=ik,
        )

    def _clutch_reason(self, features: HandFeatures) -> str:
        """Czytelne wyjasnienie, dlaczego sterowanie jest nieaktywne."""
        mode = self.cfg.clutch.mode.lower()
        if mode in ("key", "gesture") and not self._key_engaged:
            return "nacisnij SPACJE, aby wlaczyc sterowanie"
        if features.curled:
            return "palce zwiniete - pauza (wyprostuj, aby wznowic)"
        return "sprzeglo rozlaczone"

    def _capture_anchor(self, features: HandFeatures, joints_now: dict[str, float]) -> None:
        """Zapamietuje punkt odniesienia dloni i robota w chwili zalaczenia."""
        if self.cfg.mapping.relative:
            self._anchor = features
            self._anchor_joints = {n: float(joints_now.get(n, 0.0)) for n in JOINT_NAMES}
        else:
            # Tryb bezwzgledny: srodek kadru odpowiada pozycji domowej.
            self._anchor = HandFeatures(
                present=True,
                x=0.5,
                y=0.5,
                scale=features.scale,
                roll=features.roll,
                pitch=features.pitch,
            )
            self._anchor_joints = dict(self.cfg.safety.home)

        self._anchor_ee = self.kinematics.forward(
            self._anchor_joints.get("shoulder_pan", 0.0),
            self._anchor_joints.get("shoulder_lift", 0.0),
            self._anchor_joints.get("elbow_flex", 0.0),
            self._anchor_joints.get("wrist_flex", 0.0),
        )

    # ------------------------------------------------------------- odchylenia
    def _deltas(self, f: HandFeatures) -> tuple[float, float, float, float, float]:
        """Roznice wzgledem zaczepienia: (dx, dy, dgleb, droll_deg, dpitch_deg)."""
        assert self._anchor is not None
        a = self._anchor
        m = self.cfg.mapping

        dx = _apply_deadzone((f.x - a.x) * m.x_scale, m.deadzone)
        dy = _apply_deadzone((f.y - a.y) * m.y_scale, m.deadzone)

        # Glebokosc: logarytm stosunku rozmiarow jest symetryczny (zblizenie
        # i oddalenie o ten sam czynnik daja przeciwne, rowne co do modulu wartosci).
        if a.scale > 1e-6 and f.scale > 1e-6:
            depth = math.log(f.scale / a.scale) * m.depth_gain
        else:
            depth = 0.0
        depth = _apply_deadzone(depth, m.deadzone)

        droll = math.degrees(f.roll - a.roll)
        dpitch = math.degrees(f.pitch - a.pitch)
        return dx, dy, depth, droll, dpitch

    def _joint_target(self, name: str, base: float, delta: float) -> float:
        jc = self.cfg.joint(name)
        sign = -1.0 if jc.invert else 1.0
        return base + sign * jc.gain * delta + jc.offset

    # ------------------------------------------------------------ tryb direct
    def _map_direct(self, f: HandFeatures) -> dict[str, float]:
        dx, dy, depth, droll, dpitch = self._deltas(f)
        base = self._anchor_joints
        return {
            # Dlon w lewo/prawo -> obrot podstawy.
            "shoulder_pan": self._joint_target("shoulder_pan", base["shoulder_pan"], dx),
            # Dlon w gore -> ramie w gore (os Y obrazu rosnie w dol, stad minus).
            "shoulder_lift": self._joint_target("shoulder_lift", base["shoulder_lift"], -dy),
            # Dlon blizej kamery -> wysuniecie przedramienia.
            "elbow_flex": self._joint_target("elbow_flex", base["elbow_flex"], depth),
            # Pochylenie dloni -> pochylenie nadgarstka (1:1 przy gain = 1.0).
            "wrist_flex": self._joint_target("wrist_flex", base["wrist_flex"], dpitch),
            # Obrot dloni -> obrot nadgarstka.
            "wrist_roll": self._joint_target("wrist_roll", base["wrist_roll"], droll),
        }

    # ---------------------------------------------------------------- tryb ik
    def _map_ik(
        self, f: HandFeatures
    ) -> tuple[dict[str, float], tuple[float, float, float], IKResult]:
        dx, dy, depth, droll, dpitch = self._deltas(f)
        ws = self.cfg.workspace
        assert self._anchor_ee is not None

        ax, ay, az = self._anchor_ee
        # Dlon blizej kamery -> koncowka do przodu; w prawo -> w prawo (-Y);
        # w gore w kadrze (mniejszy y) -> w gore (+Z).
        x = ax + depth * ws.span_x
        y = ay - dx * ws.span_y
        z = az - dy * ws.span_z

        # Ograniczenie do pierscienia przestrzeni roboczej w rzucie z gory.
        # Promien liczymy od pionowej osi obrotu podstawy, a nie od poczatku
        # ukladu - os jest wysunieta do przodu o kilka centymetrow.
        axis_x = self.cfg.geometry.pan_axis_x
        radius = math.hypot(x - axis_x, y)
        if radius > 1e-6:
            clamped_radius = min(max(radius, ws.radius_min), ws.radius_max)
            if clamped_radius != radius:
                scale = clamped_radius / radius
                x = axis_x + (x - axis_x) * scale
                y *= scale

        anchor_pitch = self.kinematics.tool_pitch(
            self._anchor_joints.get("shoulder_lift", 0.0),
            self._anchor_joints.get("elbow_flex", 0.0),
            self._anchor_joints.get("wrist_flex", 0.0),
        )
        jc_wf = self.cfg.joint("wrist_flex")
        pitch_sign = -1.0 if jc_wf.invert else 1.0
        tool_pitch = anchor_pitch + pitch_sign * jc_wf.gain * dpitch
        tool_pitch = min(max(tool_pitch, ws.pitch_min_deg), ws.pitch_max_deg)

        ik = self.kinematics.inverse(x, y, z, tool_pitch)
        targets = ik.as_dict()
        targets["wrist_roll"] = self._joint_target(
            "wrist_roll", self._anchor_joints["wrist_roll"], droll
        )
        return targets, (x, y, z), ik

    # --------------------------------------------------------------- chwytak
    def _map_gripper(self, f: HandFeatures) -> float:
        jc = self.cfg.joint("gripper")
        span = self.pinch_open - self.pinch_closed
        if abs(span) < 1e-6:
            return jc.min
        t = (f.pinch - self.pinch_closed) / span
        t = min(max(t, 0.0), 1.0)
        if self.cfg.mapping.gripper_invert:
            t = 1.0 - t
        return jc.min + t * (jc.max - jc.min)

    # ------------------------------------------------------- kalibracja chwytaka
    def calibrate_pinch(self, features: HandFeatures, which: str) -> bool:
        """Zapisuje aktualne rozwarcie palcow jako "otwarte" albo "zamkniete".

        Pozwala dopasowac chwytak do wlasnej dloni bez edycji konfiguracji.
        """
        if not features.present:
            return False
        if which == "open":
            self.pinch_open = features.pinch
        elif which == "closed":
            self.pinch_closed = features.pinch
        else:
            raise ValueError("which musi byc 'open' albo 'closed'")
        # Gwarantujemy sensowna kolejnosc progow.
        if self.pinch_open - self.pinch_closed < 0.05:
            self.pinch_open = self.pinch_closed + 0.05
        return True

    @property
    def last_features(self) -> HandFeatures | None:
        return self._prev_features
