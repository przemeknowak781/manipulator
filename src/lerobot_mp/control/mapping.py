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
from ..vision.arm_features import ArmFeatures
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
    #: Wygladzone katy ramienia operatora (tryb `arm`).
    arm: ArmFeatures = field(default_factory=ArmFeatures.absent)
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

        # Tryb `arm`: katy sa w STOPNIACH, a nie w radianach jak katy dloni.
        # `beta` mnozy predkosc sygnalu, wiec te same nastawy co dla dloni
        # dzialalyby tu 57 razy mocniej - stad osobna sekcja konfiguracji.
        a = f.arm_angle
        self._f_azimuth = OneEuroFilter(a.min_cutoff, a.beta, a.d_cutoff)
        self._f_elevation = OneEuroFilter(a.min_cutoff, a.beta, a.d_cutoff)
        self._f_elbow = OneEuroFilter(a.min_cutoff, a.beta, a.d_cutoff)
        self._azimuth_unwrap = AngleUnwrapper(period=360.0)

        self._engaged = cfg.clutch.engaged_on_start
        self._key_engaged = cfg.clutch.engaged_on_start
        self._pending_gesture: bool | None = None
        self._pending_since = 0.0
        self._elapsed = 0.0

        self._anchor: HandFeatures | None = None
        self._anchor_arm: ArmFeatures | None = None
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
        """Czy mamy punkt odniesienia. W trybie `arm` decyduje o tym ramie."""
        if self.arm_mode:
            return self._anchor_arm is not None
        return self._anchor is not None

    @property
    def key_engaged(self) -> bool:
        """Sam stan sprzegla klawiszowego (SPACJA), bez udzialu gestu.

        Tryb `keys` nie wola `update()`, wiec nie ma skad wziac `engaged` -
        a sprzeglo z klawiatury dziala tam tak samo jak w pozostalych trybach.
        """
        return self._key_engaged

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

    @property
    def arm_mode(self) -> bool:
        return self.cfg.mapping.mode.lower() == "arm"

    def release_anchor(self) -> None:
        """Zrywa zaczepienie - kolejne zalaczenie zacznie od nowej pozycji dloni."""
        self._anchor = None
        self._anchor_arm = None
        self._anchor_joints = {}
        self._anchor_ee = None

    def _update_clutch(self, features: HandFeatures, dt: float, present: bool) -> bool:
        """Wyznacza stan sprzegla wg trybu z konfiguracji (z debouncingiem).

        `present` to obecnosc OPERATORA - w trybie `arm` decyduje o niej
        widocznosc ramienia, a nie dloni.
        """
        mode = self.cfg.clutch.mode.lower()
        self._elapsed += dt

        if mode == "always":
            desired = present
        elif mode == "key":
            desired = self._key_engaged and present
        elif mode == "gesture":
            # Zwiniete trzy ostatnie palce = pauza. Klawisz dziala jako
            # dodatkowy wylacznik (spacja moze zablokowac sterowanie calkiem).
            if features.present:
                desired = present and not features.curled and self._key_engaged
            else:
                # W trybie `arm` dlon bywa chwilowo niewidoczna, chociaz ramie
                # jest sledzone. Zamiast zgadywac gest, zostawiamy stan bez
                # zmiany - od zatrzymania sa spacja, klawisz X i watchdog.
                desired = self._engaged and present and self._key_engaged
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

    def _smooth_arm(self, arm: ArmFeatures, dt: float) -> ArmFeatures:
        if not arm.present:
            return arm
        azimuth = self._azimuth_unwrap(arm.azimuth)
        return ArmFeatures(
            present=True,
            side=arm.side,
            elevation=self._f_elevation(arm.elevation, dt),
            azimuth=self._f_azimuth(azimuth, dt),
            elbow=self._f_elbow(arm.elbow, dt),
            visibility=arm.visibility,
            points_px=arm.points_px,
        )

    def _reset_filters(self) -> None:
        for flt in (
            self._fx,
            self._fy,
            self._fscale,
            self._froll,
            self._fpitch,
            self._f_azimuth,
            self._f_elevation,
            self._f_elbow,
        ):
            flt.reset()
        self._roll_unwrap.reset()
        self._azimuth_unwrap.reset()
        self._gripper_ema.reset()

    # ------------------------------------------------------------------- krok
    def update(
        self,
        features: HandFeatures,
        joints_now: dict[str, float],
        dt: float,
        arm: ArmFeatures | None = None,
    ) -> ControlOutput:
        """Liczy zadane pozycje stawow dla biezacej klatki.

        Args:
            features: surowe cechy dloni z `extract_features` (moze byc "absent").
            joints_now: aktualnie zadane/zmierzone pozycje stawow robota.
            dt: czas od poprzedniej klatki [s].
            arm: katy ramienia operatora - wymagane w trybie `arm`.
        """
        arm = arm or ArmFeatures.absent()
        arm_mode = self.arm_mode
        # W trybie `arm` to ramie decyduje o obecnosci operatora; dlon jest
        # dodatkiem (nadgarstek i chwytak) i moze chwilowo zniknac.
        present = arm.present if arm_mode else features.present

        if not present:
            self._reset_filters()
            self._update_clutch(HandFeatures.absent(), dt, present=False)
            self.release_anchor()
            self._prev_features = None
            return ControlOutput(
                targets=None,
                engaged=False,
                hand_present=False,
                reason="nie widze ramienia" if arm_mode else "brak dloni",
            )

        smooth = self._smooth(features, dt) if features.present else features
        smooth_arm = self._smooth_arm(arm, dt)
        if features.present:
            self._prev_features = smooth
        engaged = self._update_clutch(smooth, dt, present=True)

        if not engaged:
            return ControlOutput(
                targets=None,
                engaged=False,
                hand_present=features.present,
                features=smooth,
                arm=smooth_arm,
                reason=self._clutch_reason(smooth),
            )

        if not self.anchored:
            self._capture_anchor(smooth, joints_now, smooth_arm)
        elif self._anchor is None and features.present:
            # Dlon pojawila sie dopiero teraz - zaczepiamy ja bez zrywania
            # zaczepienia ramienia, zeby robot nie drgnal.
            self._anchor = smooth

        if arm_mode:
            targets, ee, ik = self._map_arm(smooth, smooth_arm), None, None
        elif self.cfg.mapping.mode.lower() == "ik":
            targets, ee, ik = self._map_ik(smooth)
        else:
            targets, ee, ik = self._map_direct(smooth), None, None

        if self.cfg.mapping.gripper_source.lower() == "pinch" and features.present:
            targets["gripper"] = self._map_gripper(smooth)

        return ControlOutput(
            targets=targets,
            engaged=True,
            hand_present=features.present,
            features=smooth,
            arm=smooth_arm,
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
        if self.arm_mode and not features.present:
            return "nie widze dloni - nadgarstek i chwytak stoja"
        return "sprzeglo rozlaczone"

    def _capture_anchor(
        self,
        features: HandFeatures,
        joints_now: dict[str, float],
        arm: ArmFeatures | None = None,
    ) -> None:
        """Zapamietuje punkt odniesienia operatora i robota w chwili zalaczenia."""
        self._anchor_arm = arm if (arm and arm.present) else None
        if self.cfg.mapping.relative:
            self._anchor = features if features.present else None
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

    def _joint_target(
        self, name: str, base: float, delta: float, gain: float | None = None
    ) -> float:
        """Cel stawu = punkt odniesienia + wzmocniona roznica (z korekta znaku).

        `gain` pozwala podac przelozenie spoza `joints` - korzysta z tego tryb
        `arm`, ktory ma wlasne, bezwymiarowe przelozenia.
        """
        jc = self.cfg.joint(name)
        sign = -1.0 if jc.invert else 1.0
        return base + sign * (jc.gain if gain is None else gain) * delta + jc.offset

    # ------------------------------------------------------------ tryb direct
    def _map_direct(self, f: HandFeatures) -> dict[str, float]:
        dx, dy, depth, droll, dpitch = self._deltas(f)
        base = self._anchor_joints
        # ZNAKI. Nie sa dobrane "zeby wygladalo dobrze" - wynikaja z geometrii
        # SO-101 zmierzonej w `scripts/derive_geometry.py`. W tej kalibracji
        # *rosnacy* `shoulder_lift` OPUSZCZA ramie, a rosnacy `elbow_flex`
        # je CHOWA, wiec intuicyjne kierunki wymagaja minusow. Test
        # `test_direct_and_ik_move_the_tip_the_same_way` pilnuje, zeby oba
        # tryby mapowania zgadzaly sie co do skutku, a nie co do liczb.
        return {
            # Dlon w lewo/prawo -> obrot podstawy.
            "shoulder_pan": self._joint_target("shoulder_pan", base["shoulder_pan"], dx),
            # Dlon w gore -> koncowka w gore (os Y obrazu rosnie w dol).
            "shoulder_lift": self._joint_target("shoulder_lift", base["shoulder_lift"], dy),
            # Dlon blizej kamery -> ramie wysuwa sie do przodu.
            "elbow_flex": self._joint_target("elbow_flex", base["elbow_flex"], -depth),
            # Pochylenie dloni -> pochylenie narzedzia w te sama strone.
            "wrist_flex": self._joint_target("wrist_flex", base["wrist_flex"], -dpitch),
            # Obrot dloni -> obrot nadgarstka.
            "wrist_roll": self._joint_target("wrist_roll", base["wrist_roll"], droll),
        }

    # --------------------------------------------------------------- tryb arm
    def _map_arm(self, hand: HandFeatures, arm: ArmFeatures) -> dict[str, float]:
        """Ramie operatora -> ramie robota, staw w staw.

        ZNAKI wynikaja z geometrii SO-101 (`scripts/derive_geometry.py`):
        rosnacy `shoulder_lift` OPUSZCZA ramie, a rosnacy `shoulder_pan`
        obraca je w prawo robota, podczas gdy rosnacy azymut operatora to
        ruch reki do przodu, czyli w druga strone. Stad dwa minusy.
        """
        assert self._anchor_arm is not None
        base = self._anchor_joints
        anchor = self._anchor_arm
        settings = self.cfg.arm

        d_azimuth = arm.azimuth - anchor.azimuth
        d_elevation = arm.elevation - anchor.elevation
        d_elbow = arm.elbow - anchor.elbow

        targets = {
            "shoulder_pan": self._joint_target(
                "shoulder_pan", base["shoulder_pan"], -d_azimuth, settings.pan_gain
            ),
            "shoulder_lift": self._joint_target(
                "shoulder_lift", base["shoulder_lift"], -d_elevation, settings.lift_gain
            ),
            # Zgiecie lokcia przeklada sie wprost: operator zgina, robot zgina.
            "elbow_flex": self._joint_target(
                "elbow_flex", base["elbow_flex"], d_elbow, settings.elbow_gain
            ),
        }

        # Nadgarstek nadal ze sledzenia dloni - sylwetka nie niesie obrotu
        # nadgarstka z uzyteczna dokladnoscia.
        if hand.present and self._anchor is not None:
            droll = math.degrees(hand.roll - self._anchor.roll)
            dpitch = math.degrees(hand.pitch - self._anchor.pitch)
            targets["wrist_flex"] = self._joint_target(
                "wrist_flex", base["wrist_flex"], -dpitch
            )
            targets["wrist_roll"] = self._joint_target(
                "wrist_roll", base["wrist_roll"], droll
            )
        return targets

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
