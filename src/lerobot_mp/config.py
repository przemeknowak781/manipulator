"""Konfiguracja aplikacji: dataclassy + wczytywanie z YAML.

Wszystkie parametry sterowania trzymamy w jednym miejscu, zeby dostrojenie
robota do wlasnej kamery i wlasnej kalibracji sprowadzalo sie do edycji
`configs/default.yaml` (albo kilku flag CLI), a nie do grzebania w kodzie.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

from .paths import CONFIG_ENV, config_from_env

# Kolejnosc stawow taka sama jak w LeRobot dla SO-100/SO-101.
JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

#: Chwytak w LeRobot zawsze jedzie w znormalizowanym zakresie 0..100,
#: niezaleznie od tego czy reszta stawow jest w stopniach czy w -100..100.
GRIPPER = "gripper"


@dataclass
class CameraConfig:
    #: Indeks kamery (0, 1, ...) albo sciezka do pliku wideo / strumienia.
    source: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    #: Lustrzane odbicie obrazu - wtedy ruch dloni w prawo = ruch obrazu w prawo.
    mirror: bool = True
    #: Wymuszenie kodeka MJPG czesto odblokowuje 30+ FPS na kamerach USB.
    fourcc: str | None = "MJPG"
    #: Zapetlanie pliku wideo (przydatne przy demo bez kamery).
    loop_video: bool = True


@dataclass
class TrackerConfig:
    #: "auto" | "tasks" (MediaPipe Tasks API) | "legacy" (mp.solutions.hands)
    backend: str = "auto"
    #: Sciezka do modelu .task; pobierany automatycznie przy pierwszym uruchomieniu.
    model_path: str = "models/hand_landmarker.task"
    model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )
    #: Ile dloni sledzic (sterujemy jedna, ale 2 pomaga wybrac wlasciwa).
    num_hands: int = 2
    #: "any" | "Left" | "Right" - ktora dlon steruje ramieniem.
    preferred_hand: str = "any"
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    #: "video" (synchronicznie, domyslnie) albo "live_stream" (asynchronicznie, mniejsze opoznienie)
    running_mode: str = "video"


@dataclass
class OneEuroConfig:
    """Parametry filtru One-Euro (kompromis drzenie <-> opoznienie)."""

    min_cutoff: float = 1.2
    beta: float = 0.03
    d_cutoff: float = 1.0


@dataclass
class FilterConfig:
    """Nastawy filtrow wejscia sterowania.

    UWAGA NA JEDNOSTKI. `beta` mnozy *predkosc* filtrowanego sygnalu, wiec ta
    sama liczba znaczy cos zupelnie innego dla kazdego z tych czterech wejsc.
    Zbyt mala `beta` zamienia One-Euro w zwykly filtr dolnoprzepustowy o stalej
    czestotliwosci `min_cutoff` - a wtedy caly jego sens (malo opoznienia przy
    szybkim ruchu) znika. Wartosci ponizej sa zmierzone, nie zgadniete:
    przy typowym machnieciu reka (0,5 Hz) daja odpowiednio 66, 63 i 32 ms
    opoznienia grupowego wobec 143, 173 i 89 ms przy poprzednich nastawach,
    przy praktycznie tym samym drzeniu na postoju.
    """

    #: Polozenie dloni w kadrze, 0..1. Ruch przez pol kadru w pol sekundy to
    #: ok. 1,0 jednostki/s - stad `beta` rzedu jednosci.
    position: OneEuroConfig = field(default_factory=lambda: OneEuroConfig(min_cutoff=1.0, beta=3.0))
    #: Rozmiar dloni w kadrze (ok. 0,12), czyli sygnal glebokosci. Zmienia sie
    #: kilkanascie razy wolniej niz polozenie, wiec `beta` musi byc tyle razy
    #: wieksza, zeby filtr w ogole zauwazyl ruch.
    scale: OneEuroConfig = field(default_factory=lambda: OneEuroConfig(min_cutoff=0.8, beta=10.0))
    #: Katy dloni (roll, pitch) w RADIANACH.
    angle: OneEuroConfig = field(default_factory=lambda: OneEuroConfig(min_cutoff=1.5, beta=1.5))
    #: Katy ramienia operatora w STOPNIACH (tryb `arm`). Ten sam ruch daje tu
    #: 57 razy wieksza liczbe niz w radianach, wiec `beta` musi byc 57 razy
    #: mniejsza - dlatego to osobny wpis, a nie ten sam co `angle`.
    arm_angle: OneEuroConfig = field(default_factory=lambda: OneEuroConfig(min_cutoff=1.5, beta=0.05))
    #: Chwytak wygladzamy prostym EMA (0 = brak wygladzania, 0.9 = mocne).
    gripper_ema: float = 0.5


@dataclass
class JointConfig:
    """Limity i dynamika pojedynczego stawu (w stopniach, chwytak w 0..100)."""

    min: float
    max: float
    #: Maksymalna predkosc zadana [jednostka/s] - twardy limit bezpieczenstwa.
    max_vel: float = 120.0
    #: Wzmocnienie mapowania cechy dloni na ten staw. UWAGA: jednostka zalezy
    #: od tego, co steruje danym stawem:
    #:   * pozycja/glebokosc dloni (pan, lift, elbow) -> stopnie na jednostke
    #:     znormalizowana (ruch dloni przez pol kadru to ok. 0,5),
    #:   * kat dloni (wrist_flex, wrist_roll) -> przelozenie bezwymiarowe
    #:     (1.0 = obrot nadgarstka 1:1 z obrotem dloni).
    gain: float = 1.0
    #: Odwrocenie kierunku (gdy Twoja kalibracja ma przeciwny znak).
    invert: bool = False
    #: Stale przesuniecie dodawane do celu.
    offset: float = 0.0


def _default_joints() -> dict[str, JointConfig]:
    return {
        # Limity sa nieco ciasniejsze niz zakresy z kalibracji SO-101
        # (+-110 / +-100 / +-96,8 / +-95 / -157..163), zeby zostawic margines
        # przed mechanicznym koncem zakresu.
        "shoulder_pan": JointConfig(min=-105.0, max=105.0, max_vel=140.0, gain=90.0),
        "shoulder_lift": JointConfig(min=-95.0, max=95.0, max_vel=120.0, gain=90.0),
        "elbow_flex": JointConfig(min=-92.0, max=92.0, max_vel=120.0, gain=90.0),
        # Nadgarstek jest sterowany KATEM dloni, wiec wzmocnienie to przelozenie:
        # 1.0 = ruch 1:1. Wartosci rzedu dziesiatek daly by limit przy kilku
        # stopniach ruchu reki.
        "wrist_flex": JointConfig(min=-95.0, max=95.0, max_vel=160.0, gain=1.2),
        "wrist_roll": JointConfig(min=-150.0, max=150.0, max_vel=220.0, gain=1.0),
        "gripper": JointConfig(min=0.0, max=100.0, max_vel=300.0, gain=100.0),
    }


@dataclass
class ArmGeometryConfig:
    """Geometria SO-101 [m] i [stopnie] - uzywana w trybie `ik` i w podgladzie.

    Wartosci NIE sa szacunkiem: zostaly wyliczone z prawdziwego zlozenia
    SO-101 (repozytorium `articulus` -> bryly STEP i URDF producenta) przez
    `scripts/derive_geometry.py`. Ten sam skrypt sprawdza, czy uproszczony
    model plaski zgadza sie z pelna kinematyka.

    Model plaski: obrot podstawy + trzy ogniwa w plaszczyznie pionowej.
    Kat bezwzgledny i-tego ogniwa to `znak * kat_stawu + offset`.
    """

    #: Wysokosc osi `shoulder_lift` nad podstawa.
    base_height: float = 0.11660
    #: Polozenie pionowej osi obrotu podstawy w ukladzie bazowym (X do przodu).
    #: NIE lezy ona w poczatku ukladu - pominiecie tego daje 30 mm bledu
    #: przy obrocie o 45 stopni.
    pan_axis_x: float = 0.03884
    #: Wysuniecie osi `shoulder_lift` przed os obrotu podstawy.
    shoulder_offset: float = 0.03040
    #: Boczne przesuniecie koncowki wzgledem plaszczyzny obrotu podstawy.
    lateral_offset: float = 0.00046
    #: Odleglosci miedzy osiami kolejnych stawow.
    upper_arm: float = 0.11600
    forearm: float = 0.13500
    #: Od osi `wrist_flex` do koncowki szczek chwytaka.
    wrist_to_tip: float = 0.16566

    #: Kat ogniwa przy zerowych katach stawow.
    lift_offset_deg: float = 76.032
    elbow_offset_deg: float = -73.825
    wrist_offset_deg: float = -5.766

    #: Zwrot kazdego stawu wzgledem kata ogniwa (+1 albo -1). W SO-101 po
    #: kalibracji LeRobot wszystkie cztery wychodza ujemne.
    pan_sign: float = -1.0
    lift_sign: float = -1.0
    elbow_sign: float = -1.0
    wrist_sign: float = -1.0

    #: Wybor galezi rozwiazania IK (lokiec zgiety w jedna albo w druga strone).
    #: `False` jest galezia miesczaca sie w zakresach stawow SO-101 - sprawdzone
    #: siatka poz w `scripts/derive_geometry.py`.
    elbow_up: bool = False


@dataclass
class WorkspaceConfig:
    """Przestrzen robocza dla trybu `ik` (wzgledem punktu zaczepienia) [m].

    Srodek i zakres kata narzedzia sa dobrane tak, zeby jak najwieksza czesc
    przestrzeni miescila sie w zakresach stawow SO-101 (~84% siatki poz,
    sprawdzane w `tests/test_kinematics.py`).
    """

    #: Ile metrow ruchu robota odpowiada pelnemu ruchowi dloni w kadrze.
    span_x: float = 0.12
    span_y: float = 0.20
    span_z: float = 0.12
    #: Srodek przestrzeni roboczej (do przodu, w bok, w gore) [m].
    center: tuple[float, float, float] = (0.32, 0.0, 0.10)
    #: Promienie liczone od PIONOWEJ OSI OBROTU podstawy, nie od poczatku ukladu.
    radius_min: float = 0.15
    radius_max: float = 0.42
    #: Zakres kata narzedzia sterowany pochyleniem dloni [stopnie].
    pitch_min_deg: float = -50.0
    pitch_max_deg: float = -5.0


@dataclass
class MappingConfig:
    #: Sposob mapowania ruchu operatora na stawy:
    #:   "direct" - kazda os dloni steruje jednym stawem,
    #:   "ik"     - pozycja dloni wyznacza punkt, katy liczy odwrotna kinematyka,
    #:   "arm"    - sledzimy CALE ramie operatora (bark, lokiec, nadgarstek)
    #:              i przekladamy je wprost na ramie robota.
    mode: str = "direct"
    #: Co steruje szczeka chwytaka:
    #:   "pinch" - odleglosc kciuk-wskazujacy (wymaga sledzenia dloni),
    #:   "none"  - chwytak nie rusza sie sam, tylko klawiszami.
    gripper_source: str = "pinch"
    #: Tryb wzgledny: ruch liczony od pozycji zaczepienia (jak podnoszenie myszy).
    relative: bool = True
    #: Martwa strefa wokol punktu zaczepienia (w jednostkach znormalizowanych cech).
    deadzone: float = 0.012
    #: Jak dlon steruje glebokoscia: rozmiar dloni w kadrze.
    depth_gain: float = 1.5
    #: Rozwarcie chwytaka: dystans kciuk-palec wskazujacy (znormalizowany).
    pinch_open: float = 0.62
    pinch_closed: float = 0.15
    #: Odwrocenie chwytaka (szczypniecie = zamkniecie).
    gripper_invert: bool = False
    #: Skalowanie osi obrazu -> cechy (mnozniki przed wzmocnieniem stawu).
    x_scale: float = 1.0
    y_scale: float = 1.0


@dataclass
class ArmTrackingConfig:
    """Tryb `arm`: sledzenie ramienia operatora przez MediaPipe Pose.

    Katy licza sie w ukladzie TULOWIA, a nie obrazu, wiec dzialaja tak samo,
    gdy operator stoi bokiem albo przechyla sie na krzesle. Przelozenia sa
    bezwymiarowe: 1.0 znaczy, ze ramie robota powtarza ruch 1:1.
    """

    #: "auto" (to ramie, ktore lepiej widac) | "Left" | "Right" - strona OPERATORA.
    side: str = "auto"
    #: Ponizej tej wiarygodnosci punktow (visibility) ramie uznajemy za niewidoczne.
    #: Bez tego progu zaslonieta reka dawalaby zgadniete katy i losowy ruch robota.
    min_visibility: float = 0.6

    #: Przelozenia poszczegolnych osi (1.0 = ruch 1:1 z ramieniem operatora).
    pan_gain: float = 1.0
    lift_gain: float = 1.0
    elbow_gain: float = 1.0

    #: Nadgarstek i chwytak nadal ze sledzenia dloni. Wylaczenie oszczedza
    #: czas procesora, ale zabiera obrot nadgarstka i sterowanie chwytakiem.
    use_hand: bool = True

    #: Wariant `lite` jest szybszy (ok. 17 ms wobec 22 ms) i po ustabilizowaniu
    #: sledzenia daje 0,4 stopnia rozrzutu na lokciu - grubo ponizej tego, co
    #: ma znaczenie przy prowadzeniu reka. Zamiana na `pose_landmarker_full`
    #: w obu polach ponizej daje 0,2 stopnia kosztem kilku milisekund.
    model_path: str = "models/pose_landmarker_lite.task"
    model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    )
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class ClutchConfig:
    """Sprzeglo: kiedy ruch dloni faktycznie steruje robotem."""

    #: "gesture" (zwiniete 3 ostatnie palce = pauza) | "key" (tylko spacja) | "always"
    mode: str = "gesture"
    #: Ile sekund gest musi byc stabilny, zeby przelaczyc stan.
    debounce_s: float = 0.12
    #: Czy start aplikacji od razu wlacza sterowanie.
    engaged_on_start: bool = False
    #: Prog rozpoznania zwinietych palcow (otwarta dlon ~0.78, piesc ~0.40).
    curl_threshold: float = 0.58


@dataclass
class SafetyConfig:
    #: Brak dloni dluzej niz tyle sekund -> zamrozenie celu.
    hold_timeout_s: float = 0.4
    #: Brak dloni dluzej niz tyle sekund -> powrot do pozycji domowej.
    return_home_timeout_s: float = 6.0
    #: Plynne dojscie do pozycji domowej przy starcie [s].
    startup_ramp_s: float = 2.5
    #: Globalny mnoznik limitow predkosci (0.5 = polowa, tryb "wolny").
    velocity_scale: float = 1.0
    #: Pozycja domowa/spoczynkowa (srodek zakresu po kalibracji LeRobot).
    #: Poza gotowosci: koncowka w srodku przestrzeni roboczej, chwytak
    #: skierowany lekko w dol (wyliczona odwrotna kinematyka, nie "z oka").
    #: Poza zerowa jest poprawna, ale oznacza ramie wyciagniete na pelna dlugosc.
    home: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 0.0,
            "shoulder_lift": -23.3,
            "elbow_flex": 46.4,
            "wrist_flex": 3.3,
            "wrist_roll": 0.0,
            "gripper": 35.0,
        }
    )
    #: Powrot do pozycji domowej przed rozlaczeniem.
    home_on_exit: bool = True


@dataclass
class RobotConfig:
    #: "sim" | "lerobot" | "feetech" | "auto".
    #:   lerobot - pelna zgodnosc z ekosystemem LeRobot (wymaga torcha),
    #:   feetech - rozmowa wprost z serwami przez port szeregowy (sam pyserial).
    #:   auto    - lerobot, jesli jest zainstalowany, inaczej feetech.
    backend: str = "sim"
    #: Port szeregowy plytki sterujacej, np. /dev/ttyACM0 albo COM5.
    port: str | None = None
    #: Identyfikator ramienia w LeRobot (wskazuje plik kalibracji).
    robot_id: str = "so101_follower"
    #: True -> stawy w stopniach (domyslne w LeRobot >= 0.4).
    use_degrees: bool = True
    #: Sprzetowy limit skoku wzgledem aktualnej pozycji (None = wylaczony).
    max_relative_target: float | None = 12.0
    calibration_dir: str | None = None
    #: Typ ramienia przekazywany do LeRobot: "so101" albo "so100".
    kind: str = "so101"
    #: Jak czesto odczytywac faktyczna pozycje stawow [Hz]. Kazdy odczyt to
    #: transakcja po porcie szeregowym, wiec nie ma sensu robic tego co klatke.
    read_state_hz: float = 10.0

    # --- ponizsze dotyczy wylacznie backendu `feetech` -----------------------
    #: Predkosc portu. Serwa STS3215 w SO-101 wychodza z fabryki na 1 Mbaud.
    baudrate: int = 1_000_000
    #: Tik odpowiadajacy zeru stawu. Serwo ma 4096 tikow na obrot, a przy
    #: standardowym montazu SO-101 sklada sie je wysrodkowane, czyli na 2048.
    center_ticks: int = 2048
    #: Zakres szczeki chwytaka w tikach: 0 w skali aplikacji to `closed`,
    #: 100 to `open`. Wartosci pochodza z limitow, ktore serwo chwytaka mialo
    #: zapisane u siebie w EEPROM-ie - czyli ze skoku szczeki zmierzonego na
    #: prawdziwym ramieniu, a nie z oszacowania.
    #: Sprawdz swoje: rozewrzyj szczeke reka i odczytaj `Present_Position`.
    gripper_closed_ticks: int = 1986
    gripper_open_ticks: int = 2670
    #: Wylaczenie momentu przy wyjsciu. Domyslnie NIE, bo wiotkie ramie opada
    #: pod wlasnym ciezarem - wlacz tylko, gdy wiesz, ze jest podparte.
    torque_off_on_exit: bool = False


@dataclass
class UIConfig:
    show: bool = True
    window_name: str = "LeRobot 101 x MediaPipe"
    draw_skeleton: bool = True
    draw_hud: bool = True
    #: Panel z podgladem ramienia obok obrazu z kamery.
    draw_arm_view: bool = True
    #: Podglad 3D prawdziwego zlozenia SO-101 zamiast rysunku schematycznego.
    preview_3d: bool = True
    #: Sciezka do modelu 3D; None = domyslna `assets/so101_preview.npz`.
    preview_asset: str | None = None
    #: Gorny limit odswiezania podgladu 3D. Rysuje go osobny watek, wiec nie
    #: zabiera czasu petli sterowania - na wolniejszej maszynie render sam
    #: zejdzie ponizej tej wartosci, zamiast opoznic ruch ramienia.
    preview_hz: float = 30.0
    #: Szerokosc panelu podgladu w pikselach.
    preview_width: int = 360
    #: Skalowanie okna podgladu.
    display_scale: float = 1.0
    #: Zapis podgladu do pliku wideo (None = bez zapisu). Dziala takze
    #: przy `show: false`, wiec nadaje sie do demonstracji bez ekranu.
    record_path: str | None = None


@dataclass
class KeyboardConfig:
    """Tryb `keys`: prowadzenie koncowki chwytaka klawiszami, bez kamery.

    Predkosci sa podane "na sekunde trzymania", a nie "na wcisniecie", bo
    klawisz trzymany wciskiem generuje powtorzenia z autopowtarzania systemu.
    """

    #: Predkosc liniowa koncowki [m/s] dla WSAD oraz R/F.
    move_speed: float = 0.12
    #: Predkosc obrotu nadgarstka [stopnie/s] dla strzalek w lewo/prawo.
    roll_speed: float = 60.0
    #: Predkosc zaciskania chwytaka [jednostki/s] dla strzalek gora/dol.
    gripper_speed: float = 60.0
    #: Jak dlugo wcisniecie utrzymuje ruch, gdy nie przyszlo kolejne powtorzenie.
    #: Musi byc dluzsze niz odstep autopowtarzania (~33 ms), inaczej ruch rwie;
    #: i wyraznie krotsze niz czas reakcji, inaczej ramie jedzie po puszczeniu.
    hold_timeout: float = 0.18


@dataclass
class AppConfig:
    loop_hz: float = 30.0
    #: Automatyczne zakonczenie po tylu sekundach (None = bez limitu).
    #: Przydatne do demonstracji i testow bez nadzoru.
    max_runtime_s: float | None = None
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    arm: ArmTrackingConfig = field(default_factory=ArmTrackingConfig)
    clutch: ClutchConfig = field(default_factory=ClutchConfig)
    keyboard: KeyboardConfig = field(default_factory=KeyboardConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    geometry: ArmGeometryConfig = field(default_factory=ArmGeometryConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    joints: dict[str, JointConfig] = field(default_factory=_default_joints)

    def joint(self, name: str) -> JointConfig:
        try:
            return self.joints[name]
        except KeyError as exc:  # pragma: no cover - blad konfiguracji
            raise KeyError(f"Brak konfiguracji stawu '{name}' w sekcji `joints`") from exc


# --------------------------------------------------------------------------
# Wczytywanie YAML -> dataclassy
# --------------------------------------------------------------------------


def _coerce(value: Any, target_type: Any) -> Any:
    """Rzutuje wartosc z YAML na typ pola dataclassy (na tyle, na ile trzeba)."""
    origin = get_origin(target_type)

    if origin is tuple:
        return tuple(value)
    if is_dataclass(target_type) and isinstance(value, dict):
        return _from_dict(target_type, value)
    if origin is dict:
        key_t, val_t = get_args(target_type)
        if is_dataclass(val_t):
            return {k: _from_dict(val_t, v) for k, v in value.items()}
        return dict(value)
    return value


def _from_dict(cls: Any, data: dict[str, Any]) -> Any:
    """Buduje dataclasse `cls`, uzupelniajac braki wartosciami domyslnymi."""
    if not isinstance(data, dict):
        raise TypeError(f"Oczekiwano mapowania dla {cls.__name__}, dostalem {type(data).__name__}")

    # `from __future__ import annotations` zamienia adnotacje w napisy, wiec
    # rozwiazujemy je do prawdziwych typow zanim cokolwiek rzutujemy.
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Nieznane klucze w sekcji {cls.__name__}: {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        kwargs[name] = _coerce(value, hints[name])
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    """Wczytuje konfiguracje: domyslne wartosci <- plik YAML <- nadpisania CLI.

    Bez `path` plik bierze sie ze zmiennej srodowiskowej `LEROBOT_MP_CONFIG`
    (jesli jest ustawiona). Tak blizniak - ktory wola `load_config()` w kilku
    miejscach, takze w procesie treningu - dostaje tiki chwytaka i port z pliku
    podanego raz w `lerobot-twin --config ...`.
    """
    data: dict[str, Any] = {}
    if path is None:
        path = config_from_env()
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{CONFIG_ENV}={path}: nie ma takiego pliku konfiguracji")
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Plik konfiguracyjny {path} musi zawierac mapowanie YAML")
        data = raw
    if overrides:
        data = _deep_merge(data, overrides)

    # Sekcja `joints` scala sie ze slownikiem domyslnym staw po stawie,
    # zeby mozna bylo nadpisac tylko jeden limit bez przepisywania calosci.
    joints_override = data.pop("joints", None)
    cfg = _from_dict(AppConfig, data)
    if joints_override:
        for joint_name, joint_data in joints_override.items():
            if joint_name not in cfg.joints:
                raise ValueError(f"Nieznany staw w sekcji `joints`: {joint_name}")
            merged = _deep_merge(dataclasses.asdict(cfg.joints[joint_name]), joint_data)
            cfg.joints[joint_name] = _from_dict(JointConfig, merged)
    return cfg


def config_to_dict(cfg: AppConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
