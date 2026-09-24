"""Scena MuJoCo skladana z opisu stanowiska: stol, ramie, kamery, karta, obiekty.

Scena jest kompilowana od zera z `MjSpec` przy kazdym `build` - tak jak
epizody w galaxeo-manipulators - wiec nic nie zostaje z poprzedniej konfiguracji,
a randomizacja wygladu czy polozenia kamer to po prostu inny `SceneConfig`.

Konwencje, ktorych trzeba pilnowac:

* **Poza kamery jest w konwencji OpenCV** (z do przodu, y w dol) i wzgledem
  PODSTAWY ramienia - tak ja zwraca kalibracja. MuJoCo patrzy wzdluz -z z y w
  gore, stad `R_mj = R_cv @ diag(1, -1, -1)` przy wstawianiu kamery.
* **Kamera dostaje pelna macierz K**, nie samo pole widzenia: `focal_pixel`,
  `principal_pixel` i `resolution`. Prawdziwa kamera ma punkt glowny poza
  srodkiem i czesto rozne fx/fy - z samym `fovy` symulowany kadr rozjezdzalby
  sie z prawdziwym o kilka pikseli, i to po cichu. Zgodnosc pilnuje test.
* Ramie jest wstawiane pod prefiksem `robot/`, wiec jego stawy, aktuatory
  i site'y maja w scenie nazwy `robot/<nazwa>`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .calib.card import Card
from .calib.tags import render as render_tag
from .kinematics import RobotKinematics, inverse, pose
from .robots import RobotSpec

PREFIX = "robot/"
#: MuJoCo kamera patrzy wzdluz -z z y w gore; OpenCV wzdluz +z z y w dol.
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


@dataclass
class CameraView:
    """Kamera w scenie: intrynsyki i poza wzgledem podstawy ramienia (OpenCV)."""

    name: str
    K: np.ndarray
    width: int
    height: int
    T_cam2base: np.ndarray

    @staticmethod
    def from_fov(name: str, width: int, height: int, fovy_deg: float, T_cam2base: np.ndarray) -> CameraView:
        f = (height / 2) / np.tan(np.radians(fovy_deg) / 2)
        K = np.array([[f, 0.0, width / 2 - 0.5], [0.0, f, height / 2 - 0.5], [0.0, 0.0, 1.0]])
        return CameraView(name, K, width, height, np.asarray(T_cam2base, float))


@dataclass
class Table:
    #: Wymiary blatu [m] i wysokosc blatu nad podloga.
    size: tuple[float, float] = (0.9, 0.6)
    height: float = 0.75
    #: Gdzie na blacie stoi podstawa ramienia (od srodka blatu) i jak jest obrocona.
    #: Domyslnie przy krotszej krawedzi, przodem (+x) w strone blatu - wtedy cala
    #: przestrzen robocza (0,1-0,4 m przed podstawa, +-0,3 m w bok) lezy na stole.
    base_xy: tuple[float, float] = (-0.35, 0.0)
    base_yaw: float = 0.0
    rgba: tuple[float, float, float, float] = (0.55, 0.50, 0.44, 1.0)


@dataclass
class Box:
    """Prosty obiekt na stole - do zadan RL i do sprawdzania kadrow."""

    name: str
    size: tuple[float, float, float]            # polowy bokow [m]
    xy: tuple[float, float]                     # polozenie na blacie w ukladzie podstawy [m]
    rgba: tuple[float, float, float, float] = (0.85, 0.25, 0.2, 1.0)
    mass: float = 0.05
    free: bool = True


@dataclass
class Panel:
    """Plaski obraz w scenie - tablica kalibracyjna, plakat, wydruk na blacie.

    `image` (H, W) albo (H, W, 3) uint8 wypelnia prostokat `size` [m] w plaszczyznie
    xy panelu, gorny wiersz obrazu na +y, patrzy wzdluz +z. Poza wzgledem podstawy.
    """

    name: str
    image: np.ndarray
    size: tuple[float, float]
    T_panel2base: np.ndarray


@dataclass
class SceneConfig:
    robot: RobotSpec
    table: Table = field(default_factory=Table)
    cameras: list[CameraView] = field(default_factory=list)
    #: Karta kalibracyjna w szczekach: geometria i PRAWDZIWA poza w ukladzie TCP.
    card: Card | None = None
    card_pose: np.ndarray | None = None
    #: Niewidzialna, pogrubiona bryla kolizyjna wokol karty - dla sprawdzacza
    #: kolizji, zeby fala trzymala prawdziwy odstep, a nie zera milimetrow.
    card_collider: float | None = None
    objects: list[Box] = field(default_factory=list)
    #: Obiekty, dla ktorych scena dostaje czujniki kontaktu z kazda ze szczek -
    #: `<obiekt>_jaw0` (stala) i `<obiekt>_jaw1` (ruchoma), 1 gdy sie stykaja.
    grasp_sensors: list[str] = field(default_factory=list)
    panels: list[Panel] = field(default_factory=list)
    #: Oswietlenie: lista (pozycja swiatla nad blatem, jasnosc).
    lights: list[tuple[tuple[float, float, float], float]] = field(
        default_factory=lambda: [((0.4, -0.6, 1.6), 0.55), ((-0.5, 0.3, 1.4), 0.35)]
    )


class Scene:
    """Skompilowana scena: model, stan, kinematyka ramienia i render z kamer."""

    def __init__(self, cfg: SceneConfig, model: mujoco.MjModel):
        self.cfg = cfg
        self.model = model
        self.data = mujoco.MjData(model)
        self.kin = RobotKinematics(cfg.robot, model, PREFIX)
        spec = cfg.robot
        self.act_ids = np.array([model.actuator(PREFIX + j).id for j in spec.joints])
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        mujoco.mj_forward(model, self.data)
        base = model.body(PREFIX + spec.base_body).id
        self.T_base2world = pose(self.data.xmat[base].reshape(3, 3), self.data.xpos[base])

    # ------------------------------------------------------------ ramie
    def set_joints(self, joints: dict[str, float], *, hold: bool = True) -> None:
        """Ustawia poze ramienia natychmiast (bez dynamiki) i - domyslnie - trzyma ja serwami."""
        q = self.kin.to_q(joints)
        self.data.qpos[self.kin.qadr] = q
        self.data.qvel[self.kin.dadr] = 0.0
        if hold:
            self.data.ctrl[self.act_ids] = q
        mujoco.mj_forward(self.model, self.data)

    def command(self, joints: dict[str, float]) -> None:
        """Zadaje cel serwom - ramie dojedzie tam w kolejnych krokach fizyki."""
        self.data.ctrl[self.act_ids] = self.kin.to_q(joints)

    def joints(self) -> dict[str, float]:
        """Zmierzone katy stawow w jednostkach aplikacji."""
        return self.kin.from_q(self.data.qpos[self.kin.qadr])

    def step(self, seconds: float) -> None:
        n = max(1, int(round(seconds / self.model.opt.timestep)))
        mujoco.mj_step(self.model, self.data, nstep=n)

    # ------------------------------------------------------------ kamery
    def render(self, camera: str) -> np.ndarray:
        """Kadr RGB z kamery sceny w jej natywnej rozdzielczosci."""
        view = self.camera(camera)
        key = (view.width, view.height)
        r = self._renderers.get(key)
        if r is None:
            r = mujoco.Renderer(self.model, height=view.height, width=view.width)
            self._renderers[key] = r
        r.update_scene(self.data, camera=camera)
        return r.render().copy()

    def camera(self, name: str) -> CameraView:
        for view in self.cfg.cameras:
            if view.name == name:
                return view
        raise KeyError(f"brak kamery {name!r} w scenie")

    def camera_pose(self, name: str) -> np.ndarray:
        """Poza kamery wzgledem podstawy w konwencji OpenCV, wyczytana z MODELU.

        Do sprawdzenia, ze scena wstawila kamere tam, gdzie kazal config.
        """
        c = self.model.camera(name).id
        T_world = pose(self.data.cam_xmat[c].reshape(3, 3) @ CV_TO_MJ, self.data.cam_xpos[c])
        return inverse(self.T_base2world) @ T_world

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()

    def __enter__(self) -> Scene:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ------------------------------------------------------------------ budowa
def _quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def _add_tag_plate_mesh(spec: mujoco.MjSpec, plate: float) -> None:
    """Kwadratowa plytka z UV, wspolna dla wszystkich tagow - jak w galaxeo."""
    m = spec.add_mesh()
    m.name = "tag_plate"
    m.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    h = plate / 2
    m.uservert = np.array([-h, -h, 0, h, -h, 0, h, h, 0, -h, h, 0], np.float32)
    m.userface = np.array([0, 1, 2, 0, 2, 3], np.int32)
    m.usertexcoord = np.array([0, 1, 1, 1, 1, 0, 0, 0], np.float32)
    m.userfacetexcoord = np.array([0, 1, 2, 0, 2, 3], np.int32)


def _add_tag_material(spec: mujoco.MjSpec, tag_id: int, px: int = 400) -> str:
    img = render_tag(tag_id, px)
    tex = spec.add_texture()
    tex.name = f"tag{tag_id}"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.width, tex.height, tex.nchannel = img.shape[1], img.shape[0], 3
    tex.data = np.repeat(img[:, :, None], 3, axis=2).tobytes()
    mat = spec.add_material()
    mat.name = f"tag{tag_id}"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tag{tag_id}"
    mat.texuniform = False
    return mat.name


def _add_card(spec: mujoco.MjSpec, robot: mujoco.MjSpec, rspec: RobotSpec, card: Card,
              T_card2tcp: np.ndarray, collider: float | None = None) -> None:
    """Karta w szczekach: widoczna plytka, obudowa kolizyjna i tagi na obu stronach."""
    site = robot.site(rspec.tcp_site)
    T_tcp2hand = pose(_rotmat(site.quat), np.asarray(site.pos, float))
    T = T_tcp2hand @ T_card2tcp
    Rc, tc = T[:3, :3], T[:3, 3]
    hand = robot.body(rspec.hand_body)

    # Srodek pudelka cofniety od srodka taga: tylko `out` karty wystaje za
    # szczeki, reszta jest miedzy nimi.
    centre = tc + Rc @ np.array([card.out / 2 - card.length / 2, 0.0, 0.0])
    half = np.array([card.length / 2, card.width / 2, card.thickness / 2])
    body = hand.add_geom()
    body.name = "calib_card"
    body.type = mujoco.mjtGeom.mjGEOM_BOX
    # Rysowana karta minimalnie ciensza od prawdziwej: tagi lezace dokladnie na
    # jej powierzchni migotalyby (z-fighting) i detektor widzialby bialy prostokat.
    body.size = half - np.array([0.0, 0.0, 0.0006])
    body.pos, body.quat = centre, _quat(Rc)
    body.rgba = [0.97, 0.97, 0.97, 1.0]
    body.contype = body.conaffinity = 0
    body.mass = 0.0
    body.group = 1

    if collider is not None:
        pad = hand.add_geom()
        pad.name = "calib_card_pad"
        pad.type = mujoco.mjtGeom.mjGEOM_BOX
        pad.size = half + collider
        pad.pos, pad.quat = centre, _quat(Rc)
        pad.rgba = [0.0, 0.0, 0.0, 0.0]
        pad.contype = pad.conaffinity = 1
        pad.mass = 0.0
        pad.group = 3

    _add_tag_plate_mesh(robot, card.plate)
    for tag_id, T_tag in card.tag_poses().items():
        _add_tag_material(robot, tag_id)
        g = hand.add_geom()
        g.name = f"tag{tag_id}"
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = "tag_plate"
        g.material = f"tag{tag_id}"
        g.pos = tc + Rc @ T_tag[:3, 3]
        g.quat = _quat(Rc @ T_tag[:3, :3])
        g.contype = g.conaffinity = 0
        g.mass = 0.0
        g.group = 1


def _add_panel(spec: mujoco.MjSpec, panel: Panel, T_base2world: np.ndarray) -> None:
    img = np.asarray(panel.image, np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    tex = spec.add_texture()
    tex.name = f"panel_{panel.name}"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.width, tex.height, tex.nchannel = img.shape[1], img.shape[0], 3
    tex.data = np.ascontiguousarray(img).tobytes()
    mat = spec.add_material()
    mat.name = f"panel_{panel.name}"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    mat.texuniform = False
    w, h = panel.size[0] / 2, panel.size[1] / 2
    mesh = spec.add_mesh()
    mesh.name = f"panel_{panel.name}"
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    mesh.uservert = np.array([-w, -h, 0, w, -h, 0, w, h, 0, -w, h, 0], np.float32)
    mesh.userface = np.array([0, 1, 2, 0, 2, 3], np.int32)
    # Wiersz 0 obrazu na gorze (+y) - tak jak tagi karty.
    mesh.usertexcoord = np.array([0, 1, 1, 1, 1, 0, 0, 0], np.float32)
    mesh.userfacetexcoord = np.array([0, 1, 2, 0, 2, 3], np.int32)
    T = T_base2world @ np.asarray(panel.T_panel2base, float)
    g = spec.worldbody.add_geom()
    g.name = f"panel_{panel.name}"
    g.type = mujoco.mjtGeom.mjGEOM_MESH
    g.meshname = mesh.name
    g.material = mat.name
    g.pos, g.quat = T[:3, 3], _quat(T[:3, :3])
    g.contype = g.conaffinity = 0
    g.group = 1


def _rotmat(quat) -> np.ndarray:
    R = np.zeros(9)
    q = np.asarray(quat, float)
    mujoco.mju_quat2Mat(R, q / np.linalg.norm(q))
    return R.reshape(3, 3)


def build(cfg: SceneConfig) -> Scene:
    """Kompiluje scene od zera."""
    t = cfg.table
    top = t.height
    # Ramie jako pierwsze: jego opcje fizyki (menagerie stroi je pod manipulacje -
    # stozek eliptyczny, impratio, iteracje solvera) sa czescia opisu ramienia
    # i scena ma je przejac. Bez tego `attach` zostawia domyslne sceny i tylko
    # ostrzega, a chwyt zachowuje sie inaczej niz w modelu, ktory ktos dostroil.
    robot = mujoco.MjSpec.from_file(str(cfg.robot.mjcf_path))
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    for name in ("timestep", "integrator", "cone", "impratio", "iterations", "ls_iterations"):
        setattr(spec.option, name, getattr(robot.option, name))
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    width = max([v.width for v in cfg.cameras] + [640])
    height = max([v.height for v in cfg.cameras] + [480])
    spec.visual.global_.offwidth, spec.visual.global_.offheight = width, height

    world = spec.worldbody
    floor = world.add_geom()
    floor.name, floor.type = "floor", mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [3.0, 3.0, 0.05]
    floor.rgba = [0.32, 0.34, 0.37, 1.0]

    tab = world.add_geom()
    tab.name, tab.type = "table", mujoco.mjtGeom.mjGEOM_BOX
    tab.size = [t.size[0] / 2, t.size[1] / 2, top / 2]
    tab.pos = [0.0, 0.0, top / 2]
    tab.rgba = list(t.rgba)

    for k, (pos, strength) in enumerate(cfg.lights):
        light = world.add_light()
        light.name = f"light{k}"
        light.pos = [pos[0], pos[1], top + pos[2]]
        light.dir = -np.asarray([pos[0], pos[1], pos[2]], float) / np.linalg.norm(pos)
        light.diffuse = [strength] * 3
        light.castshadow = k == 0

    # Karta doklejana do dloni PRZED wstawieniem ramienia, potem attach pod prefiksem.
    if cfg.card is not None:
        from .calib.card import pinch_point

        T_card = cfg.card_pose
        if T_card is None:
            T_card = cfg.card.nominal(pinch_point(RobotKinematics(cfg.robot)))
        _add_card(spec, robot, cfg.robot, cfg.card, T_card, cfg.card_collider)

    c, s = np.cos(t.base_yaw), np.sin(t.base_yaw)
    R_base = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T_base2world = pose(R_base, np.array([t.base_xy[0], t.base_xy[1], top]))
    mount = world.add_frame()
    mount.pos = T_base2world[:3, 3]
    mount.quat = _quat(R_base)
    spec.attach(robot, prefix=PREFIX, frame=mount)

    for obj in cfg.objects:
        p = T_base2world @ np.array([obj.xy[0], obj.xy[1], obj.size[2], 1.0])
        body = world.add_body()
        body.name = obj.name
        body.pos = p[:3]
        if obj.free:
            body.add_freejoint()
        g = body.add_geom()
        g.name = obj.name
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = list(obj.size)
        g.rgba = list(obj.rgba)
        g.mass = obj.mass
        g.condim = 4
        g.friction = [1.0, 0.02, 0.002]

    for panel in cfg.panels:
        _add_panel(spec, panel, T_base2world)

    for obj in cfg.grasp_sensors:
        for k, jaw in enumerate(cfg.robot.jaw_bodies):
            s = spec.add_sensor()
            s.name = f"{obj}_jaw{k}"
            s.type = mujoco.mjtSensor.mjSENS_CONTACT
            s.objtype, s.objname = mujoco.mjtObj.mjOBJ_BODY, PREFIX + jaw
            s.reftype, s.refname = mujoco.mjtObj.mjOBJ_BODY, obj
            s.intprm = [1, 0, 1]                 # dane: "found"; bez redukcji; jeden kontakt

    for view in cfg.cameras:
        T_world = T_base2world @ np.asarray(view.T_cam2base, float)
        cam = world.add_camera()
        cam.name = view.name
        cam.pos = T_world[:3, 3]
        cam.quat = _quat(T_world[:3, :3] @ CV_TO_MJ)
        K = np.asarray(view.K, float)
        cam.resolution = [view.width, view.height]
        # Rozmiar matrycy jest tylko po to, zeby MuJoCo przeszlo z `fovy` na
        # model ogniskowej: przy ogniskowej w pikselach sam sie skraca.
        cam.sensor_size = [view.width * 1e-6, view.height * 1e-6]
        cam.focal_pixel = [K[0, 0], K[1, 1]]
        # Punkt glowny jako przesuniecie srodka obrazu wzgledem niego, w obu osiach
        # "srodek minus K". Srodek piksela (0, 0) to w OpenCV 0, wiec srodek
        # obrazu to (W - 1) / 2 i (H - 1) / 2. Znak ZMIERZONY na renderze, nie
        # wziety z dokumentacji: przy odwrotnym kadr przesuwal sie o dwukrotnosc
        # przesuniecia (62 px przy 30 px), zgodnie w calym kadrze.
        #
        # Obie osie sa symetryczne. Wczesniej w y bylo dodatkowe pol piksela,
        # zmierzone na srodkach CZERWONYCH plam - a dolna polowa kulki jest
        # w cieniu, odpada na progu koloru i podnosi srodek plamy o ~0,4 px.
        # Maska segmentacji (czysta geometria) pokazala, ze to pol piksela
        # przesuwalo caly kadr o 0,5 px w dol wzgledem rzutu K - to samo
        # w OpenGL i w rendererze MuJoCo Warp, ktory liczy promienie przez
        # srodki pikseli z tych samych intrynsyk. Przy f = 600 px to 0,05 st
        # pochylenia kamery w kalibracji.
        # Pilnuje tego `test_twin_scene.py` na maskach kulek w znanych punktach.
        cam.principal_pixel = [(view.width - 1) / 2 - K[0, 2], (view.height - 1) / 2 - K[1, 2]]

    return Scene(cfg, spec.compile())
