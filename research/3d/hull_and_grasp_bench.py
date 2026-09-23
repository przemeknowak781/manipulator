"""Pomiary CPU na tym laptopie dla czesci potoku, ktore nie wymagaja sieci neuronowych.

Uruchamiac z katalogu scratchpad, interpreterem projektu, z -B (bez .pyc w repo):
    .venv/Scripts/python.exe -B research/3d/hull_and_grasp_bench.py

1. Przepustowosc fp32 GEMM (numpy) - baza do szacowania czasu modeli ViT na CPU.
2. Kompilacja sceny blizniaka (SO-101 + stol + 4 kamery) i z obiektami-siatkami.
3. spec.recompile(model, data) po dodaniu obiektu.
4. Szybkosc symulacji z kontaktami (ile prob chwytu na sekunde da sie sprawdzic).
5. Otoczka wizualna (visual hull) z masek 4 kamer: czas i nadmiar objetosci.
6. Jedna proba chwytu z gory w MuJoCo (IK projektu) - czy obiekt zostaje w szczekach.
"""

from __future__ import annotations

import json
import time

import mujoco
import numpy as np
import trimesh

from lerobot_mp.twin import scene as sc
from lerobot_mp.twin.kinematics import RobotKinematics, inverse, pose
from lerobot_mp.twin.robots import SO101

OUT = {}
rng = np.random.default_rng(0)


def tic():
    return time.perf_counter()


# ---------------------------------------------------------------- 1. GEMM
def gemm_gflops(n=2048, reps=8):
    a = rng.random((n, n), dtype=np.float32)
    b = rng.random((n, n), dtype=np.float32)
    a @ b
    t = tic()
    for _ in range(reps):
        a @ b
    dt = tic() - t
    return reps * 2 * n**3 / dt / 1e9


OUT["gemm_fp32_gflops"] = None
print("GEMM fp32 GFLOPS:", OUT["gemm_fp32_gflops"], flush=True)


# ---------------------------------------------------------------- scena
def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    return pose(np.column_stack([x, np.cross(z, x), z]), np.asarray(eye, float))


CENTER = np.array([0.22, 0.08, 0.0])  # srodek pola roboczego w ukladzie podstawy
W, H = 640, 480


def cameras(n=4, radius=0.55, height=0.40):
    cams = []
    for k in range(n):
        a = np.radians(45 + k * 360 / n)
        eye = CENTER + np.array([radius * np.cos(a), radius * np.sin(a), height])
        cams.append(sc.CameraView.from_fov(f"cam{k}", W, H, 55.0, look_at(eye, CENTER + [0, 0, 0.04])))
    return cams


# Kawalek wypukly = sama chmura wierzcholkow; MuJoCo liczy z niej otoczke wypukla
# (siatka bez scian), wiec nie trzeba scipy/qhull po stronie Pythona.
def mug_parts(r_out=0.040, r_in=0.036, h=0.095, bottom=0.005, sectors=12):
    """Kubek jako lista WYPUKLYCH kawalkow (tak jak po dekompozycji) + analityczna zajetosc."""
    parts = []
    for k in range(sectors):
        a0, a1 = 2 * np.pi * k / sectors, 2 * np.pi * (k + 1) / sectors
        pts = [[r * np.cos(a), r * np.sin(a), z]
               for r in (r_in, r_out) for z in (bottom, h) for a in np.linspace(a0, a1, 4)]
        parts.append(np.array(pts))
    ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    parts.append(np.array([[r_out * np.cos(a), r_out * np.sin(a), z] for z in (0.0, bottom) for a in ang]))
    # ucho: 5 odcinkow "torusa" w plaszczyznie xz, po stronie +x
    R, r = 0.028, 0.006

    def centre(t):
        return np.stack([r_out - 0.004 + 0.8 * R * np.cos(t), 0 * t, h / 2 + R * np.sin(t)], -1)

    for k in range(5):
        t0, t1 = -np.pi / 2 + k * np.pi / 5, -np.pi / 2 + (k + 1) * np.pi / 5
        pts = []
        for t in np.linspace(t0, t1, 3):
            c = centre(np.array(t))
            for u in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                pts.append(c + r * np.array([np.cos(u) * np.cos(t), np.sin(u), np.cos(u) * np.sin(t)]))
        parts.append(np.array(pts))
    curve = centre(np.linspace(-np.pi / 2, np.pi / 2, 90))

    def inside(p):
        rr = np.hypot(p[:, 0], p[:, 1])
        wall = (rr >= r_in) & (rr <= r_out) & (p[:, 2] >= 0) & (p[:, 2] <= h)
        base = (rr <= r_out) & (p[:, 2] >= 0) & (p[:, 2] <= bottom)
        handle = np.zeros(len(p), bool)
        near = (p[:, 0] > r_out - 0.01) & (np.abs(p[:, 1]) < r)
        q = p[near]
        dmin = np.full(len(q), np.inf)
        for c in curve:
            dmin = np.minimum(dmin, np.linalg.norm(q - c, axis=1))
        handle[near] = dmin <= r
        return wall | base | handle

    return parts, inside


def box_parts(size=(0.03, 0.02, 0.05)):
    hx, hy, hz = size
    m = trimesh.creation.box(extents=[2 * hx, 2 * hy, 2 * hz]).apply_translation([0, 0, hz])
    return [np.asarray(m.vertices)], lambda p: (np.abs(p[:, 0]) <= hx) & (np.abs(p[:, 1]) <= hy) & (p[:, 2] >= 0) & (p[:, 2] <= 2 * hz)


def can_parts(r=0.033, h=0.12):
    a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    v = np.array([[r * np.cos(t), r * np.sin(t), z] for z in (0.0, h) for t in a])
    return [v], lambda p: (np.hypot(p[:, 0], p[:, 1]) <= r) & (p[:, 2] >= 0) & (p[:, 2] <= h)


def pen_parts(r=0.0055, length=0.14):
    a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    v = np.array([[x, r * np.cos(t), r + r * np.sin(t)] for x in (-length / 2, length / 2) for t in a])
    return [v], lambda p: (np.abs(p[:, 0]) <= length / 2) & (np.hypot(p[:, 1], p[:, 2] - r) <= r)


def build_spec(cfg: sc.SceneConfig):
    """Kopia sc.build az do kompilacji - zeby moc dokladac siatki do tego samego spec."""
    t = cfg.table
    top = t.height
    robot = mujoco.MjSpec.from_file(str(cfg.robot.mjcf_path))
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    for name in ("timestep", "integrator", "cone", "impratio", "iterations", "ls_iterations"):
        setattr(spec.option, name, getattr(robot.option, name))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960
    world = spec.worldbody
    floor = world.add_geom()
    floor.name, floor.type = "floor", mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [3.0, 3.0, 0.05]
    tab = world.add_geom()
    tab.name, tab.type = "table", mujoco.mjtGeom.mjGEOM_BOX
    tab.size = [t.size[0] / 2, t.size[1] / 2, top / 2]
    tab.pos = [0.0, 0.0, top / 2]
    light = world.add_light()
    light.pos = [0.3, -0.5, top + 1.5]
    light.dir = [-0.2, 0.3, -1.0]
    c, s = np.cos(t.base_yaw), np.sin(t.base_yaw)
    R_base = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T_base2world = pose(R_base, np.array([t.base_xy[0], t.base_xy[1], top]))
    mount = world.add_frame()
    mount.pos = T_base2world[:3, 3]
    mount.quat = sc._quat(R_base)
    spec.attach(robot, prefix=sc.PREFIX, frame=mount)
    for view in cfg.cameras:
        T_world = T_base2world @ np.asarray(view.T_cam2base, float)
        cam = world.add_camera()
        cam.name = view.name
        cam.pos = T_world[:3, 3]
        cam.quat = sc._quat(T_world[:3, :3] @ sc.CV_TO_MJ)
        K = np.asarray(view.K, float)
        cam.resolution = [view.width, view.height]
        cam.sensor_size = [view.width * 1e-6, view.height * 1e-6]
        cam.focal_pixel = [K[0, 0], K[1, 1]]
        cam.principal_pixel = [(view.width - 1) / 2 - K[0, 2], view.height / 2 - 1 - K[1, 2]]
    return spec, T_base2world


def add_object(spec, T_base2world, name, parts, xy, yaw=0.0, mass=0.08):
    """Obiekt ze zrekonstruowanej siatki: kazdy wypukly kawalek = osobna siatka kolizyjna."""
    p = T_base2world @ np.array([xy[0], xy[1], 0.0005, 1.0])
    body = spec.worldbody.add_body()
    body.name = name
    body.pos = p[:3]
    body.quat = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    body.add_freejoint()
    for k, v in enumerate(parts):
        mesh = spec.add_mesh()
        mesh.name = f"{name}_p{k}"
        mesh.uservert = np.asarray(v, np.float32).ravel()  # bez scian -> MuJoCo bierze otoczke
        g = body.add_geom()
        g.name = f"{name}_g{k}"
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = mesh.name
        g.mass = mass / len(parts)
        g.condim = 4
        g.friction = [0.8, 0.01, 0.001]
        g.rgba = [0.2 + 0.6 * rng.random(), 0.3, 0.7, 1.0]
    return body


# ---------------------------------------------------------------- 2. kompilacja
cfg = sc.SceneConfig(SO101, cameras=cameras())
t = tic()
base_scene = sc.build(cfg)
OUT["build_scene_empty_s"] = round(tic() - t, 3)
base_scene.close()

objects = {
    "mug": (mug_parts(), (0.16, 0.17), 0.3),
    "box": (box_parts(), (0.37, -0.07), 0.4),
    "can": (can_parts(), (0.10, 0.25), 0.0),
    "pen": (pen_parts(), (0.27, 0.07), 0.7),
    "block": (box_parts((0.03, 0.02, 0.025)), (0.20, -0.04), 0.4),
}
t = tic()
spec, T_b2w = build_spec(cfg)
for name, ((parts, _), xy, yaw) in objects.items():
    add_object(spec, T_b2w, name, parts, xy, yaw)
model = spec.compile()
OUT["build_scene_4_objects_s"] = round(tic() - t, 3)
OUT["convex_parts_total"] = int(sum(len(p[0][0]) for p in objects.values()))
data = mujoco.MjData(model)

# ---------------------------------------------------------------- 3. recompile
extra_parts, _ = can_parts(0.02, 0.08)
add_object(spec, T_b2w, "extra", extra_parts, (0.34, 0.26))
t = tic()
model2, data2 = spec.recompile(model, data)
OUT["recompile_after_adding_object_s"] = round(tic() - t, 3)

# ---------------------------------------------------------------- 4. symulacja
kin = RobotKinematics(SO101, model2, sc.PREFIX)
act = np.array([model2.actuator(sc.PREFIX + j).id for j in SO101.joints])
out_of_view = {"shoulder_pan": 100.0, "shoulder_lift": -95.0, "elbow_flex": 90.0}
q = kin.to_q(out_of_view)
data2.qpos[kin.qadr] = q
data2.ctrl[act] = q
mujoco.mj_forward(model2, data2)
for _ in range(200):  # obiekty osiadaja na stole
    mujoco.mj_step(model2, data2)
n = 2000
t = tic()
mujoco.mj_step(model2, data2, nstep=n)
dt = tic() - t
OUT["sim_steps_per_s"] = round(n / dt)
OUT["sim_realtime_factor"] = round(n * model2.opt.timestep / dt, 1)
OUT["timestep_s"] = model2.opt.timestep
OUT["ncon_resting"] = int(data2.ncon)
print("sim:", OUT["sim_steps_per_s"], "steps/s", flush=True)

import cv2  # noqa: E402

rgb_r = mujoco.Renderer(model2, height=H, width=W)
for v in cfg.cameras:
    rgb_r.update_scene(data2, camera=v.name)
    cv2.imwrite(f"{v.name}.png", rgb_r.render()[..., ::-1])
rgb_r.close()


def tip_gap(g):
    kin._apply(kin.to_q({**SO101.home, "gripper": g}))
    a = kin.data.geom_xpos[model2.geom(sc.PREFIX + SO101.fingertips[0]).id]
    b = kin.data.geom_xpos[model2.geom(sc.PREFIX + SO101.fingertips[1]).id]
    return float(np.linalg.norm(a - b))


OUT["tip_gap_mm"] = {str(g): round(1000 * tip_gap(g), 1) for g in (0, 25, 50, 75, 100)}
CLOSED, OPEN = (0.0, 100.0) if tip_gap(0) < tip_gap(100) else (100.0, 0.0)
print("tip gaps:", OUT["tip_gap_mm"], flush=True)

# ---------------------------------------------------------------- 5. visual hull
views = cfg.cameras
T_base2world = pose(data2.xmat[kin.base_id].reshape(3, 3), data2.xpos[kin.base_id])
GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)
ROBOT_GEOMS = np.array([g for g in range(model2.ngeom)
                        if model2.body(model2.geom_bodyid[g]).name.startswith(sc.PREFIX)])


def render_masks(hide_robot: bool):
    r = mujoco.Renderer(model2, height=H, width=W)
    r.enable_segmentation_rendering()
    opt = mujoco.MjvOption()
    if hide_robot:
        opt.geomgroup[2] = 0  # widoczne siatki ramienia sa w grupie 2
    out = {}
    r.update_scene(data2, camera=views[0].name, scene_option=opt)
    r.render()  # rozgrzewka kontekstu GL
    t = tic()
    for v in views:
        r.update_scene(data2, camera=v.name, scene_option=opt)
        out[v.name] = r.render().copy()
    dt = tic() - t
    r.close()
    return out, dt


# Scena 1: ramie ukryte (idealny przypadek). Scena 2: ramie w pozie spoczynkowej nad stolem.
masks_ideal, dt_seg = render_masks(hide_robot=True)
OUT["seg_render_per_cam_ms"] = round(1000 * dt_seg / len(views), 1)
home_q = kin.to_q(SO101.home)
data2.qpos[kin.qadr] = home_q
mujoco.mj_forward(model2, data2)
masks_occl, _ = render_masks(hide_robot=False)
rgb_r = mujoco.Renderer(model2, height=H, width=W)
for v in views:
    rgb_r.update_scene(data2, camera=v.name)
    cv2.imwrite(f"home_{v.name}.png", rgb_r.render()[..., ::-1])
rgb_r.close()


def body_geoms(body_name):
    b = model2.body(body_name).id
    return [g for g in range(model2.ngeom) if model2.geom_bodyid[g] == b]


def carve(masks, body_name, inside_fn, occluder_aware: bool, voxel=0.002):
    """Otoczka wizualna. Piksel poza kadrem = brak informacji (nie rzezbimy).

    occluder_aware: rzezbimy tylko tam, gdzie piksel to NA PEWNO tlo (stol, podloga),
    a nie inny obiekt albo ramie, ktore moga zaslaniac cel.
    """
    bid = model2.body(body_name).id
    T_obj2world = pose(data2.xmat[bid].reshape(3, 3), data2.xpos[bid])
    T_obj2base = inverse(T_base2world) @ T_obj2world
    c = T_obj2base[:3, 3]
    lo = c + np.array([-0.09, -0.09, -0.005])
    hi = c + np.array([0.09, 0.09, 0.13])
    lo[2] = max(lo[2], 0.0)  # blat: nic pod stolem
    axes = [np.arange(lo[i] + voxel / 2, hi[i], voxel) for i in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    target = body_geoms(body_name)
    t0 = tic()
    keep = np.ones(len(g), bool)
    for v in views:
        seg = masks[v.name]
        is_geom = seg[..., 1] == GEOM
        allowed = is_geom & np.isin(seg[..., 0], target)
        if occluder_aware:
            # blizniak zna poze ramienia -> jego piksele to "nie wiem", nie "tlo"
            allowed |= is_geom & np.isin(seg[..., 0], ROBOT_GEOMS)
        Tc = inverse(np.asarray(v.T_cam2base, float))
        pc = g @ Tc[:3, :3].T + Tc[:3, 3]
        uv = pc @ np.asarray(v.K).T
        u = np.round(uv[:, 0] / uv[:, 2]).astype(int)
        w = np.round(uv[:, 1] / uv[:, 2]).astype(int)
        ok = (pc[:, 2] > 0) & (u >= 0) & (u < v.width) & (w >= 0) & (w < v.height)
        hit = np.ones(len(g), bool)  # poza kadrem: nie wiemy nic -> zostaw
        hit[ok] = allowed[w[ok], u[ok]]
        keep &= hit
    t_carve = tic() - t0
    p_obj = (g - c) @ T_obj2base[:3, :3]
    truth = inside_fn(p_obj)
    inter = (keep & truth).sum()
    # szerokosc w poprzek na polowie wysokosci: max rozpietosc w plaszczyznie XY
    zs = p_obj[:, 2]
    zmid = 0.5 * zs[truth].max()
    sl = keep & (np.abs(zs - zmid) < voxel)
    st = truth & (np.abs(zs - zmid) < voxel)

    def span(sel):
        q = p_obj[sel][:, :2]
        if len(q) == 0:
            return 0.0
        best = 0.0
        for a in np.radians(np.arange(0, 180, 5)):
            d = q @ np.array([np.cos(a), np.sin(a)])
            best = max(best, d.max() - d.min())
        return best

    return {
        "carve_s": round(t_carve, 3),
        "hull_over_true_volume": round(float(keep.sum() / max(truth.sum(), 1)), 2),
        "recall_true_inside_hull": round(float(inter / max(truth.sum(), 1)), 3),
        "max_width_mid_mm_hull_vs_true": (round(1000 * span(sl), 1), round(1000 * span(st), 1)),
    }


def joint_carve(masks, voxel=0.0025, robot_unknown=True):
    """Wspolna otoczka calej sceny + przypisanie wokseli do obiektow glosowaniem widokow.

    Etykieta piksela: 0 tlo (stol/podloga/nic), k obiekt k, -2 ramie (twin zna jego poze).
    Woksel zostaje, jesli ZADEN widok nie widzi w jego miejscu tla. Potem nalezy do
    obiektu, na ktory rzutuje sie w najwiekszej liczbie widokow.
    """
    names = list(objects)
    lab_of_geom = np.zeros(model2.ngeom, int)
    for k, n in enumerate(names, start=1):
        lab_of_geom[body_geoms(n)] = k
    lab_of_geom[ROBOT_GEOMS] = -2 if robot_unknown else 0
    lo = np.array([0.02, -0.12, 0.0])
    hi = np.array([0.46, 0.33, 0.13])
    axes = [np.arange(lo[i] + voxel / 2, hi[i], voxel) for i in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    t0 = tic()
    fg = np.ones(len(g), bool)
    votes = np.zeros((len(g), len(names) + 1), np.int8)
    for v in views:
        seg = masks[v.name]
        lab = np.zeros(seg.shape[:2], int)
        is_geom = (seg[..., 1] == GEOM) & (seg[..., 0] >= 0)
        lab[is_geom] = lab_of_geom[seg[..., 0][is_geom]]
        Tc = inverse(np.asarray(v.T_cam2base, float))
        pc = g @ Tc[:3, :3].T + Tc[:3, 3]
        uv = pc @ np.asarray(v.K).T
        u = np.round(uv[:, 0] / uv[:, 2]).astype(int)
        w = np.round(uv[:, 1] / uv[:, 2]).astype(int)
        ok = (pc[:, 2] > 0) & (u >= 0) & (u < v.width) & (w >= 0) & (w < v.height)
        l = np.full(len(g), -1)  # -1: poza kadrem, brak informacji
        l[ok] = lab[w[ok], u[ok]]
        fg &= l != 0
        pos = l > 0
        votes[np.nonzero(pos)[0], l[pos]] += 1
    owner = votes.argmax(axis=1)
    owner[(votes.max(axis=1) == 0) | ~fg] = 0
    dt = tic() - t0
    res = {"voxels": int(len(g)), "carve_s": round(dt, 2)}
    for k, n in enumerate(names, start=1):
        _, inside_fn = objects[n][0]
        bid = model2.body(n).id
        T = inverse(T_base2world) @ pose(data2.xmat[bid].reshape(3, 3), data2.xpos[bid])
        p_obj = (g - T[:3, 3]) @ T[:3, :3]
        truth = inside_fn(p_obj)
        mine = owner == k
        res[n] = {"hull_over_true_volume": round(float(mine.sum() / max(truth.sum(), 1)), 2),
                  "recall_true_inside_hull": round(float((mine & truth).sum() / max(truth.sum(), 1)), 3),
                  "precision": round(float((mine & truth).sum() / max(mine.sum(), 1)), 3)}
    return res


OUT["joint_hull_4cams_2.5mm"] = {}
for label, masks, unk in (("ideal_arm_hidden", masks_ideal, True), ("arm_at_home_robot_unknown", masks_occl, True),
                          ("arm_at_home_robot_as_background", masks_occl, False)):
    OUT["joint_hull_4cams_2.5mm"][label] = joint_carve(masks, robot_unknown=unk)
    print(label, json.dumps(OUT["joint_hull_4cams_2.5mm"][label]), flush=True)

# ---------------------------------------------------------------- 6. proba chwytu z gory
def grasp_trial(target_body, grip_height, closing_yaw, lift=0.04, pre_up=0.04, width=None, opening=None):
    """width: szerokosc obiektu wzdluz osi zamykania. Szczeka STALA jest ~20 mm od TCP po
    stronie -z (os zamykania), wiec TCP przesuwamy tak, by stala szczeka stanela 3 mm
    przed scianka obiektu, a ruchoma domknela sie z drugiej strony."""
    global OPEN
    open_saved = OPEN
    if opening is not None:
        OPEN = opening
    try:
        return _grasp_trial(target_body, grip_height, closing_yaw, lift, pre_up, width)
    finally:
        OPEN = open_saved


def _grasp_trial(target_body, grip_height, closing_yaw, lift, pre_up, width):
    m, d = model2, data2
    mujoco.mj_resetData(m, d)
    # obiekty na miejsce startowe z kompilacji (qpos0), ramie w pozie spoczynkowej
    q = kin.to_q(SO101.home)
    d.qpos[kin.qadr] = q
    d.ctrl[act] = q
    mujoco.mj_forward(m, d)
    mujoco.mj_step(m, d, nstep=100)
    bid = m.body(target_body).id
    T_obj = inverse(T_base2world) @ pose(d.xmat[bid].reshape(3, 3), d.xpos[bid])
    p = T_obj[:3, 3] + np.array([0, 0, grip_height])
    approach = np.array([0.0, 0.0, -1.0])
    if closing_yaw is None:
        # chwyt za scianke (kubek): punkt na sciance najblizej podstawy, zamykanie promieniowo
        to_base = -T_obj[:2, 3] / np.linalg.norm(T_obj[:2, 3])
        p[:2] += 0.038 * to_base
        closing_yaw = float(np.arctan2(to_base[1], to_base[0]))
    closing = np.array([np.cos(closing_yaw), np.sin(closing_yaw), 0.0])
    y = np.cross(closing, approach)
    R = np.column_stack([approach, y, closing])  # osie site'u TCP: x=podejscie, z=zamykanie
    if width is not None:
        p = p - (width / 2 + 0.003 - 0.0201) * closing
    t0 = tic()
    pre = kin.ik(p + [0, 0, pre_up], R, seed={**SO101.home, "gripper": OPEN})
    at = kin.ik(p, R, seed=pre.joints)
    up = kin.ik(p + [0, 0, lift], R, seed=at.joints)
    t_ik = tic() - t0
    if not (pre.ok and at.ok and up.ok):
        return {"ik_ok": False, "ik_s": round(t_ik, 2),
                "pos_err_mm": [round(1000 * s.pos_err, 1) for s in (pre, at, up)],
                "rot_err_deg": [round(float(np.degrees(s.rot_err)), 1) for s in (pre, at, up)]}
    t0 = tic()
    steps = 0

    def go(joints, seconds):
        nonlocal steps
        d.ctrl[act] = kin.to_q(joints)
        k = int(seconds / m.opt.timestep)
        mujoco.mj_step(m, d, nstep=k)
        steps += k

    z0 = d.xpos[bid][2]
    go({**pre.joints, "gripper": OPEN}, 1.0)
    go({**at.joints, "gripper": OPEN}, 0.8)
    go({**at.joints, "gripper": CLOSED}, 0.8)
    grip_cmd = kin.from_q(d.qpos[kin.qadr])["gripper"]
    go({**up.joints, "gripper": CLOSED}, 1.0)
    go({**up.joints, "gripper": CLOSED}, 0.5)  # trzymanie w gorze
    t_sim = tic() - t0
    dz = d.xpos[bid][2] - z0
    return {
        "ik_ok": True,
        "rot_err_deg": round(float(np.degrees(at.rot_err)), 1),
        "gripper_after_close_0_100": round(grip_cmd, 1),
        "lifted_mm": round(1000 * dz, 1),
        "success": bool(dz > 0.8 * lift),
        "ik_s": round(t_ik, 2),
        "sim_s": round(t_sim, 3),
        "sim_steps": steps,
    }


OUT["grasp_trials"] = {}
trials = (
    # (obiekt, wysokosc TCP, kat osi zamykania, szerokosc w osi zamykania, otwarcie 0..100, nazwa)
    ("block", 0.02, 0.4, None, 100.0, "block60_tcp_at_centre_open100"),
    ("block", 0.02, 0.4, 0.060, 100.0, "block60_fixed_jaw_offset_open100"),
    ("block", 0.02, 0.4, 0.060, 60.0, "block60_fixed_jaw_offset_open60"),
    ("block", 0.02, 0.4 + np.pi / 2, 0.040, 45.0, "block40_fixed_jaw_offset_open45"),
    ("block", 0.02, 0.4 + np.pi, 0.060, 60.0, "block60_offset_open60_flipped_side"),
    ("pen", 0.004, 0.7 + np.pi / 2, 0.011, 30.0, "pen_offset_open30"),
    ("mug", 0.075, None, 0.004, 30.0, "mug_wall_grasp_2cm_below_rim"),
)
for name, h, yaw, wdt, opening, key in trials:
    kw = {"lift": 0.03, "pre_up": 0.03} if name == "mug" else {}
    OUT["grasp_trials"][key] = grasp_trial(name, h, yaw, width=wdt, opening=opening, **kw)
    print(key, OUT["grasp_trials"][key], flush=True)

print(json.dumps(OUT, indent=1))
with open("hull_and_grasp_bench_out.json", "w") as f:
    json.dump(OUT, f, indent=1)
