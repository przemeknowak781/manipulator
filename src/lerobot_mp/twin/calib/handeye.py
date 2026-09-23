"""Eye-to-hand calibration by reprojection: cameras fixed, tags on the arm.

Przeniesione z galaxeo-manipulators `sim/calib/handeye.py` (commit 02641c4)
i rozszerzone z jednej kamery na wiele. Matematyka jest ta sama: Levenberg-
Marquardt na bledzie reprojekcji kazdego rogu kazdego taga w kazdym kadrze,
start z zamknietego AX = XB (Park i Martin) albo z pozy nominalnej, bramki na
residuum i na rozrzut obrotow. Doszly dwie rzeczy:

* **Wiele kamer, jedna karta.** Karta jedzie w dloni i jest ta sama dla
  wszystkich kamer, wiec jej poza w chwytaku jest wspolnym parametrem. Kamera,
  ktora widzi karte dobrze, pomaga wtedy tej, ktora widzi ja slabiej: zamiast
  C osobnych problemow po 12 stopni swobody jest jeden o 6C + 6.
* **Residua liczone wektorowo.** Jedno `projectPoints` na kamere zamiast na
  obserwacje - przy kilku kamerach i ~100 obserwacjach petla pythonowa byla
  waskim gardlem calego dopasowania.

    from lerobot_mp.twin.calib.handeye import Observation, solve
    fit = solve(obs, {"front": (K, dist)}, tag_size, {"card": T_card2tcp_nominal})
    fit.cameras["front"].T_cam2base, fit.mounts["card"], fit.rms_px

Every observation is one tag seen in one image while the arm stood still:

    Observation(T_frame2base, T_tag2mount, corners, tag_id, mount, camera)

`T_frame2base` is the pose of the body the mount rides (from forward kinematics
at the measured joint angles), `T_tag2mount` the tag's pose *within* the mount,
which is exact printed geometry, and `corners` the four detected pixel corners.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .tags import corners_in_tag, tag_pose


@dataclass
class Observation:
    T_frame2base: np.ndarray
    T_tag2mount: np.ndarray
    corners: np.ndarray
    tag_id: int = -1
    mount: str = "card"
    camera: str = "cam"


@dataclass
class CameraFit:
    """Wynik dla jednej kamery."""

    T_cam2base: np.ndarray
    rms_px: float
    max_px: float
    n_obs: int
    #: Rozrzut orientacji karty widzianej przez te kamere, patrz `rotation_spread_R`.
    spread_deg: float
    trusted: bool = False
    reason: str = ""


@dataclass
class Fit:
    cameras: dict[str, CameraFit]
    mounts: dict[str, np.ndarray] = field(default_factory=dict)   # mount name -> T_mount2frame
    rms_px: float = float("nan")
    max_px: float = float("nan")
    n_obs: int = 0
    init: str = ""                                                 # which starting point won

    @property
    def trusted(self) -> bool:
        return bool(self.cameras) and all(c.trusted for c in self.cameras.values())


# ------------------------------------------------------------ SE(3) helpers
def _se3(xi):
    """Exponential map of a 6-vector (rotation vector, translation) to 4x4."""
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(np.asarray(xi[:3], float))
    T[:3, 3] = xi[3:]
    return T


def _log(T):
    return np.concatenate([cv2.Rodrigues(np.asarray(T)[:3, :3])[0].ravel(), np.asarray(T)[:3, 3]])


def _inv(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def _median_T(Ts):
    """Elementwise median of a list of transforms, re-orthonormalised."""
    t = np.median([T[:3, 3] for T in Ts], axis=0)
    rv = np.array([cv2.Rodrigues(T[:3, :3])[0].ravel() for T in Ts])
    R, _ = cv2.Rodrigues(np.median(rv, axis=0))
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = R, t
    return out


def rotation_spread_R(Rs):
    """How much a set of orientations turned, in the worst direction, in degrees.

    Hand-eye is only conditioned if the carrier rotated about several
    non-parallel axes: a wave that only yaws leaves the camera pose free to
    slide along that axis, and the fit comes back confident and wrong. Stack
    the rotation vectors of every pair of orientations, take the smallest
    singular value of the (3, N) matrix normalised by sqrt(N), and you get the
    rotation available about the least-covered axis. Turning about one axis
    only scores 0, however far it turns."""
    Rs = list(Rs)
    if len(Rs) < 2:
        return 0.0
    v = [cv2.Rodrigues(Rs[i] @ Rs[j].T)[0].ravel() for i in range(len(Rs)) for j in range(i + 1, len(Rs))]
    M = np.array(v).T                                  # (3, N) rotation vectors
    # eigenvalues of M M^T rather than the SVD of M: with fewer than three
    # pairs the SVD returns fewer than three singular values and the smallest
    # of them is not the smallest axis.
    lam = np.linalg.eigvalsh(M @ M.T / M.shape[1])
    return float(np.degrees(np.sqrt(max(lam[0], 0.0))))


def rotation_spread(obs):
    """`rotation_spread_R` over the distinct frames of a list of observations."""
    Rs, seen = [], set()
    for o in obs:
        key = o.T_frame2base.tobytes()
        if key not in seen:                     # dwa tagi z jednego kadru to jedna orientacja
            seen.add(key)
            Rs.append(o.T_frame2base[:3, :3])
    return rotation_spread_R(Rs)


# ---------------------------------------------------------------- residuals
class _Problem:
    """Obserwacje ulozone w tablice raz, zeby kazde residuum bylo kilkoma mnozeniami."""

    def __init__(self, obs, cameras, tag_size, names_cam, names_mount):
        self.names_cam, self.names_mount = names_cam, names_mount
        X = np.c_[corners_in_tag(tag_size), np.ones(4)].T            # (4 wsp., 4 rogi)
        self.A = np.array([o.T_frame2base for o in obs])              # (N, 4, 4)
        self.B = np.array([o.T_tag2mount @ X for o in obs])           # (N, 4, 4) rogi w montazu
        self.cam = np.array([names_cam.index(o.camera) for o in obs])
        self.mount = np.array([names_mount.index(o.mount) for o in obs])
        self.corners = np.array([np.asarray(o.corners, float) for o in obs])   # (N, 4, 2)
        self.K = [np.asarray(cameras[c][0], float) for c in names_cam]
        self.dist = [np.zeros(5) if cameras[c][1] is None else np.asarray(cameras[c][1], float)
                     for c in names_cam]

    def unpack(self, p):
        nc = len(self.names_cam)
        cams = [_se3(p[6 * k:6 * k + 6]) for k in range(nc)]
        mounts = [_se3(p[6 * (nc + k):6 * (nc + k) + 6]) for k in range(len(self.names_mount))]
        return cams, mounts

    def residuals(self, p):
        cams, mounts = self.unpack(p)
        M = np.array(mounts)[self.mount]                              # (N, 4, 4)
        P = np.einsum("nij,njk,nkl->nil", self.A, M, self.B)          # rogi w ukladzie podstawy
        out = np.empty((len(self.A), 4, 2))
        for k, T_cam2base in enumerate(cams):
            sel = self.cam == k
            if not sel.any():
                continue
            pts = np.einsum("ij,njl->nil", _inv(T_cam2base), P[sel])[:, :3, :]   # (n, 3, 4)
            pts = pts.transpose(0, 2, 1).reshape(-1, 3)
            px, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), self.K[k], self.dist[k])
            out[sel] = px.reshape(-1, 4, 2)
        return (out - self.corners).ravel()


def _params(cams, mounts):
    return np.concatenate([_log(T) for T in list(cams) + list(mounts)])


# ------------------------------------------------------------------- starts
def mount_poses_in_cam(obs, K, dist, tag_size):
    """Per-observation PnP, lifted from the tag to its mount frame."""
    return [tag_pose(o.corners, tag_size, K, dist) @ _inv(o.T_tag2mount) for o in obs]


MIN_PAIR_DEG = 10.0            # relative rotations smaller than this carry no information


def solve_axxb(A, B):
    """Closed-form AX = XB (Park and Martin), the classic hand-eye step.

    `cv2.calibrateHandEye` is gone in OpenCV 5, so this is the same method
    written out. Rotation first: for A X = X B the rotation vectors obey
    R_X log(R_B) = log(R_A), which is an orthogonal Procrustes problem over
    every motion pair. Translation follows linearly from
    (R_A - I) t_X = R_X t_B - t_A, stacked and solved in least squares."""
    a = np.array([cv2.Rodrigues(T[:3, :3])[0].ravel() for T in A])
    b = np.array([cv2.Rodrigues(T[:3, :3])[0].ravel() for T in B])
    U, _, Vt = np.linalg.svd(a.T @ b)
    R_X = U @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))]) @ Vt
    M = np.vstack([TA[:3, :3] - np.eye(3) for TA in A])
    rhs = np.concatenate([R_X @ TB[:3, 3] - TA[:3, 3] for TA, TB in zip(A, B)])
    t_X = np.linalg.lstsq(M, rhs, rcond=None)[0]
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R_X, t_X
    return X


def handeye_init(obs, K, dist, tag_size, names, max_pairs=400):
    """Closed-form start for ONE camera, in the eye-to-hand arrangement.

    The camera is fixed in the base and the card rides the gripper, so what is
    constant across the wave is the card's pose in the gripper:

        T_card2gripper = T_base2gripper_i . T_cam2base . T_card2cam_i

    Equate two poses i and j and the unknowns separate into the standard form
    A X = X B with X = T_cam2base, A = FK_j FK_i^-1 the arm's relative motion
    and B = T_card2cam_j T_card2cam_i^-1 the card's motion as the camera saw
    it. The card's pose in the gripper then follows one per observation, and
    their median is the start."""
    T_m2c = mount_poses_in_cam(obs, K, dist, tag_size)
    pairs = [(i, j) for i in range(len(obs)) for j in range(i + 1, len(obs))]
    A, B = [], []
    for i, j in pairs:
        a = obs[j].T_frame2base @ _inv(obs[i].T_frame2base)
        if np.linalg.norm(cv2.Rodrigues(a[:3, :3])[0]) < np.radians(MIN_PAIR_DEG):
            continue
        A.append(a)
        B.append(T_m2c[j] @ _inv(T_m2c[i]))
        if len(A) >= max_pairs:
            break
    if len(A) < 3:
        raise ValueError("too few distinct rotations for the closed-form hand-eye")
    T_cam2base = solve_axxb(A, B)
    mounts = {}
    for n in names:
        Ts = [_inv(o.T_frame2base) @ T_cam2base @ T for o, T in zip(obs, T_m2c) if o.mount == n]
        if Ts:
            mounts[n] = _median_T(Ts)
    return T_cam2base, mounts


def nominal_init(obs, K, dist, tag_size, mounts_nominal):
    """Start for ONE camera from the rough nominal mount: each observation then
    gives a camera pose, and their median is the start."""
    T_m2c = mount_poses_in_cam(obs, K, dist, tag_size)
    Ts = [o.T_frame2base @ mounts_nominal[o.mount] @ _inv(T) for o, T in zip(obs, T_m2c)]
    return _median_T(Ts), dict(mounts_nominal)


def _lm(p, problem, iters=60):
    lam = 1e-3
    r = problem.residuals(p)
    cost = float(r @ r)
    step = np.zeros_like(p)
    for _ in range(iters):
        J = np.empty((r.size, p.size))
        for k in range(p.size):
            dp = np.zeros_like(p)
            dp[k] = 1e-6
            J[:, k] = (problem.residuals(p + dp) - r) / 1e-6
        H, g = J.T @ J, J.T @ r
        improved = False
        for _ in range(10):
            step = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), -g)
            r_new = problem.residuals(p + step)
            c_new = float(r_new @ r_new)
            if c_new < cost:
                p, r, cost, lam, improved = p + step, r_new, c_new, max(lam / 3, 1e-9), True
                break
            lam *= 10
        if not improved or np.linalg.norm(step) < 1e-10:
            break
    return p, r


def _fit(p, r, problem, obs, tag=""):
    per_corner = np.linalg.norm(r.reshape(-1, 2), axis=1).reshape(-1, 4)   # (N, 4)
    cams, mounts = problem.unpack(p)
    out = {}
    for k, name in enumerate(problem.names_cam):
        sel = problem.cam == k
        e = per_corner[sel].ravel()
        mine = [o for o, s in zip(obs, sel) if s]
        out[name] = CameraFit(cams[k], float(np.sqrt(np.mean(e**2))) if e.size else float("inf"),
                              float(e.max()) if e.size else float("inf"), int(sel.sum()),
                              rotation_spread(mine))
    e = per_corner.ravel()
    return Fit(out, dict(zip(problem.names_mount, mounts)), float(np.sqrt(np.mean(e**2))),
               float(e.max()), len(obs), tag)


def solve(obs, cameras, tag_size, mounts_nominal, iters=60, restarts=5, max_px=1.5, seed=0):
    """Levenberg-Marquardt on the reprojection error, from the best of several starts.

    `cameras` maps every camera name appearing in `obs` to `(K, dist)`;
    `mounts_nominal` maps every mount name to a rough guess at its pose in the
    frame it rides on. Every camera pose and every mount pose is solved for,
    jointly. Starts: each camera from its own closed-form hand-eye where that
    has enough distinct rotations (the mounts from the median over cameras),
    and everything from the nominal. The lower residual wins; if that is still
    above `max_px` the fit is restarted from jittered versions of it, because
    on a thin session LM can settle in a mirrored minimum with a residual of
    hundreds of pixels."""
    obs = list(obs)
    names_cam = sorted({o.camera for o in obs})
    names_mount = sorted({o.mount for o in obs})
    for group, have in (("camera", cameras), ("mount", mounts_nominal)):
        wanted = names_cam if group == "camera" else names_mount
        missing = [n for n in wanted if n not in have]
        if missing:
            raise ValueError(f"no {group} entry for {missing}")
    problem = _Problem(obs, cameras, tag_size, names_cam, names_mount)

    by_cam = {c: [o for o in obs if o.camera == c] for c in names_cam}
    nominal_cams, eye_cams, eye_mounts = {}, {}, {}
    for c, mine in by_cam.items():
        K, dist = cameras[c]
        nominal_cams[c], _ = nominal_init(mine, K, dist, tag_size, mounts_nominal)
        try:
            T, m = handeye_init(mine, K, dist, tag_size, names_mount)
        except (ValueError, cv2.error, np.linalg.LinAlgError):
            continue                   # too few distinct rotations for the closed form
        eye_cams[c] = T
        for n, Tm in m.items():
            eye_mounts.setdefault(n, []).append(Tm)

    starts = [("nominal", [nominal_cams[c] for c in names_cam],
               [mounts_nominal[n] for n in names_mount])]
    if eye_cams:
        starts.insert(0, ("handeye", [eye_cams.get(c, nominal_cams[c]) for c in names_cam],
                          [_median_T(eye_mounts[n]) if n in eye_mounts else mounts_nominal[n]
                           for n in names_mount]))

    best = None
    for tag, cams0, mounts0 in starts:
        p, r = _lm(_params(cams0, mounts0), problem, iters)
        fit = _fit(p, r, problem, obs, tag)
        if best is None or fit.rms_px < best.rms_px:
            best = fit
    if best.rms_px > max_px:
        rng = np.random.default_rng(seed)
        for i in range(restarts):
            cams0 = []
            for c in names_cam:
                T = best.cameras[c].T_cam2base.copy()
                T[:3, 3] += rng.normal(0, 0.05, 3)
                cams0.append(T)
            p, r = _lm(_params(cams0, [best.mounts[n] for n in names_mount]), problem, iters)
            fit = _fit(p, r, problem, obs, f"jitter{i}")
            if fit.rms_px < best.rms_px:
                best = fit
            if best.rms_px <= max_px:
                break
    return best


def judge(fit, max_px=1.5, min_spread=8.0, min_obs=14):
    """Bramki galaxeo, osobno dla kazdej kamery.

    Kamera nie jest zaufana, gdy residuum jest za duze, gdy karta widziana
    przez nia obracala sie za malo wokol ktorejs osi (dopasowanie jest wtedy
    pewne siebie i bledne), albo gdy obserwacji jest za malo. Werdykt ma powod,
    a nie tylko "nie" - uzytkownik musi wiedziec, co poprawic w nastepnej fali.
    """
    for cam in fit.cameras.values():
        reasons = []
        if cam.rms_px > max_px:
            reasons.append(f"residuum {cam.rms_px:.2f} px > {max_px} px")
        if cam.spread_deg < min_spread:
            reasons.append(f"rozrzut obrotow {cam.spread_deg:.1f} st. < {min_spread} st. (zdegenerowana fala)")
        if cam.n_obs < min_obs:
            reasons.append(f"tylko {cam.n_obs} obserwacji, potrzeba {min_obs}")
        cam.trusted, cam.reason = not reasons, "; ".join(reasons)
    return fit


def pose_error(T_a, T_b):
    """(translation error [m], rotation error [rad]) between two transforms."""
    dT = _inv(T_a) @ T_b
    ang = np.linalg.norm(cv2.Rodrigues(dT[:3, :3])[0])
    return float(np.linalg.norm(dT[:3, 3])), float(ang)
