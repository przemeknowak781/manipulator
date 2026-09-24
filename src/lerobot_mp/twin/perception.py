"""Percepcja stanowiska z kalibrowanych kamer: mapa stolu na zywo i kostka na niej.

    mapper = TableMapper.from_workspace(ws)
    table, weight = mapper.fuse(hub.grab())             # (n, n, 3) mapa blatu z wszystkich kamer
    det = CubeDetector().detect(table, mapper)          # polozenie i obrot kostki w ukladzie podstawy

Mapa jest liczona na PLASZCZYZNIE, a poprawnie lezy tylko to, co jest na jej
wysokosci - to, co wystaje, rozmazuje sie od kamery (paralaksa). Dlatego do
szukania kostki mapa idzie na wysokosci jej GORNEJ sciany: gorna sciana jest
tym, co kamery widza najlepiej, i na tej wysokosci wszystkie kamery rysuja ja
w tym samym miejscu. Mapa na blacie (z = 0) jest do ogladania stanowiska.

Ta sama sciezka jedzie na kamerach symulowanych w blizniaku - wiec polityka
`lift` moze byc sprawdzona w symulacji na DOKLADNIE tej percepcji, ktora bedzie
miala na biurku, a nie na pozycji kostki wzietej z fizyki.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calib import topdown
from .workspace import Workspace


def _nelder_mead3(f, x0: np.ndarray, step: np.ndarray, iters: int = 90, tol: float = 1e-4):
    """Maly Nelder-Mead dla trzech parametrow (bez scipy)."""
    pts = [x0] + [x0 + np.eye(3)[i] * step[i] for i in range(3)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c)
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


class TableMapper:
    """Zszywanie kadrow w mape blatu; siatki rzutu liczone raz na poze kamery."""

    def __init__(self, cameras: dict[str, tuple[np.ndarray, np.ndarray | None, np.ndarray]],
                 centre_xy=(0.22, 0.0), side: float = 0.56, n: int = 224, table_z: float = 0.0):
        self.cameras = dict(cameras)
        self.centre, self.side, self.n, self.table_z = tuple(centre_xy), float(side), int(n), float(table_z)
        self._maps: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._heights: dict[float, TableMapper] = {}
        pts = topdown.grid(self.centre, self.side, self.n, self.table_z)
        for name, (K, dist, T) in self.cameras.items():
            self._maps[name] = topdown.warp_maps(K, dist, T, pts)

    @staticmethod
    def from_workspace(ws: Workspace, *, only_trusted: bool = True, **kw) -> TableMapper:
        cams = {}
        for c in ws.cameras:
            if not (c.enabled and c.calibrated) or (only_trusted and not c.trusted and c.source != "sim"):
                continue
            K, dist = c.intrinsics()
            cams[c.name] = (K, dist, np.asarray(c.T_cam2base, float))
        return TableMapper(cams, **kw)

    def at_height(self, z: float) -> TableMapper:
        """Ta sama mapa na innej wysokosci - siatki rzutu liczone raz (kilkanascie ms na kamere)."""
        key = round(float(z), 6)
        m = self._heights.get(key)
        if m is None:
            m = self._heights[key] = TableMapper(self.cameras, self.centre, self.side, self.n, z)
        return m

    def warp(self, frames: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Kazdy kadr osobno na plaszczyzne mapy: {kamera: (obraz (n, n, 3), waga (n, n))}."""
        out = {}
        for name, img in frames.items():
            if name not in self._maps or img is None:
                continue
            mx, my, w = self._maps[name]
            h, wid = img.shape[:2]
            w = w * ((mx >= 0) & (mx <= wid - 1) & (my >= 0) & (my <= h - 1))
            out[name] = (cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), w)
        return out

    def fuse(self, frames: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Mapa zszyta z wagami (kamera z gory liczy sie bardziej) i suma wag."""
        acc = np.zeros((self.n, self.n, 3), np.float32)
        wsum = np.zeros((self.n, self.n), np.float32)
        for warped, w in self.warp(frames).values():
            acc += warped.astype(np.float32) * w[..., None]
            wsum += w
        out = np.where(wsum[..., None] > 1e-6, acc / np.maximum(wsum[..., None], 1e-6), 0.0)
        return out.clip(0, 255).astype(np.uint8), wsum

    def to_base(self, u: float, v: float) -> np.ndarray:
        """Piksel mapy -> punkt (x, y, z mapy) w ukladzie podstawy."""
        s = self.side / self.n
        x0, y0 = self.centre[0] - self.side / 2, self.centre[1] + self.side / 2
        return np.array([x0 + s * (u + 0.5), y0 - s * (v + 0.5), self.table_z])

    @property
    def m_per_px(self) -> float:
        return self.side / self.n


@dataclass
class CubeDetection:
    pos: np.ndarray          # srodek kostki w ukladzie podstawy [m]
    rot: np.ndarray          # obrot (3, 3) - tylko wokol z
    area_px: float
    confidence: float
    #: Ile kamer realnie widzialo kostke w dopasowanej pozie; -1 = nie wiadomo (z mapy,
    #: `CubeDetector.detect`). Jedna kamera nie odrozni kostki podniesionej od lezacej
    #: dalej na tym samym promieniu, wiec dla trackera ponizej 2 (takze 0 i -1) to jeden swiadek.
    #: 0 to NIE "nie wiadomo": dopasowanie sylwetki daje 0, gdy ramie zaslania kostke kazdej
    #: kamerze w ponad polowie - czyli wlasnie przy kostce w szczekach.
    n_cameras: int = -1
    #: Chwila wykonania kadrow (time.monotonic), z ktorych jest detekcja; 0 = nie podano.
    t: float = 0.0


class CubeTracker:
    """Kostka dla polityki - takze wtedy, gdy kamery jej nie widza.

    Przy chwytaniu szczeki zaslaniaja gorna sciane, a podniesiona kostka
    schodzi z plaszczyzny mapy - wizja gubi ja dokladnie w najwazniejszym
    momencie. Wtedy:

    * widac ja - polozenie z kamer; gdy lezy miedzy szczekami, zapamietujemy
      jej poze wzgledem TCP;
    * nie widac, ale szczeka jest ZABLOKOWANA na czyms - serwo nie dojezdza
      do zadanego zamkniecia i stoi wyraznie szerzej niz przy pustym - a kostka
      byla ostatnio przy szczekach: jedzie z dlonia (TCP x zapamietana poza);
    * nie widac i nic nie trzyma - ostatnie polozenie przez `hold_s`, potem None.

    Zablokowana szczeka, a nie "rozkaz = zamknij do konca": nauczona polityka
    sciska kostke celem tylko troche ciasniejszym niz jej szerokosc - pierwsza
    wersja czekala na pelne zamkniecie i gubila kostke niesiona w powietrzu.

    Detekcja z JEDNEJ kamery (`n_cameras == 1`) przy dloni jest pomijana: jedna
    kamera nie odrozni kostki podniesionej w szczekach od kostki lezacej dalej na
    tym samym promieniu, a prog IoU odrzuca podniesiona kostke tylko wtedy, gdy
    dwie kamery sie nie zgadzaja. Zmierzone w blizniaku (lift-v2, tylko kamera
    `a`): podniesiona kostka przechodzila bramke z pewnoscia 0,87-0,98 i bledem
    3-40 cm, a tracker podawal ja polityce jako "kamery" zamiast "w dloni" -
    polityka wracala po nia na blat. Z dala od dloni kostka lezy na blacie
    (nic innego jej nie podnosi), wiec tam jedna kamera wystarcza.

    Wyjatek: kostka potracona chwytakiem. Lezy wtedy obok, jedna kamera widzi ja
    dobrze - a pomijanie zamrazalo ja w starym miejscu na `hold_s` (zmierzone:
    lift-v3, tylko kamera `a`, kostka wypadla ze szczek 3,5 cm dalej - polityka
    przez 130 taktow chwytala puste miejsce, potem "kamery jej nie widza").
    Pojedynczy swiadek przy dloni jest wiec przyjmowany, gdy szczeka nic nie trzyma
    (nie sciska, brak kostki w dloni), a `settle_n` kolejnych detekcji lezy w
    promieniu `settle_spread` od siebie, choc TCP przesunal sie w tym czasie
    w poziomie o `settle_move` (wiecej niz `settle_spread`). Duch kostki niesionej
    w szczekach tak nie umie: to rzut kostki wzdluz promienia kamery na blat, wiec
    jedzie w poziomie razem z dlonia (i szybciej od niej). Dlon stojaca w miejscu
    niczego nie dowodzi - wtedy zostaje stare polozenie, a zrodlo mowi, ze jedna
    kamera widzi kostke gdzie indziej.

    Kostka w dloni, a szczeka przestaje sciskac (zamknela sie na pustym albo
    otworzyla) - kostka wypadla. Nie wisi w powietrzu tam, gdzie byla w szczekach:
    zmierzone (lift-v3, kamera `a`) - "ostatnie widziane" 8 cm nad blatem po
    upuszczeniu, runner liczyl to jako podniesiona i konczyl "zadanie wykonane"
    z kostka na blacie. Teraz: ostatnie polozenie opuszczone na blat (pod dlonia).
    """

    def __init__(self, hold_s: float = 6.0, grab_radius: float = 0.06, grip_margin: float = 0.05,
                 block_margin: float = 0.05, near_radius: float = 0.10, confirm_dist: float = 0.01,
                 settle_n: int = 3, settle_spread: float = 0.01, settle_move: float = 0.015,
                 settle_window: float = 1.5, drop_height: float = 0.01):
        self.hold_s, self.grab_radius = hold_s, grab_radius
        self.grip_margin, self.block_margin = grip_margin, block_margin
        #: "Przy dloni": TCP blizej kostki (ostatniej albo wykrytej) niz tyle [m].
        self.near_radius = near_radius
        #: Pominieta detekcja blizej ostatniego polozenia niz tyle [m] tylko je potwierdza.
        self.confirm_dist = confirm_dist
        #: Pojedynczy swiadek przy dloni przyjety, gdy `settle_n` detekcji z ostatnich
        #: `settle_window` s lezy parami blizej niz `settle_spread` [m], a TCP w tym czasie
        #: przejechal w poziomie co najmniej `settle_move` [m]. Rozrzut polozenia kostki
        #: lezacej z jednej kamery w blizniaku: 1-3 mm, toczacej sie jeszcze po upadku do 9 mm.
        self.settle_n, self.settle_spread = settle_n, settle_spread
        self.settle_move, self.settle_window = settle_move, settle_window
        #: Kostka upuszczona wyzej niz tyle [m] nad polozeniem lezacej - opuszczana na blat.
        self.drop_height = drop_height
        self.last: tuple[np.ndarray, np.ndarray] | None = None
        self.t_last = -np.inf
        self.in_hand: np.ndarray | None = None          # poza kostki w ukladzie TCP
        self.source = "brak"
        self._grip_prev: float | None = None
        self._witness: list[tuple[float, np.ndarray, np.ndarray]] = []   # (t, kostka, TCP) pominiete
        self._rest_z: float | None = None               # wysokosc srodka kostki lezacej (z kamer)
        self._carried = False                           # byla w dloni, szczeka dalej sciska
        #: Dopisek do "ostatnie widziane": pominieta detekcja widzi kostke gdzie indziej
        #: albo kostka wypadla z dloni - panel mowi to wprost zamiast cicho trzymac stare.
        self._note = ""

    def _single_camera_near_hand(self, det: CubeDetection, T_tcp: np.ndarray, squeezing: bool) -> bool:
        """Czy detekcja moze byc podniesiona kostka, ktora jedna kamera "polozyla" na blacie."""
        if det.n_cameras >= 2:
            return False
        if self.in_hand is not None or squeezing:
            return True
        tcp = T_tcp[:3, 3]
        near = np.linalg.norm(det.pos - tcp) < self.near_radius
        if self.last is not None:
            near |= np.linalg.norm(self.last[0] - tcp) < self.near_radius
        return bool(near)

    def _settled(self, det: CubeDetection, T_tcp: np.ndarray, squeezing: bool, now: float) -> bool:
        """Czy pojedynczy swiadek przy dloni to kostka LEZACA: stoi, choc dlon jedzie."""
        if self.in_hand is not None or squeezing:
            self._witness.clear()
            return False
        new = (now, np.asarray(det.pos, float).copy(), T_tcp[:3, 3].copy())
        # Ciagly ogon swiadkow, ktorzy PARAMI leza blizej niz `settle_spread` - duch, ktory
        # "wraca" z dlonia w to samo miejsce, i tak przerywa ciag po drodze.
        keep = [new]
        for w in reversed(self._witness):
            if now - w[0] > self.settle_window \
                    or any(np.linalg.norm(w[1] - k[1]) > self.settle_spread for k in keep):
                break
            keep.append(w)
        self._witness = keep[::-1]
        if len(self._witness) < self.settle_n:
            return False
        # W POZIOMIE: dlon jadaca tylko w gore/dol nad kamera patrzaca z gory przesuwa
        # ducha kostki w szczekach o ulamek swojego ruchu - moglby "stac".
        xy = np.array([w[2][:2] for w in self._witness])
        if np.linalg.norm(xy[:, None] - xy[None], axis=2).max() < self.settle_move:
            return False
        self._witness.clear()
        return True

    def _dropped(self, now: float) -> None:
        """Kostka byla w dloni, szczeka juz nic nie trzyma - lezy na blacie pod miejscem upadku."""
        if self.last is None or self._rest_z is None or self.last[0][2] < self._rest_z + self.drop_height:
            return
        p = self.last[0].copy()
        p[2] = self._rest_z
        # Po upadku zostaje tylko obrot wokol pionu (kostka lezy na scianie).
        yaw = np.arctan2(self.last[1][1, 0], self.last[1][0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        self.last = (p, np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))
        self.t_last, self._note = now, "upuszczona"

    def update(self, det: CubeDetection | None, T_tcp: np.ndarray, grip_q: float, grip_cmd: float,
               grip_closed: float, now: float) -> tuple[np.ndarray, np.ndarray] | None:
        # Zablokowana = nie dojezdza do rozkazu I STOI. Szczeka w trakcie zamykania tez
        # odstaje od rozkazu, ale jeszcze jedzie - bez warunku postoju "trzymanie"
        # wlaczalo sie, zanim szczeka w ogole dotknela kostki.
        still = self._grip_prev is not None and abs(grip_q - self._grip_prev) < 0.02
        self._grip_prev = grip_q
        squeezing = grip_q - grip_cmd > self.block_margin and grip_q > grip_closed + self.grip_margin
        holding = still and squeezing
        if det is not None and self._single_camera_near_hand(det, T_tcp, squeezing):
            if not self._settled(det, T_tcp, squeezing, now):
                # Zgodna z ostatnim polozeniem (kostka lezy, dlon nad nia) - tylko odswieza jego
                # waznosc, zeby kostka nie "znikala" po `hold_s`, gdy dlon dlugo nad nia krazy.
                if self.in_hand is None and self.last is not None:
                    if np.linalg.norm(det.pos - self.last[0]) < self.confirm_dist:
                        self.t_last, self._note = now, ""
                    elif self._note != "upuszczona":
                        self._note = "1 kamera widzi ja gdzie indziej"
                det = None
        elif det is not None:
            self._witness.clear()
        if det is not None and not (holding and self.in_hand is not None):
            self.last, self.t_last, self.source = (det.pos, det.rot), now, "kamery"
            self._rest_z, self._note, self._carried = float(det.pos[2]), "", False
            if not holding:
                self.in_hand = None
            return self.last
        if holding:
            # Poza w dloni zapisywana w CHWILI chwytu, z ostatniej detekcji: kostka lezala
            # wtedy nieruchomo, a kamery i tak juz jej nie widza - szczeki zaslaniaja gorna
            # sciane, zanim TCP zejdzie do niej blizej niz kilka centymetrow.
            if self.in_hand is None and self.last is not None \
                    and np.linalg.norm(self.last[0] - T_tcp[:3, 3]) < self.grab_radius:
                T_cube = np.eye(4)
                T_cube[:3, :3], T_cube[:3, 3] = self.last[1], self.last[0]
                self.in_hand = np.linalg.inv(T_tcp) @ T_cube
            if self.in_hand is not None:
                T = T_tcp @ self.in_hand
                self.last, self.t_last, self.source = (T[:3, 3].copy(), T[:3, :3].copy()), now, "w dloni"
                self._note, self._carried = "", False
                return self.last
        else:
            # Szczeka stoi ciasniej niz kostka (albo sie otworzyla) - w dloni nic nie ma.
            # Samo "jeszcze jedzie" przy sciskaniu (drgniecie serwa, kostka sie przekreca)
            # to jeszcze NIE upadek - ale zapamietane: w blizniaku szczeka po takim takcie
            # zamykala sie na pustym w nastepnym i upadek przechodzil niezauwazony.
            if self.in_hand is not None or self._carried:
                if squeezing:
                    self._carried = True
                else:
                    self._dropped(now)
                    self._carried = False
            self.in_hand = None
        if self.last is not None and now - self.t_last < self.hold_s:
            self.source = f"ostatnie widziane ({self._note})" if self._note else "ostatnie widziane"
            return self.last
        self.source = "brak"
        return None


@dataclass
class CubeDetector:
    """Kostka po kolorze na mapie blatu. Progi HSV w konwencji OpenCV (H 0..180)."""

    hsv_lo: tuple[int, int, int] = (0, 120, 70)
    hsv_hi: tuple[int, int, int] = (12, 255, 255)
    #: Czerwien przechodzi przez H = 0, wiec drugi zakres z drugiej strony kola.
    hsv_lo2: tuple[int, int, int] = (170, 120, 70)
    hsv_hi2: tuple[int, int, int] = (180, 255, 255)
    cube_half: float = 0.015
    #: Najmniejsze pokrycie sylwetki (IoU), przy ktorym kostka "lezy na blacie tam, gdzie mowimy".
    min_iou: float = 0.75
    #: Kamera liczy sie do `n_cameras`, gdy widzi (nie zaslonieta ramieniem) co najmniej
    #: taka czesc przewidzianej sylwetki.
    min_visible: float = 0.5

    def mask(self, table: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(table, cv2.COLOR_RGB2HSV)
        m = cv2.inRange(hsv, np.array(self.hsv_lo), np.array(self.hsv_hi))
        m |= cv2.inRange(hsv, np.array(self.hsv_lo2), np.array(self.hsv_hi2))
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def top_mask(self, frames: dict[str, np.ndarray], mapper: TableMapper,
                 occluders: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Gorna sciana kostki: czesc wspolna masek koloru z kamer, ktore widza dany punkt.

        Mapa na wysokosci gornej sciany rysuje ja w tym samym miejscu z kazdej
        kamery, a boki kostki - w roznych, odsuniete od kazdej kamery. Czesc
        wspolna zostawia wiec sama gorna sciane; srednia ze wszystkich kamer
        (jak w `fuse`) zostawialaby sciane plus rozmazane boki - 12-17 mm bledu
        srodka przy dwoch kamerach w symulacji.

        `occluders[kamera]` - maska (H, W) pikseli ZASLONIETYCH ramieniem (z blizniaka).
        Tam kamera nie glosuje: ramie przed kostka to "nie wiem", nie "tu nie ma kostki".
        Bez tego ramie zaslaniajace kostke jednej kamerze gasilo ja w czesci wspolnej.
        """
        seen = np.zeros((mapper.n, mapper.n), np.int32)
        red = np.zeros((mapper.n, mapper.n), np.int32)
        for name, (warped, w) in mapper.warp(frames).items():
            valid = w > 1e-6
            if occluders and name in occluders:
                occ = mapper.warp({name: occluders[name].astype(np.uint8) * 255})[name][0]
                valid &= (occ[..., 0] if occ.ndim == 3 else occ) < 128
            seen += valid
            red += valid & (self.mask(warped) > 0)
        top = (seen > 0) & (red == seen)
        return cv2.morphologyEx(top.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def detect_frames(self, frames: dict[str, np.ndarray], mapper: TableMapper,
                      occluders: dict[str, np.ndarray] | None = None, t: float = 0.0) -> CubeDetection | None:
        """Kostka z kadrow kamer: start z czesci wspolnej masek, potem dopasowanie sylwetki.

        Czesc wspolna na plaszczyznie gornej sciany potrzebuje dwoch kamer, ktore
        widza kostke - gdy ramie zaslania ja jednej, zostaje plama ze scianami
        bocznymi i pewnosc spada do zera (4 z 8 epizodow `lift` w symulacji tracilo
        kostke wlasnie tak). Dopasowanie sylwetki bryly do masek wszystkich kamer
        dziala z jedna kamera i z wieloma - boki kostki sa czescia modelu.

        `t` - chwila wykonania kadrow; trafia do detekcji, zeby konsument wiedzial,
        jak stara jest (dopasowanie trwa, a kadry bywaja sprzed kilku taktow).
        """
        top = mapper if abs(mapper.table_z - 2 * self.cube_half) < 1e-6 else mapper.at_height(2 * self.cube_half)
        init = self._from_mask(self.top_mask(frames, top, occluders), top)
        masks = {n: self.mask(img) > 0 for n, img in frames.items() if n in mapper.cameras}
        start = None if init is None else (init.pos[0], init.pos[1], np.arctan2(init.rot[1, 0], init.rot[0, 0]))
        if start is None:
            start = self._ray_guess(masks, mapper.cameras, occluders)
        if start is None:
            return None
        det = self.fit_silhouette(masks, mapper.cameras, start, occluders)
        if det is not None:
            det.t = float(t)
        return det

    # ------------------------------------------------------ dopasowanie sylwetki
    def _corners(self, x: float, y: float, yaw: float) -> np.ndarray:
        h = self.cube_half
        c, s = np.cos(yaw), np.sin(yaw)
        local = np.array([[sx * h, sy * h, sz * h] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return local @ R.T + np.array([x, y, h])

    def _ray_guess(self, masks, cameras, occluders) -> tuple[float, float, float] | None:
        """Srodek plamy z kazdej kamery rzutowany na wysokosc srodka kostki - zgrubny start."""
        pts = []
        for n, m in masks.items():
            vis = m & ~occluders[n] if occluders and n in occluders else m
            if vis.sum() < 20:
                continue
            v, u = np.nonzero(vis)
            K, dist, T = cameras[n]
            ray_c = np.linalg.inv(K) @ np.array([u.mean(), v.mean(), 1.0])
            T = np.asarray(T, float)
            d, o = T[:3, :3] @ ray_c, T[:3, 3]
            if abs(d[2]) < 1e-6:
                continue
            s = (self.cube_half - o[2]) / d[2]
            if s > 0:
                pts.append(o + s * d)
        if not pts:
            return None
        p = np.mean(pts, axis=0)
        return float(p[0]), float(p[1]), 0.0

    def fit_silhouette(self, masks: dict[str, np.ndarray], cameras, start, occluders=None,
                       scale: float = 1.0) -> CubeDetection | None:
        """(x, y, obrot) kostki lezacej na blacie, przy ktorych jej rzut najlepiej pokrywa maski kamer."""
        prepared = []
        # Wszystko liczymy w WYCINKU kadru wokol startu: kostka to kilkadziesiat pikseli,
        # a pelny kadr kosztowal ~40 ms na detekcje - i trzymal GIL, spowalniajac petle ramienia.
        around = self._corners(start[0], start[1], 0.0)
        around = np.vstack([around + [dx, dy, 0.0] for dx in (-0.04, 0.04) for dy in (-0.04, 0.04)])
        for n, m in masks.items():
            K, dist, T = cameras[n]
            Ti = np.linalg.inv(np.asarray(T, float))
            dist = None if dist is None else np.asarray(dist, float)
            if ((around @ Ti[:3, :3].T + Ti[:3, 3])[:, 2] < 0.05).any():
                continue
            px, _ = cv2.projectPoints(around, cv2.Rodrigues(Ti[:3, :3])[0], Ti[:3, 3], np.asarray(K, float), dist)
            px = px.reshape(-1, 2)
            H, W = m.shape
            x0, y0 = max(0, int(px[:, 0].min()) - 10), max(0, int(px[:, 1].min()) - 10)
            x1, y1 = min(W, int(px[:, 0].max()) + 10), min(H, int(px[:, 1].max()) + 10)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            crop = m[y0:y1, x0:x1]
            valid = np.ones_like(crop)
            if occluders and n in occluders:
                valid = ~occluders[n][y0:y1, x0:x1]
            small = cv2.resize(crop.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) > 0
            valid = cv2.resize(valid.astype(np.uint8), (small.shape[1], small.shape[0]),
                               interpolation=cv2.INTER_NEAREST) > 0
            Ks = np.asarray(K, float).copy()
            Ks[0, 2] -= x0
            Ks[1, 2] -= y0
            Ks[:2] *= scale
            prepared.append((small & valid, valid, Ks, dist, Ti))
        if not prepared:
            return None

        def stats(p) -> tuple[float, int, int]:
            """(suma IoU, ile kamer liczy sie do sredniej, ile widzi wiekszosc sylwetki)."""
            corners = self._corners(*p)
            total, n_used, n_seen = 0.0, 0, 0
            for obs, valid, Ks, dist, Ti in prepared:
                pc = corners @ Ti[:3, :3].T + Ti[:3, 3]
                if (pc[:, 2] < 0.05).any():
                    continue
                px, _ = cv2.projectPoints(corners, cv2.Rodrigues(Ti[:3, :3])[0], Ti[:3, 3], Ks, dist)
                hull = cv2.convexHull(px.reshape(-1, 2).astype(np.float32)).astype(np.int32)
                full = np.zeros(obs.shape, np.uint8)
                cv2.fillConvexPoly(full, hull, 1)
                pred = (full > 0) & valid
                union = (pred | obs).sum()
                if union == 0 or pred.sum() < 4:
                    continue
                total += (pred & obs).sum() / union
                n_used += 1
                # Kamera, ktorej ramie zaslania prawie cala przewidziana sylwetke, glosuje
                # kilkoma pikselami - nie jest drugim, niezaleznym swiadkiem polozenia.
                n_seen += int(pred.sum() >= self.min_visible * full.sum())
            return total, n_used, n_seen

        def score(p) -> float:
            total, n_used, _ = stats(p)
            return -total / n_used if n_used else 0.0

        best, best_s = np.asarray(start, float), score(start)
        # Kilka startow obrotu (symetria 90 st.), potem Nelder-Mead po (x, y, obrot).
        for yaw in np.radians([0, 22.5, 45, 67.5]):
            p = np.array([start[0], start[1], start[2] + yaw])
            s = score(p)
            if s < best_s:
                best, best_s = p, s
        best, best_s = _nelder_mead3(score, best, np.array([0.006, 0.006, np.radians(12)]), iters=90)
        iou = -best_s
        # Model zaklada kostke LEZACA na blacie. Kostke w powietrzu (w szczekach) da sie
        # "wcisnac" w jakas poze na blacie, ktora czesciowo pasuje - zmierzone wzdluz
        # epizodow lift: prawdziwe detekcje na blacie IoU 0,88-0,96, falszywe przy
        # podniesionej kostce 0,36-0,64 (i 26-970 mm bledu). Prog miedzy nimi.
        # Dziala tylko wtedy, gdy kostke widza co najmniej DWIE kamery: jednej kamerze
        # kostke podniesiona o 2-6 cm zastepuje lezaca dalej na tym samym promieniu
        # (IoU 0,96-0,82 przy 26-92 mm bledu) - stad `n_cameras` w detekcji.
        if iou < self.min_iou:
            return None
        n_seen = stats(best)[2]
        yaw = (best[2] + np.pi / 4) % (np.pi / 2) - np.pi / 4
        c_, s_ = np.cos(yaw), np.sin(yaw)
        R = np.array([[c_, -s_, 0.0], [s_, c_, 0.0], [0.0, 0.0, 1.0]])
        return CubeDetection(np.array([best[0], best[1], self.cube_half]), R, float(iou), float(iou), n_seen)

    def detect(self, table: np.ndarray, mapper: TableMapper) -> CubeDetection | None:
        """Kostka na gotowej mapie (np. z jednej kamery) - mniej dokladnie niz `detect_frames`."""
        return self._from_mask(self.mask(table), mapper)

    def _from_mask(self, m: np.ndarray, mapper: TableMapper) -> CubeDetection | None:
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        expected = (2 * self.cube_half / mapper.m_per_px) ** 2
        if area < 0.25 * expected:
            return None
        (u, v), _, angle = cv2.minAreaRect(c)
        p = mapper.to_base(u, v)
        p[2] = self.cube_half
        # Obrot wokol z: kat prostokata na mapie (v rosnie w dol = -y), kostka ma symetrie 90 st.
        yaw = -np.radians(angle)
        yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        c_, s_ = np.cos(yaw), np.sin(yaw)
        R = np.array([[c_, -s_, 0.0], [s_, c_, 0.0], [0.0, 0.0, 1.0]])
        conf = float(np.clip(1.0 - abs(area / expected - 1.0), 0.0, 1.0))
        return CubeDetection(p, R, area, conf)
