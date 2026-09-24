"""Czy kamera sie przestawila od kalibracji - kadr teraz kontra kadr zapamietany po niej.

Kalibracja jest prawdziwa tylko dopoty, dopoki kamera stoi tam, gdzie stala.
Potracenie statywu o 1 st. to przy f = 600 px ok. 10 px przesuniecia kadru -
mapa stolu rozjezdza sie o centymetry, a polityka widzi co innego niz w
symulacji, i nikt tego nie zauwaza.

Porownanie to dopasowanie obrotu, skali i przesuniecia (podobienstwo, 4 parametry)
kadru teraz do kadru odniesienia, startujace z przesuniecia z korelacji fazowej,
na obrazie w odcieniach szarosci, z RAMIENIEM WYCIETYM: ramie rusza sie miedzy
kadrami, a jego piksele przesuwalyby wynik. Maske ramienia daje blizniak -
segmentacja geomow robota wyrenderowana z tej samej kamery w jej skalibrowanej pozie.
Tego, czego maska nie zna (reka, tulow operatora, cien ramienia), pozbywaja sie wagi
odpornosciowe (IRLS), a "przestawiona" wymaga dwoch kolejnych, zgodnych sprawdzen.

Miara jest NAJWIEKSZE przesuniecie naroznika kadru, nie przesuniecie calosci:
sama korelacja fazowa nie widziala obrotu wokol osi optycznej ani ruchu wzdluz
niej (srodek kadru stoi, brzegi jada). Zmierzone na kadrze 640x480: obrot
o 3 st. przesuwal narozniki o 21 px, a korelacja zglaszala 1,6 px; zblizenie
o 4% - 16 px naroznikow i 0,55 px korelacji. Prog 3 px nie ruszal.
"""

from __future__ import annotations

import cv2
import mujoco
import numpy as np

from ..scene import PREFIX


def arm_mask(scene, camera: str, dilate: int = 9) -> np.ndarray:
    """Piksele ramienia w kadrze kamery sceny (bool, poszerzone o `dilate` px).

    Render sceny to kamera otworkowa bez dystorsji - do surowego kadru prawdziwej
    kamery maske trzeba jeszcze przepuscic przez `distort_mask`.
    """
    view = scene.camera(camera)
    key = ("seg", view.width, view.height)
    r = scene._renderers.get(key)
    if r is None:
        r = mujoco.Renderer(scene.model, height=view.height, width=view.width)
        r.enable_segmentation_rendering()
        scene._renderers[key] = r
    r.update_scene(scene.data, camera=camera)
    seg = r.render()
    m = scene.model
    robot = np.array([m.body(m.geom_bodyid[g]).name.startswith(PREFIX) for g in range(m.ngeom)])
    ids, types = seg[..., 0], seg[..., 1]
    is_geom = (types == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0)
    mask = np.zeros(ids.shape, bool)
    mask[is_geom] = robot[ids[is_geom]]
    if dilate:
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((dilate, dilate), np.uint8)) > 0
    return mask


_DISTORT_MAPS: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def distortion_maps(K: np.ndarray, dist: np.ndarray | None, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    """Mapy `cv2.remap`: piksel SUROWEGO kadru -> piksel renderu otworkowego z tym samym K.

    None, gdy kamera nie ma dystorsji (render i kadr maja wtedy te sama geometrie).
    """
    if dist is None or not np.any(np.asarray(dist, float)):
        return None
    K = np.asarray(K, float)
    d = np.asarray(dist, float).ravel()
    w, h = size
    key = (w, h, K.tobytes(), d.tobytes())
    maps = _DISTORT_MAPS.get(key)
    if maps is None:
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        pts = np.stack([u.ravel(), v.ravel()], axis=1)[:, None, :]
        und = cv2.undistortPoints(pts, K, d, P=K).reshape(h, w, 2)
        maps = (np.ascontiguousarray(und[..., 0], np.float32), np.ascontiguousarray(und[..., 1], np.float32))
        if len(_DISTORT_MAPS) > 16:
            _DISTORT_MAPS.clear()
        _DISTORT_MAPS[key] = maps
    return maps


def distort_mask(mask: np.ndarray, K: np.ndarray, dist: np.ndarray | None) -> np.ndarray:
    """Maska z renderu (otworkowego) w geometrii surowego kadru kamery z dystorsja `dist`.

    Dla kamerki z k1 ok. -0,3 maska renderu mijala ramie w kadrze o 6 px przy
    (480, 360) i ~20 px przy (560, 420) - piksele ramienia glosowaly wtedy jako
    "tu nie ma kostki", a percepcja przy chwytaniu gubila kostke.
    """
    maps = distortion_maps(K, dist, (mask.shape[1], mask.shape[0]))
    if maps is None:
        return mask
    out = cv2.remap(mask.astype(np.uint8), maps[0], maps[1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return out > 0


def _gray_small(img: np.ndarray, scale: float) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    # Najpierw zmniejszenie, potem reszta: rozmycie sigma 15 px na pelnym kadrze 640x480
    # kosztowalo wiecej niz cale dopasowanie.
    return cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)


def _small_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None or not mask.any():
        return None
    # INTER_AREA > 0: piksel pomniejszonego kadru, w ktorym jest COKOLWIEK ramienia, wypada.
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_AREA) > 0


def _fill(g: np.ndarray, m: np.ndarray | None, scale: float) -> np.ndarray:
    if m is None:
        return g
    # Wyciete piksele wypelniamy rozmyciem otoczenia, zeby krawedz maski nie
    # stala sie "krawedzia", ktora korelacja chetnie by dopasowala.
    return np.where(m, cv2.GaussianBlur(g, (0, 0), 15 * scale), g)


def _about(C: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Przeksztalcenie C (wspolrzedne wzgledem srodka c) w pikselach kadru."""
    return np.array([[1.0, 0, c[0]], [0, 1.0, c[1]], [0, 0, 1.0]]) @ C @ \
        np.array([[1.0, 0, -c[0]], [0, 1.0, -c[1]], [0, 0, 1.0]])


class _Level:
    """Kadr odniesienia na jednym pietrze piramidy - z tym, co dopasowanie liczy raz.

    Dopasowanie odwrotnie-zlozeniowe (inverse compositional, Baker-Matthews): gradienty
    i jakobian liczone na ODNIESIENIU, raz przy `remember`, a nie w kazdej iteracji.
    """

    def __init__(self, ref: np.ndarray, mask: np.ndarray | None):
        self.ref = ref
        h, w = ref.shape
        self.c = np.array([(w - 1) / 2, (h - 1) / 2])
        valid = np.ones(ref.shape, bool) if mask is None else ~mask
        valid[0, :] = valid[-1, :] = valid[:, 0] = valid[:, -1] = False
        #: Waga piksela odniesienia (0 = ramie albo brzeg); dalej wszystko na CALYCH
        #: kadrach z wagami - wybieranie pikseli indeksami w kazdej iteracji kosztowalo
        #: wiecej niz sam rachunek.
        self.valid = valid.astype(np.float32).ravel()
        ys, xs = np.mgrid[0:h, 0:w]
        X, Y = (xs - self.c[0]).astype(np.float32).ravel(), (ys - self.c[1]).astype(np.float32).ravel()
        gx = (cv2.Sobel(ref, cv2.CV_32F, 1, 0, ksize=1) * 0.5).ravel()
        gy = (cv2.Sobel(ref, cv2.CV_32F, 0, 1, ksize=1) * 0.5).ravel()
        # Podobienstwo w 4 parametrach (a, b, tx, ty): x' = [[1+a, -b], [b, 1+a]] x + t.
        self.J = np.stack([gx * X + gy * Y, -gx * Y + gy * X, gx, gy], 1)
        self.JT = np.ascontiguousarray(self.J.T)
        #: Modul gradientu odniesienia: reszta, ktora da sie wyjasnic malym ruchem kadru
        #: (krawedz przesunieta o piksel), nie jest "obcym obiektem" w wagach odpornosciowych.
        self.grad = np.hypot(gx, gy).astype(np.float32)
        self.T = ref.ravel()
        self.ones = np.ones(ref.shape, np.float32)


def _tukey(r: np.ndarray, sel: np.ndarray, k: float, floor: float, slack: np.ndarray) -> np.ndarray:
    """Wagi Tukeya (0..1) reszt `r`; skala z MAD reszt pikseli `sel`, nie mniejsza niz `floor`.

    `slack` - reszta, ktora piksel moze miec od samego ruchu kadru (gradient x kilka px):
    dodawana do progu. Bez niej krawedzie przesuniete prawdziwym ruchem kamery wypadaly
    razem z reka - obrot o 1 st. mierzony 5,5 px zamiast 7,0.
    """
    a = np.abs(r)
    s = a[sel][::3]                  # co trzeci piksel: mediana tak samo pewna, 3x taniej
    if s.size == 0:
        return np.ones_like(r)
    mad = float(np.partition(s, s.size // 2)[s.size // 2])
    u = np.minimum(a / (k * max(1.4826 * mad, floor) + slack), 1.0)
    return ((1.0 - u * u) ** 2).astype(np.float32)


def _moments(T: np.ndarray, I: np.ndarray, wgt: np.ndarray) -> tuple[float, float, float, float, float]:
    """Suma wag, srednie i odchylenia odniesienia i kadru (jasnosc i kontrast do dopasowania)."""
    n = float(wgt.sum())
    if n < 1e-6:
        return 0.0, 0.0, 0.0, 1e-6, 1e-6
    Tm, Im = float(wgt @ T) / n, float(wgt @ I) / n
    dT, dI = T - Tm, I - Im
    return (n, Tm, Im, np.sqrt(float((dT * wgt) @ dT) / n) + 1e-6,
            np.sqrt(float((dI * wgt) @ dI) / n) + 1e-6)


def _fit_similarity(L: _Level, cur: np.ndarray, keep: np.ndarray, C: np.ndarray, iters: int,
                    tol: float, give_up: float, robust: tuple[float, float, float] | None = None,
                    robust_from: int = 0, max_reweights: int = 3) -> tuple[np.ndarray | None, float]:
    """Obrot + skala + przesuniecie kadru `cur` wzgledem odniesienia; (C, korelacja) albo (None, nan).

    Jasnosc i kontrast dopasowywane w kazdej iteracji (jak w ECC) - automatyczna
    ekspozycja kamery nie jest ruchem. `keep` - waga pikseli kadru (0 = ramie w kadrze).
    Korelacja ponizej `give_up` po kilku iteracjach konczy dopasowanie: ten sam kadr
    po ruchu kamery ma 0,9+ juz po starcie z korelacji fazowej, a "inny kadr" krecil
    sie wszystkie iteracje (~30 ms) po to, zeby i tak wyjsc "nie pasuje".

    `robust` = (k, dolna skala reszt, ruch kadru [px] wyjasniajacy reszte) - od iteracji
    `robust_from` piksele wazone wagami Tukeya reszty (IRLS): to, czego nie bylo w kadrze
    odniesienia i czego maska ramienia nie zna (reka, tulow operatora), ma reszte
    wielokrotnie wieksza niz blat i wypada.
    Bez tego (zmierzone, kadry z blizniaka, reka jako elipsa z brzegu): 10% kadru -
    4-60 px "przesuniecia", 20% - 7-130 px, przy korelacji 0,88-0,97 (powyzej `min_cc`),
    wiec panel wolal "Kamera przestawiona" przy kazdym siegnieciu do stanowiska.
    """
    h, w = L.ref.shape
    base = L.valid * keep
    T = L.T
    cc, H = float("nan"), None
    rob: np.ndarray | None = None
    reweights, pending = 0, False
    for it in range(iters):
        M = _about(C, L.c)[:2].astype(np.float32)
        I = cv2.warpAffine(cur, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REPLICATE).ravel()
        inside = cv2.warpAffine(L.ones, M, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0).ravel()
        wgt0 = base * inside
        if robust is not None and reweights < max_reweights and (it == robust_from or pending):
            # IRLS, petla zewnetrzna: wagi PRZED krokiem - pierwszy krok Gaussa-Newtona z reka
            # w kadrze potrafil sam skoczyc o kilka procent skali (i dalej juz sie nie wracalo).
            # Nowe wagi tylko na starcie i po zbiegnieciu przy starych: nowe w kazdej iteracji to
            # nowy hesjan i mediana co iteracje - 70-100 ms na kamere zamiast 10-20 ms.
            # Reszta liczona BEZ wag odpornosciowych (tez tam, gdzie waga spadla do zera) -
            # piksel, ktory wrocil na swoje miejsce, dostaje z powrotem glos.
            _, Tm, Im, Ts, Is = _moments(T, I, wgt0 if rob is None else wgt0 * rob)
            r = (I - Im) * (Ts / Is) - (T - Tm)
            rob, H = _tukey(r, wgt0 > 0, robust[0], robust[1], L.grad * robust[2]), None
            reweights, pending = reweights + 1, False
        wgt = wgt0 if rob is None else wgt0 * rob
        n, Tm, Im, Ts, Is = _moments(T, I, wgt)
        if n < 200:
            return None, float("nan")
        dT, dI = (T - Tm) * wgt, (I - Im) * wgt
        if rob is None:
            cc = float(dT @ dI) / (n * Ts * Is)
        else:
            # Korelacja (do `min_cc`) ZWYKLA, bez wag odpornosciowych: te podnosza wage gladkiego
            # blatu, na ktorym jest glownie szum kamery - korelacja kadru z blizniaka z szumem
            # spadala z 0,93-0,97 do 0,88, a na zaszumionej kamerce spadalaby dalej, pod `min_cc`
            # ("kadr nie pasuje", inf). Wagi decyduja, GDZIE kadr jest; o tym, czy to ten sam
            # kadr, decyduje caly kadr, jak przed zmiana.
            n0, Tm0, Im0, Ts0, Is0 = _moments(T, I, wgt0)
            cc = float(((T - Tm0) * wgt0) @ (I - Im0)) / (n0 * Ts0 * Is0)
        if it >= 4 and cc < give_up:
            return C, cc
        e = dI * (Ts / Is) - dT                            # blad z dopasowana jasnoscia, 0 poza waga
        if H is None:
            # Hesjan od nowa tylko po zmianie wag odpornosciowych: miedzy iteracjami zmienia sie
            # poza tym tylko pasek pikseli przy brzegu, co spowalnia zbieganie o ulamek, a nie
            # przesuwa wyniku (ten wyznacza gradient J^T e liczony zawsze z biezacymi wagami).
            H = ((L.JT * wgt) @ L.J).astype(np.float64)
        try:
            a, b, tx, ty = np.linalg.solve(H, (L.JT @ e).astype(np.float64))
        except np.linalg.LinAlgError:
            return None, float("nan")
        C = C @ np.linalg.inv(np.array([[1 + a, -b, tx], [b, 1 + a, ty], [0, 0, 1.0]]))
        step = max(abs(tx), abs(ty), max(abs(a), abs(b)) * L.c[0])
        if robust is not None and reweights < max_reweights:
            # Posrednie wagi nie potrzebuja dokladnego zbiegniecia - nastepne i tak je poprawia;
            # z pelna dokladnoscia kazde z nich kosztowalo 2-3 iteracje wiecej.
            if step < 5 * tol:
                pending = True
            continue
        if step < tol:
            break
    return C, cc


class CameraWatch:
    #: Ponizej tej korelacji dopasowanie nie jest wiarygodne - kadr jest
    #: "nie wiadomo jaki", a to traktujemy jak przestawiony, nie jak zero ruchu.
    #: Zmierzone: ten sam kadr po obrocie/przesunieciu/zblizeniu 0,99+, zupelnie inny kadr ~0.
    min_cc = 0.6
    #: Iteracje na pietrze zgrubnym (1/4 kadru) i dokladnym (1/2 kadru). Zmierzone
    #: (640x480, obrot do 3 st., skala do 4%, przesuniecie do 47 px): zbiega w 5-15
    #: zgrubnych i 2-4 dokladnych; wiecej tylko wtedy, gdy kadr i tak jest "inny".
    iters = (25, 6)
    #: IRLS: (k Tukeya, najmniejsza skala reszty [poziomy szarosci]) i od ktorej iteracji
    #: na pietrze (zgrubne, dokladne). None - bez wag odpornosciowych.
    robust: tuple[float, float] | None = (3.0, 3.0)
    robust_from = (0, 0)
    #: Ile razy na pietrze liczyc wagi od nowa. Zgrubne: 2 zamiast 3 gubily 0,3-0,7 px
    #: z obrotu 1 st. (7,0 px) - krawedzie przesuniete ruchem kamery wracaja do wag stopniowo.
    reweights = (3, 1)
    #: Ruch kadru [px pietra], ktory tlumaczy reszte piksela (dodawany do progu jako gradient x px).
    #: 0 - zblizenie 1% mierzone 2,9-3,6 px zamiast 4,0 (ponizej progu 3 px w najgorszej kamerze);
    #: (1,0, 0,5) - reka 10% kadru dawala do 15 px, (0,5, 0,25) - do 3,4 px.
    slack_px = (0.5, 0.25)
    #: "Przestawiona" dopiero, gdy DWA kolejne sprawdzenia (panel: co 2 s) widza ten sam ruch
    #: kadru - kazdy naroznik w obu na tyle samo, z dokladnoscia do `agree_px`. Reka czy tulow
    #: operatora w kadrze nie stoi w miejscu przez dwa sprawdzenia, a przestawiona kamera tak.
    agree_px = 3.0

    def __init__(self, threshold_px: float = 3.0, scale: float = 0.5):
        self.threshold = threshold_px
        self.scale = scale
        self._levels: dict[str, tuple[_Level, _Level]] = {}
        #: Najwieksze przesuniecie naroznika z OSTATNIEGO sprawdzenia (bez potwierdzenia).
        self.shift: dict[str, float] = {}
        #: Szczegoly ostatniego sprawdzenia: przesuniecie calosci, obrot, skala, pewnosc,
        #: `waiting` - ponad progiem, ale jeszcze bez potwierdzenia drugim sprawdzeniem.
        self.detail: dict[str, dict[str, float]] = {}
        #: Przesuniecia naroznikow (4, 2) [px] z poprzedniego sprawdzenia; None - kadr nie pasuje.
        self._prev: dict[str, np.ndarray | None] = {}
        self._moved: dict[str, bool] = {}

    @property
    def refs(self) -> dict[str, np.ndarray]:
        """Zapamietane kadry odniesienia (w polowie rozdzielczosci) - kto ma odniesienie."""
        return {n: fine.ref for n, (_, fine) in self._levels.items()}

    def _pyramid(self, img: np.ndarray,
                 mask: np.ndarray | None) -> list[tuple[np.ndarray, np.ndarray | None]]:
        fine = _gray_small(img, self.scale)
        coarse = cv2.resize(fine, (fine.shape[1] // 2, fine.shape[0] // 2), interpolation=cv2.INTER_AREA)
        out = []
        for g, s in ((coarse, self.scale / 2), (fine, self.scale)):
            m = _small_mask(mask, g.shape)
            out.append((_fill(g, m, s), m))
        return out

    def remember(self, name: str, img: np.ndarray, mask: np.ndarray | None = None) -> None:
        (c, cm), (f, fm) = self._pyramid(img, mask)
        self._levels[name] = (_Level(c, cm), _Level(f, fm))
        self._reset(name)

    def forget(self, name: str) -> None:
        self._levels.pop(name, None)
        self._reset(name)

    def _reset(self, name: str) -> None:
        for d in (self.shift, self.detail, self._prev, self._moved):
            d.pop(name, None)

    def check(self, name: str, img: np.ndarray, mask: np.ndarray | None = None) -> float | None:
        """Najwieksze przesuniecie naroznika kadru wzgledem odniesienia [px pelnej rozdzielczosci] albo None.

        `inf`, gdy kadru nie da sie dopasowac do odniesienia (inna scena, zaslonieta
        kamera) - nie wiadomo, gdzie kamera stoi, wiec kalibracji nie ufamy.

        Dopasowanie PODOBIENSTWA (obrot + skala + przesuniecie, 4 parametry) na
        piramidzie 1/4 -> 1/2 kadru. Wczesniej: ECC euklidesowe (bez skali) plus skala
        z osobnego dopasowania afinicznego (6 parametrow) z martwa strefa 1,5% - bo
        afiniczne na ubogim kadrze z ruchomym ramieniem mialo 0,3-0,6% szumu skali
        (cienie ramienia). Zblizenie o 1-2% (4-8 px naroznika) przechodzilo wtedy
        niezauwazone, a dwa dopasowania po 100 iteracji kosztowaly 13-300 ms na kamere
        w petli panelu.

        Z wagami odpornosciowymi (IRLS, patrz `_fit_similarity`) - zmierzone w blizniaku
        (kadr 640x480, dwie kamery jak w panelu, 20 losowych poz ramienia, zamaskowane,
        z twardym cieniem na blacie, dwa ziarna losowania): szum ruchu ramienia <= 0,09 px
        naroznika (bez wag: do 0,88 / 1,30 px - cien ramienia nie jest w masce, a teraz
        wypada jak reka); zblizenie 1% mierzone 3,9-4,5 px (prawda 4,0), 2% - 7,9-8,7 px,
        obrot 1 st. - 6,7-6,9 px (prawda 7,0); reka 10-20% kadru 0,1-1,3 px (bez wag
        1-68 px); ~15 ms na kamere (mediana w panelu, maszyna obciazona jak bez wag).
        Zostaje przypadek graniczny: w kadrze ubogim w teksture (gladki blat z symulacji)
        reka 20% zaslaniajaca jedyna mocna krawedz (tyl blatu) daje dalej 35-39 px -
        przed falszywym alarmem chroni wtedy wymog dwoch zgodnych sprawdzen (`moved`).
        """
        lv = self._levels.get(name)
        if lv is None or img is None:
            return None
        cur = self._pyramid(img, mask)
        if cur[1][0].shape != lv[1].ref.shape:
            return None
        coarse = lv[0]
        win = cv2.createHanningWindow(coarse.ref.shape[::-1], cv2.CV_32F)
        # KOPIE: `phaseCorrelate` z oknem mnozy wejscia przez okno W MIEJSCU (OpenCV 5.0) -
        # zapamietany kadr odniesienia wygaszal sie co sprawdzenie (co 2 s) do samego
        # srodka, a brzegi, na ktorych widac obrot, znikaly z porownania.
        (dx, dy), response = cv2.phaseCorrelate(coarse.ref.copy(), cur[0][0].copy(), win)
        s0 = self.scale / 2
        C = np.array([[1.0, 0, dx], [0, 1.0, dy], [0, 0, 1.0]])
        cc = float("nan")
        for k, (L, (g, m)) in enumerate(zip(lv, cur)):
            if k:
                C = C.copy()
                C[:2, 2] *= 2.0                       # przesuniecie z 1/4 na 1/2 kadru
            keep = L.ones.ravel() if m is None else (~m).astype(np.float32).ravel()
            C, cc = _fit_similarity(L, g, keep, C, self.iters[k], tol=0.02 if k == 0 else 0.01,
                                    give_up=self.min_cc / 2,
                                    robust=None if self.robust is None else (*self.robust, self.slack_px[k]),
                                    robust_from=self.robust_from[k], max_reweights=self.reweights[k])
            # Zgrubne dopasowanie juz "nie pasuje" - dokladne tego nie naprawi, a kosztuje.
            if C is None or cc < self.min_cc:
                break
        h, w = lv[1].ref.shape
        corners = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]], float)
        if C is not None and cc >= self.min_cc:
            M = _about(C, lv[1].c)
            disp = (corners @ M[:2].T - corners[:, :2]) / self.scale
            shift = float(np.linalg.norm(disp, axis=1).max())
            rot = float(np.degrees(np.arctan2(C[1, 0], C[0, 0])))
            scale = float(np.hypot(C[0, 0], C[1, 0]))
        elif C is None and response >= 0.3:
            # Kadr bez tekstury do dopasowania, ale korelacja fazowa pewna - zostaje jej przesuniecie.
            disp = np.tile([dx / s0, dy / s0], (4, 1))
            shift, rot, scale = float(np.hypot(dx, dy)) / s0, float("nan"), float("nan")
        else:
            disp, shift, rot, scale = None, float("inf"), float("nan"), float("nan")
        prev, self._prev[name] = self._prev.get(name, "brak"), disp
        if shift <= self.threshold:
            self._moved[name] = False
        elif not isinstance(prev, str) and (
                (prev is None and disp is None)
                or (prev is not None and disp is not None
                    and float(np.linalg.norm(prev - disp, axis=1).max()) <= self.agree_px)):
            self._moved[name] = True
        # Ponad progiem, ale inaczej niz poprzednio: stan bez zmian - potwierdzona przestawiona
        # kamera nie "wraca", gdy ktos przejdzie przed nia, a niepotwierdzona czeka.
        self.shift[name] = shift
        self.detail[name] = dict(dx=float(dx) / s0, dy=float(dy) / s0, rot_deg=rot, scale=scale,
                                 cc=cc, response=float(response),
                                 waiting=bool(shift > self.threshold and not self._moved.get(name, False)))
        return shift

    def moved(self, name: str) -> bool:
        """Kamera przestawiona: dwa kolejne sprawdzenia ponad progiem i zgodne (`agree_px`).

        Jedno sprawdzenie nie wystarcza: nie zamaskowana reka albo tulow operatora w kadrze
        (maska jest tylko dla ramienia) dawaly pojedyncze "przesuniecia" 4-60 px przy
        korelacji 0,86-0,97 i panel wolal "Kamera przestawiona" przy kazdym siegnieciu do
        stanowiska (zmierzone na kadrach z blizniaka). Wagi odpornosciowe zbijaja wiekszosc
        z nich ponizej progu; to, co zostaje, nie powtarza sie w nastepnym sprawdzeniu.
        """
        return self._moved.get(name, False)
