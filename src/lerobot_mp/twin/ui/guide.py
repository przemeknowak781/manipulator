"""Przewodnik panelu: gdzie operator jest w procedurze pierwszego testu i co zrobic teraz.

Czysta logika bez visera - `GuideSnapshot` (stan panelu i ramienia) na wejsciu,
`Guide` (kroki 0-8 z docs/TWIN.md, jedna nastepna akcja, blokady) na wyjsciu.
Panel (`app.TwinApp._tick_guide`) tylko sklada migawke i wyswietla `render_head`
(zawsze widoczne "Teraz" i uwagi) oraz `render_steps` (zwinieta lista krokow).

Kroki sa te same w symulacji i na prawdziwym ramieniu - w symulacji z kamerami
symulowanymi (znaja swoje K, wiec krok 2 odpada; Sim-Real zastepuje blad
wzgledem prawdy w zakladce Kamery).

Styl akcji: "Zakladka > Sekcja: ..., **Przycisk**" - nazwy przyciskow pogrubione jak
w notkach "Jak uzywac"; "Teraz" z kroku zaczyna sie od "Krok N/8 (tytul):".
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

DONE, CURRENT, TODO, SKIP = "gotowe", "teraz", "do zrobienia", "nie dotyczy"
#: Znacznik kroku w liscie (czysty tekst - markdown panelu nie ma pol wyboru).
MARK = {DONE: "`[x]`", CURRENT: "`[>]`", TODO: "`[ ]`", SKIP: "`[-]`"}
#: Mediana rozjazdu krawedzi Sim-Real [px], powyzej ktorej cos jest zle (TWIN.md, krok 4).
SIMREAL_MAX_PX = 3.0
PANEL = "panel"
LAST_STEP = 8
#: Dluzszy opis zrobionego kroku nie trafia do listy (jest w `Step.detail`).
DONE_DETAIL_MAX = 48

#: Jak zatrzymac zadanie, ktore ma ramie (wlasciciel `Twin.owner` / `TwinApp._busy`).
HOW_TO_STOP = {
    "fala kalibracyjna": "Kalibracja > Przerwij",
    "kalibracja": "Kalibracja > Przerwij",
    "identyfikacja dynamiki": "~20 s nagrania; przerywa je STOP albo Pozycja domowa",
    "identyfikacja": "~20 s nagrania; przerywa je STOP albo Pozycja domowa",
    "polityka": "Polityki > Zatrzymaj",
}
WAVE_OWNERS = ("fala kalibracyjna", "kalibracja")
SYSID_OWNERS = ("identyfikacja dynamiki", "identyfikacja")


def dynamics_backend(source: str) -> str:
    """Backend, na ktorym zidentyfikowano dynamike, z `Dynamics.source` ("" = nie wiadomo).

    `sysid.fit` pisze "identyfikacja <czas> (<backend>); niepewnosc: ...".
    """
    m = re.match(r"\s*identyfikacja\b[^(;]*\(([\w-]+)\)", source or "")
    return m.group(1) if m else ""


@dataclass(frozen=True)
class CameraSnap:
    name: str
    simulated: bool
    enabled: bool = True
    #: Kamera daje kadr (panel liczy ja za martwa dopiero po kilku sekundach bez kadru).
    has_frame: bool = True
    #: K wystarcza do zaufanej pozy (`CameraRecord.intrinsics_problem() == ""`).
    intrinsics_ok: bool = False
    intrinsics_problem: str = ""
    calibrated: bool = False
    trusted: bool = False
    #: Powod niezaufanej pozy (z kalibracji).
    reason: str = ""
    #: Straznik kamer: kadr przesunal sie od kalibracji.
    moved: bool = False
    #: Najlepsza mediana rozjazdu krawedzi w Sim-Real [px] od ostatniej zmiany pozy/stolu/K;
    #: None = nie mierzono, NaN = render bez krawedzi (nic do porownania).
    simreal_px: float | None = None


@dataclass(frozen=True)
class GuideSnapshot:
    connected: bool = False
    #: "sim", "feetech", "lerobot" (albo "brak" przed pierwszym polaczeniem).
    backend: str = "brak"
    simulated: bool = True
    arm_error: str = ""
    estop: bool = False
    #: Kto INNY niz panel ma ramie ("" = nikt albo panel).
    owner: str = ""
    warnings: tuple[str, ...] = ()
    cameras: tuple[CameraSnap, ...] = ()
    #: Zrodlo zidentyfikowanej dynamiki; "" = model Menagerie (brak identyfikacji).
    dynamics: str = ""
    #: Backend, na ktorym ja zmierzono (`dynamics_backend`); "" = nie wiadomo.
    dynamics_backend: str = ""
    #: (nazwa, zadanie, bazowa z repozytorium).
    policies: tuple[tuple[str, str, bool], ...] = ()
    #: Zadania w tle: "fala kalibracyjna", "intrynsyki", "identyfikacja", "trening", "ewaluacja".
    jobs: tuple[str, ...] = ()
    #: Zadanie jadacej polityki ("reach"/"lift"), "" = nie jedzie.
    policy_running: str = ""
    policy_from_cameras: bool = False
    calib_result_pending: bool = False
    dyn_result_pending: bool = False
    #: Postep fali kalibracyjnej 0..1 (None = nie jedzie).
    calib_progress: float | None = None
    #: Co juz jechalo na tym backendzie w tej sesji panelu: "reach", "lift-kamery".
    ran: frozenset[str] = field(default_factory=frozenset)
    #: Kamera, z ktorej zbierane sa kadry intrynsyk ("" = nie trwa).
    intr_camera: str = ""


@dataclass(frozen=True)
class Step:
    number: int
    title: str
    status: str
    detail: str = ""
    #: Co zrobic w tym kroku (zakladka > przycisk).
    action: str = ""


@dataclass(frozen=True)
class Guide:
    steps: tuple[Step, ...]
    now: str
    blockers: tuple[str, ...] = ()
    background: tuple[str, ...] = ()
    mode: str = ""

    @property
    def current(self) -> Step | None:
        return next((s for s in self.steps if s.status == CURRENT), None)


# ------------------------------------------------------------------ kroki
def _relevant(s: GuideSnapshot) -> list[CameraSnap]:
    """Kamery tego trybu: w symulacji symulowane, na prawdziwym ramieniu prawdziwe (wlaczone)."""
    return [c for c in s.cameras if c.enabled and c.simulated == s.simulated]


def _pose_ok(c: CameraSnap) -> bool:
    """Zaufana poza liczona z zaufanym K (stara kalibracja bez zapisanego K przy nominalnym K - nie)."""
    return c.trusted and (c.simulated or c.intrinsics_ok)


def _px(v: float) -> str:
    """Piksele zaokraglone do 0,5 - lista nie drga od dziesiatych czesci."""
    return f"{round(v * 2) / 2:.1f} px"


def _step_arm(s: GuideSnapshot) -> Step:
    title = "Ramie do maszyny i Polacz"
    if s.connected:
        det = f"polaczone: {s.backend}" + (" (blizniak jest ramieniem)" if s.simulated else "")
        return Step(0, title, DONE, det)
    act = ("Ramie > Polaczenie: Ramie = sim, **Polacz**" if s.simulated
           else "Ramie > Polaczenie: Ramie = feetech, **Wykryj porty** (albo socket://adres:5555 z mostu), "
                "odznacz 'Po polaczeniu jedz do pozycji domowej', **Polacz**")
    det = "rozlaczone" + (f" ({s.arm_error})" if s.arm_error else "")
    return Step(0, title, TODO, det, act)


def _step_cameras(s: GuideSnapshot) -> Step:
    title = "Kamery"
    cams = _relevant(s)
    live = [c for c in cams if c.has_frame]
    if live:
        det = f"{len(live)} z kadrem" + (f", bez kadru: {', '.join(c.name for c in cams if not c.has_frame)}"
                                          if len(live) < len(cams) else "")
        return Step(1, title, DONE, det)
    if s.simulated:
        act = "Kamery > Dodaj kamere: **Dodaj symulowana przed ramieniem** (dwa razy - dwie kamery to lepsza mapa)"
    else:
        act = "Kamery > Dodaj kamere: **Szukaj kamer USB**, wybierz w Znalezione, **Dodaj kamere USB**"
    det = ("kamery sa, ale zadna nie daje kadru" if cams else
           "brak kamer " + ("symulowanych" if s.simulated else "USB"))
    return Step(1, title, TODO, det, act)


def _step_intrinsics(s: GuideSnapshot) -> Step:
    title = "Intrynsyki (tablica ChArUco)"
    cams = _relevant(s)
    if s.simulated:
        return Step(2, title, SKIP, "kamery symulowane znaja swoje K")
    if not cams:
        return Step(2, title, TODO, "najpierw kamery")
    bad = [c for c in cams if not c.intrinsics_ok]
    if not bad:
        return Step(2, title, DONE, f"zaufane K: {', '.join(c.name for c in cams)}")
    first = bad[0]
    if s.intr_camera:
        act = (f"pokazuj tablice kamerze {s.intr_camera} w rogach kadru, blizej, dalej i pochylona (12+ kadrow, "
               f"pokrycie 55%+), potem Kalibracja > 1. Intrynsyki: **Oblicz i zapisz K**")
    else:
        act = (f"Kalibracja > 1. Intrynsyki: **Pobierz arkusz tablicy** (druk 100%), wpisz zmierzony bok "
               f"kwadratu, Kamera = {first.name}, **Zbieraj kadry**")
    det = "; ".join(f"{c.name}: {c.intrinsics_problem or 'brak zaufanego K'}" for c in bad)
    return Step(2, title, TODO, det, act)


def _step_poses(s: GuideSnapshot) -> Step:
    title = "Polozenie kamer (fala z karta)"
    cams = _relevant(s)
    if not cams:
        return Step(3, title, TODO, "najpierw kamery")
    moved = [c for c in cams if c.calibrated and c.moved]
    bad = [c for c in cams if not _pose_ok(c)]
    if not bad and not moved:
        return Step(3, title, DONE, f"zaufane pozy: {', '.join(c.name for c in cams)}")
    confirm = "" if s.simulated else ", zaznacz 'Karta w szczekach...'"
    if moved and not bad:
        c = moved[0]
        act = (f"Kalibracja > 2. Polozenie: Szybka relokalizacja kamery = {c.name}{confirm}, "
               f"**Relokalizuj wybrana**")
        return Step(3, title, TODO, f"przestawiona: {', '.join(c.name for c in moved)}", act)
    if s.simulated:
        act = ("Kalibracja > 2. Polozenie: **Start fali**, potem **Zapisz wynik kalibracji** (skrot bez fali: "
               "Kamery > **Symulowana: uznaj prawdziwa poze za kalibracje**)")
    else:
        act = (f"Kalibracja > 2. Polozenie: karta w szczekach{confirm}, **Start fali**, potem **Zapisz wynik "
               f"kalibracji** (arkusz i bok taga - notka Jak uzywac)")

    def why(c: CameraSnap) -> str:
        if not c.calibrated:
            return "nieskalibrowana"
        if c.trusted:                                   # zaufana, ale K nie jest zaufane
            return "poza z niezaufanego K - po kroku 2 powtorz fale"
        return c.reason or "niezaufana"
    det = "; ".join(f"{c.name}: {why(c)}" for c in bad)
    return Step(3, title, TODO, det, act)


def _step_simreal(s: GuideSnapshot) -> Step:
    title = "Sprawdzenie Sim-Real"
    if s.simulated:
        return Step(4, title, SKIP, "kamery symulowane: blad wzgledem prawdy w zakladce Kamery")
    cams = [c for c in _relevant(s) if _pose_ok(c)]
    if not cams:
        return Step(4, title, TODO, "najpierw zaufane pozy kamer")

    def passed(c: CameraSnap) -> bool:
        return c.simreal_px is not None and not math.isnan(c.simreal_px) and c.simreal_px <= SIMREAL_MAX_PX
    todo = [c for c in cams if not passed(c)]
    if not todo:
        return Step(4, title, DONE, f"zgodne: {', '.join(c.name for c in cams)}")
    c = todo[0]
    if c.simreal_px is None:
        det = f"nie sprawdzone: {', '.join(x.name for x in todo)}"
    elif math.isnan(c.simreal_px):
        det = f"{c.name}: nie sprawdzone (render bez krawedzi - czy kamera widzi ramie i stol?)"
    else:
        det = (f"{c.name}: najlepsza mediana {_px(c.simreal_px)} > {SIMREAL_MAX_PX:.0f} px - sprawdz zmierzony "
               f"bok taga i stol (Ramie > Stanowisko), potem powtorz fale")
    act = (f"Sim-Real: Kamera = {c.name}, zaznacz **Krawedzie symulacji** - zolte krawedzie maja lezec na ramieniu "
           f"i stole (mediana <= {SIMREAL_MAX_PX:.0f} px)")
    return Step(4, title, TODO, det, act)


def _step_dynamics(s: GuideSnapshot) -> Step:
    title = "Dynamika serw (identyfikacja)"
    # W symulacji wystarczy proba na blizniaku; na prawdziwym ramieniu dynamika z sim
    # (zapisana przy probie procedury) nie jest dynamika serw.
    if s.dynamics and (s.simulated or s.dynamics_backend != "sim"):
        return Step(5, title, DONE, f"zmierzona na {s.dynamics_backend}" if s.dynamics_backend else "zmierzona")
    act = ("Trening > Dynamika serw: " + ("" if s.simulated else "zaznacz 'przestrzen wokol wolna', ")
           + "**Identyfikuj** (~20 s ruchu), potem **Zapisz jako dynamike stanowiska**")
    det = "zmierzona na sim - zmierz na ramieniu" if s.dynamics else "model Menagerie (nie zmierzona)"
    return Step(5, title, TODO, det, act)


def _step_training(s: GuideSnapshot) -> Step:
    title = "Trening / douczanie"
    own = [n for n, _, bundled in s.policies if not bundled]
    if "trening" in s.jobs:
        return Step(6, title, TODO, "trening trwa", "poczekaj na koniec treningu (wykres w zakladce Trening)")
    if own:
        return Step(6, title, DONE, f"polityki stanowiska: {', '.join(own[:3])}")
    if not s.policies:
        return Step(6, title, TODO, "brak polityk", "Trening > Nowy trening: Zadanie = reach, Iteracje 160, **Ucz**")
    if s.simulated:                                     # proba procedury: bazowe wystarcza
        return Step(6, title, DONE, "bazowe " + ", ".join(n for n, _, _ in s.policies[:2]))
    reach = next((n for n, t, _ in s.policies if t == "reach"), s.policies[0][0])
    act = (f"Trening > Nowy trening: Start z polityki = {reach}, Iteracje 100-300, **Ucz** "
           f"(potem to samo z lift)")
    return Step(6, title, TODO, "tylko bazowe - douczaj na zmierzonej dynamice", act)


def _step_reach(s: GuideSnapshot) -> Step:
    title = "reach na ramieniu"
    names = [n for n, t, _ in s.policies if t == "reach"]
    if s.policy_running == "reach":
        return Step(7, title, DONE if "reach" in s.ran else TODO, "jedzie - przeciagaj zolta kulke celu w 3D",
                    "przeciagaj zolta kulke celu w 3D; Polityki > **Zatrzymaj**, gdy wystarczy")
    if "reach" in s.ran:
        return Step(7, title, DONE, f"doszla do celu na {s.backend} w tej sesji")
    if not names:
        return Step(7, title, TODO, "brak polityki reach", "Trening > Nowy trening: Zadanie = reach, **Ucz**")
    act = (f"Polityki: Polityka = {names[0]}" + ("" if s.simulated else ", zaznacz 'rozumiem, ze sie ruszy'")
           + ", **Uruchom**; przeciagaj zolta kulke celu w 3D")
    return Step(7, title, TODO, "sprawdzenie w kazdej sesji panelu", act)


def _step_lift(s: GuideSnapshot) -> Step:
    title = "lift z kamer"
    names = [n for n, t, _ in s.policies if t == "lift"]
    if s.policy_running == "lift" and s.policy_from_cameras:
        return Step(8, title, DONE if "lift-kamery" in s.ran else TODO, "jedzie - kostka z kamer",
                    "patrz na ramie, reka przy STOP; polityka konczy sama ('zadanie wykonane')")
    if "lift-kamery" in s.ran:
        return Step(8, title, DONE, f"podniosla kostke na {s.backend} w tej sesji")
    if not names:
        return Step(8, title, TODO, "brak polityki lift", "Trening > Nowy trening: Zadanie = lift, **Ucz**")
    if s.simulated:
        act = (f"Polityki > **lift (sim): poloz kostke losowo**; Polityka = {names[0]}, skad polozenie kostki = "
               f"kamery, **Uruchom**")
    else:
        act = (f"kostka 3 cm przed ramieniem (widac ja na Mapie); Polityki: Polityka = {names[0]}, skad polozenie "
               f"kostki = kamery, zaznacz 'rozumiem, ze sie ruszy', **Uruchom**")
    return Step(8, title, TODO, "sprawdzenie w kazdej sesji panelu", act)


STEP_BUILDERS = (_step_arm, _step_cameras, _step_intrinsics, _step_poses, _step_simreal, _step_dynamics,
                 _step_training, _step_reach, _step_lift)


# ------------------------------------------------------------------ calosc
def blockers(s: GuideSnapshot) -> list[str]:
    """Uwagi, ktorych NIE mowi "Teraz" (STOP, wlasciciel ramienia i jazda polityki sa juz tam)."""
    out = []
    if s.estop and s.owner and s.owner != PANEL:
        # Przy STOP-ie "Teraz" mowi o STOP-ie - kto ma ramie, trzeba powiedziec osobno.
        how = HOW_TO_STOP.get(s.owner, "STOP")
        if s.owner == "polityka":
            out.append(f"ramie ma: polityka - sprzeglo, fala i identyfikacja sa odrzucane, Uruchom zastapi "
                       f"jadaca polityke ({how})")
        else:
            out.append(f"ramie ma: {s.owner} - sprzeglo, fala, identyfikacja i polityka sa odrzucane, "
                       f"dopoki nie skonczy ({how})")
    elif not s.estop and s.connected and s.arm_error:
        out.append(f"ramie: {s.arm_error}")
    out += [str(w) for w in s.warnings]
    if s.connected and s.backend == "lerobot":
        out.append("backend lerobot liczy katy inaczej niz blizniak - do blizniaka uzywaj feetech")
    for c in s.cameras:
        if not c.enabled:
            continue
        if c.calibrated and c.moved:
            out.append(f"kamera {c.name} przestawiona - Kalibracja > **Relokalizuj wybrana**")
        if not c.has_frame:
            out.append(f"kamera {c.name} nie daje kadru" + ("" if c.simulated else
                                                           " (na Shadow: przepuszczenie USB w kliencie)"))
    if s.cameras and not _relevant(s) and s.connected:
        kind = "symulowane" if s.simulated else "prawdziwe (USB)"
        out.append(f"ramie {s.backend}: do procedury potrzebne kamery {kind}")
    return out


def _step_now(st: Step) -> str:
    return f"Krok {st.number}/{LAST_STEP} ({st.title}): {st.action or st.detail}."


def _now(s: GuideSnapshot, steps: list[Step]) -> str:
    if s.estop:
        why = f" ({s.arm_error})" if s.arm_error else ""
        return f"**aktywny STOP**{why} - usun przyczyne, potem Ramie > Sterowanie: **Skasuj STOP**."
    cur = next((x for x in steps if x.status == CURRENT), None)
    if not s.connected:
        return _step_now(steps[0])
    if s.policy_running == "reach":
        return ("Jedzie reach - przeciagaj zolta kulke celu w 3D, reka przy STOP; Polityki > **Zatrzymaj**, "
                "gdy wystarczy.")
    if s.policy_running or s.owner == "polityka":
        src = " (kostka z kamer)" if s.policy_from_cameras else ""
        return (f"Jedzie polityka {s.policy_running}{src} - reka przy STOP; konczy sama albo Polityki > "
                f"**Zatrzymaj**.")
    if s.owner in WAVE_OWNERS:
        prog = f" ({int(10 * s.calib_progress) * 10}%)" if s.calib_progress is not None else ""
        return (f"Fala kalibracyjna jedzie{prog} - nie wchodz w obszar ramienia; potem Kalibracja > **Zapisz "
                f"wynik kalibracji** (przerwac: **Przerwij**).")
    if s.owner in SYSID_OWNERS:
        return ("Identyfikacja nagrywa ruch (~20 s) - nie wchodz w obszar ramienia (przerywa ja STOP albo "
                "Pozycja domowa).")
    if s.owner and s.owner != PANEL:
        return f"Ramie ma: {s.owner} - poczekaj na koniec albo STOP."
    if s.calib_result_pending:
        return "Kalibracja > 2. Polozenie: sprawdz werdykt w tabeli i kliknij **Zapisz wynik kalibracji**."
    if s.dyn_result_pending:
        return "Trening > Dynamika serw: sprawdz wynik i kliknij **Zapisz jako dynamike stanowiska**."
    if "identyfikacja" in s.jobs:
        return "Dopasowanie dynamiki do nagrania (7-11 s, ramie juz wolne) - poczekaj na wynik w zakladce Trening."
    if cur is None:
        return ("Procedura przejdziona. Po kazdym przestawieniu kamery: relokalizacja i Sim-Real; po identyfikacji "
                "douczaj polityki (Trening > Start z polityki).")
    return _step_now(cur)


def build(s: GuideSnapshot) -> Guide:
    """Kroki 0-8 z docs/TWIN.md: pierwszy niezrobiony (i nie "nie dotyczy") to "teraz"."""
    raw = [b(s) for b in STEP_BUILDERS]
    steps, found = [], False
    for st in raw:
        if st.status in (DONE, SKIP):
            steps.append(st)
        elif not found:
            steps.append(Step(st.number, st.title, CURRENT, st.detail, st.action))
            found = True
        else:
            steps.append(st)
    # W tle tylko to, czego "Teraz" nie opisuje (fala, identyfikacja i jadaca polityka juz tam sa).
    bg = [j for j in s.jobs if j not in ("fala kalibracyjna", "identyfikacja")]
    mode = ("symulacja - te same kroki i przyciski co na stanowisku" if s.simulated
            else f"prawdziwe ramie ({s.backend}) - reka przy STOP")
    return Guide(tuple(steps), _now(s, steps), tuple(blockers(s)), tuple(bg), mode)


def render_head(g: Guide) -> str:
    """Zawsze widoczna czesc: "Teraz", uwagi, tryb (ten sam tekst dla tego samego stanu)."""
    lines = [f"**Teraz:** {g.now}"]
    if g.blockers:
        lines.append("")
        lines += [f"- **Uwaga:** {b}" for b in g.blockers]
    lines.append("")
    lines.append(f"Tryb: {g.mode}" + (f" | w tle: {', '.join(g.background)}" if g.background else ""))
    return "\n".join(lines)


def render_steps(g: Guide) -> str:
    """Lista krokow 0-8 (w panelu zwinieta): opis przy biezacym i krotki przy zrobionych."""
    lines = []
    for st in g.steps:
        if st.status == CURRENT:
            line = f"**{st.number}. {st.title}**" + (f" - {st.detail}" if st.detail else "")
        elif st.status == DONE and st.detail and len(st.detail) <= DONE_DETAIL_MAX:
            line = f"{st.number}. {st.title} - {st.detail}"
        elif st.status == SKIP:
            line = f"{st.number}. {st.title} (nie dotyczy)"
        else:
            line = f"{st.number}. {st.title}"
        lines.append(f"{MARK[st.status]} {line}  ")
    return "\n".join(lines)


def render(g: Guide) -> str:
    """Calosc jako jeden markdown (panel wysyla tylko zmiany, wiec ten sam stan = ten sam tekst)."""
    return render_head(g) + "\n\n" + render_steps(g)
