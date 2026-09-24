"""Przewodnik panelu: gdzie operator jest w procedurze pierwszego testu i co zrobic teraz.

Czysta logika bez visera - `GuideSnapshot` (stan panelu i ramienia) na wejsciu,
`Guide` (kroki 0-8 z docs/TWIN.md, jedna nastepna akcja, blokady) na wyjsciu.
Panel (`app.TwinApp._tick_guide`) tylko sklada migawke i wyswietla `render`.

Kroki sa te same w symulacji i na prawdziwym ramieniu - w symulacji z kamerami
symulowanymi (znaja swoje K, wiec krok 2 odpada; Sim-Real zastepuje blad
wzgledem prawdy w zakladce Kamery).
"""

from __future__ import annotations

from dataclasses import dataclass, field

DONE, CURRENT, TODO, SKIP = "gotowe", "teraz", "do zrobienia", "nie dotyczy"
#: Znacznik kroku w liscie (czysty tekst - markdown panelu nie ma pol wyboru).
MARK = {DONE: "`[x]`", CURRENT: "`[>]`", TODO: "`[ ]`", SKIP: "`[-]`"}
#: Mediana rozjazdu krawedzi Sim-Real [px], powyzej ktorej cos jest zle (TWIN.md, krok 4).
SIMREAL_MAX_PX = 3.0
PANEL = "panel"
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


@dataclass(frozen=True)
class CameraSnap:
    name: str
    simulated: bool
    enabled: bool = True
    #: Kamera daje swiezy kadr (ostatni `_grab`).
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
    #: Ostatnia mediana rozjazdu krawedzi w Sim-Real [px]; None = nie mierzono.
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
    #: (nazwa, zadanie, bazowa z repozytorium).
    policies: tuple[tuple[str, str, bool], ...] = ()
    #: Zadania w tle: "fala kalibracyjna", "intrynsyki", "identyfikacja", "trening", "ewaluacja".
    jobs: tuple[str, ...] = ()
    #: Zadanie jadacej polityki ("reach"/"lift"), "" = nie jedzie.
    policy_running: str = ""
    policy_from_cameras: bool = False
    calib_result_pending: bool = False
    dyn_result_pending: bool = False
    #: Co juz jechalo na tym backendzie w tej sesji: "reach", "lift-kamery".
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


def _step_arm(s: GuideSnapshot) -> Step:
    title = "Ramie do maszyny i Polacz"
    if s.connected:
        det = f"polaczone: {s.backend}" + (" (blizniak jest ramieniem)" if s.simulated else "")
        return Step(0, title, DONE, det)
    act = ("Ramie > Polaczenie: Ramie = sim, Polacz" if s.simulated
           else "Ramie > Polaczenie: Ramie = feetech, Wykryj porty (albo socket://adres:5555 z mostu), Polacz "
                "bez jazdy do domu")
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
        act = "Kamery > Dodaj kamere: Dodaj symulowana przed ramieniem (dwa razy - dwie kamery to lepsza mapa)"
    else:
        act = "Kamery > Dodaj kamere: Szukaj kamer USB, wybierz w Znalezione, Dodaj kamere USB"
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
        act = (f"Zbieranie kadrow ({s.intr_camera}): pokazuj tablice w rogach kadru, blizej, dalej i pochylona "
               f"(12+ kadrow, pokrycie 55%+), potem Kalibracja > 1. Intrynsyki > Oblicz i zapisz K")
    else:
        act = (f"Kalibracja > 1. Intrynsyki: Pobierz arkusz tablicy (druk 100%), zmierz bok kwadratu i wpisz, "
               f"Kamera = {first.name}, Zbieraj kadry")
    det = "; ".join(f"{c.name}: {c.intrinsics_problem or 'brak zaufanego K'}" for c in bad)
    return Step(2, title, TODO, det, act)


def _step_poses(s: GuideSnapshot) -> Step:
    title = "Polozenie kamer (fala z karta)"
    cams = _relevant(s)
    if not cams:
        return Step(3, title, TODO, "najpierw kamery")
    moved = [c for c in cams if c.calibrated and c.moved]
    bad = [c for c in cams if not c.trusted]
    if not bad and not moved:
        return Step(3, title, DONE, f"zaufane pozy: {', '.join(c.name for c in cams)}")
    if moved and not bad:
        c = moved[0]
        act = (f"Kalibracja > 2. Polozenie: Szybka relokalizacja kamery = {c.name}"
               + ("" if s.simulated else ", zaznacz potwierdzenie karty") + ", Relokalizuj wybrana")
        return Step(3, title, TODO, f"przestawiona: {', '.join(c.name for c in moved)}", act)
    if s.simulated:
        act = ("Kalibracja > 2. Polozenie: Start fali (w symulacji karta pojawia sie sama), potem Zapisz wynik "
               "kalibracji. Skrot bez fali: Kamery > Symulowana: uznaj prawdziwa poze za kalibracje")
    else:
        act = ("Kalibracja > 2. Polozenie: Pobierz arkusz karty, zmierz bok taga i wpisz, karta w szczekach "
               "(~70 mm wystaje), zaznacz 'Karta w szczekach...', Start fali, potem Zapisz wynik kalibracji")
    det = "; ".join(f"{c.name}: " + ((c.reason or "niezaufana") if c.calibrated else "nieskalibrowana") for c in bad)
    return Step(3, title, TODO, det, act)


def _step_simreal(s: GuideSnapshot) -> Step:
    title = "Sprawdzenie Sim-Real"
    if s.simulated:
        return Step(4, title, SKIP, "kamery symulowane: blad wzgledem prawdy w zakladce Kamery")
    cams = [c for c in _relevant(s) if c.trusted]
    if not cams:
        return Step(4, title, TODO, "najpierw zaufane pozy kamer")
    todo = [c for c in cams if c.simreal_px is None or not c.simreal_px <= SIMREAL_MAX_PX]
    if not todo:
        return Step(4, title, DONE, ", ".join(f"{c.name} {c.simreal_px:.1f} px" for c in cams))
    c = todo[0]
    if c.simreal_px is not None:
        det = (f"{c.name}: mediana {c.simreal_px:.1f} px > {SIMREAL_MAX_PX:.0f} px - sprawdz zmierzony bok taga "
               f"i stol (Ramie > Stanowisko), potem powtorz fale")
    else:
        det = f"nie sprawdzone: {', '.join(x.name for x in todo)}"
    act = (f"Sim-Real: Kamera = {c.name}, zaznacz Krawedzie symulacji - zolte krawedzie maja lezec na ramieniu "
           f"i stole (mediana <= {SIMREAL_MAX_PX:.0f} px)")
    return Step(4, title, TODO, det, act)


def _step_dynamics(s: GuideSnapshot) -> Step:
    title = "Dynamika serw (identyfikacja)"
    if s.dynamics:
        return Step(5, title, DONE, f"zmierzona ({s.dynamics})")
    act = ("Trening > Dynamika serw: " + ("" if s.simulated else "zaznacz 'przestrzen wokol wolna', ")
           + "Identyfikuj na polaczonym ramieniu (~20 s ruchu), potem Zapisz jako dynamike stanowiska")
    return Step(5, title, TODO, "model Menagerie (nie zmierzona)", act)


def _step_training(s: GuideSnapshot) -> Step:
    title = "Trening / douczanie"
    own = [n for n, _, bundled in s.policies if not bundled]
    if "trening" in s.jobs:
        return Step(6, title, TODO, "trening trwa (zakladka Trening)", "Poczekaj na koniec treningu (wykres "
                                                                         "w zakladce Trening)")
    if own:
        return Step(6, title, DONE, f"polityki stanowiska: {', '.join(own[:3])}")
    if s.policies:
        return Step(6, title, DONE, "bazowe " + ", ".join(n for n, _, _ in s.policies[:2])
                    + "; douczanie zalecane")
    return Step(6, title, TODO, "brak polityk", "Trening > Nowy trening: Zadanie = reach, Iteracje 160, Ucz")


def _step_reach(s: GuideSnapshot) -> Step:
    title = "reach na ramieniu"
    names = [n for n, t, _ in s.policies if t == "reach"]
    if s.policy_running == "reach":
        return Step(7, title, DONE if "reach" in s.ran else TODO, "jedzie - przeciagaj zolta kulke celu w 3D",
                    "Przeciagaj zolta kulke celu w 3D; Polityki > Zatrzymaj, gdy wystarczy")
    if "reach" in s.ran:
        return Step(7, title, DONE, f"jechala na {s.backend}")
    if not names:
        return Step(7, title, TODO, "brak polityki reach", "Trening: naucz reach (Zadanie = reach, Ucz)")
    act = (f"Polityki: Polityka = {names[0]}" + ("" if s.simulated else ", zaznacz 'rozumiem, ze sie ruszy'")
           + ", Uruchom; przeciagaj zolta kulke celu w 3D")
    return Step(7, title, TODO, "", act)


def _step_lift(s: GuideSnapshot) -> Step:
    title = "lift z kamer"
    names = [n for n, t, _ in s.policies if t == "lift"]
    if s.policy_running == "lift" and s.policy_from_cameras:
        return Step(8, title, DONE if "lift-kamery" in s.ran else TODO, "jedzie - kostka z kamer",
                    "Patrz na ramie, reka przy STOP; polityka konczy sama ('zadanie wykonane')")
    if "lift-kamery" in s.ran:
        return Step(8, title, DONE, f"jechala na {s.backend}, kostka z kamer")
    if not names:
        return Step(8, title, TODO, "brak polityki lift", "Trening: naucz albo douczaj lift")
    cube = ("Polityki: lift (sim): poloz kostke losowo" if s.simulated
            else "kostka 3 cm w kolorze z listy na blacie przed ramieniem")
    act = (f"{cube}; zakladka Mapa ma ja pokazywac (Szukaj kostki). Polityki: Polityka = {names[0]}, "
           f"lift: skad polozenie kostki = kamery" + ("" if s.simulated else ", zaznacz 'rozumiem, ze sie ruszy'")
           + ", Uruchom")
    return Step(8, title, TODO, "", act)


STEP_BUILDERS = (_step_arm, _step_cameras, _step_intrinsics, _step_poses, _step_simreal, _step_dynamics,
                 _step_training, _step_reach, _step_lift)


# ------------------------------------------------------------------ calosc
def blockers(s: GuideSnapshot) -> list[str]:
    out = []
    if s.estop:
        out.append("aktywny STOP - usun przyczyne i skasuj go w zakladce Ramie (Skasuj STOP)")
    elif s.connected and s.arm_error:
        out.append(f"ramie: {s.arm_error}")
    if s.owner and s.owner != PANEL:
        how = HOW_TO_STOP.get(s.owner, "STOP")
        out.append(f"ramie ma: {s.owner} - sprzeglo, fala, identyfikacja i polityka sa odrzucane, "
                   f"dopoki nie skonczy ({how})")
    out += [str(w) for w in s.warnings]
    if s.connected and s.backend == "lerobot":
        out.append("backend lerobot liczy katy inaczej niz blizniak - do blizniaka uzywaj feetech")
    for c in s.cameras:
        if not c.enabled:
            continue
        if c.calibrated and c.moved:
            out.append(f"kamera {c.name} przestawiona - Kalibracja > Relokalizuj wybrana")
        if not c.has_frame:
            out.append(f"kamera {c.name} nie daje kadru" + ("" if c.simulated else
                                                           " (na Shadow: przepuszczenie USB w kliencie)"))
    if s.cameras and not _relevant(s) and s.connected:
        kind = "symulowane" if s.simulated else "prawdziwe (USB)"
        out.append(f"ramie {s.backend}: do procedury potrzebne kamery {kind}")
    return out


def _now(s: GuideSnapshot, steps: list[Step]) -> str:
    if s.estop:
        return "Usun przyczyne STOP-u, potem Ramie > Sterowanie > Skasuj STOP."
    cur = next((x for x in steps if x.status == CURRENT), None)
    if not s.connected:
        return steps[0].action + "."
    if s.policy_running == "reach":
        return "Jedzie reach - przeciagaj zolta kulke celu w 3D, reka przy STOP; Polityki > Zatrzymaj, gdy wystarczy."
    if s.policy_running or s.owner == "polityka":
        src = " (kostka z kamer)" if s.policy_from_cameras else ""
        return (f"Jedzie polityka {s.policy_running}{src} - reka przy STOP; konczy sama albo Polityki > "
                f"Zatrzymaj.")
    if s.owner in ("fala kalibracyjna", "kalibracja"):
        return "Fala kalibracyjna jedzie - nie wchodz w obszar ramienia; potem Kalibracja > Zapisz wynik kalibracji."
    if s.owner in ("identyfikacja dynamiki", "identyfikacja"):
        return "Identyfikacja nagrywa ruch (~20 s) - nie wchodz w obszar ramienia."
    if s.owner and s.owner != PANEL:
        return f"Ramie ma: {s.owner} - poczekaj na koniec albo STOP."
    if s.calib_result_pending:
        return "Kalibracja > 2. Polozenie: sprawdz werdykt w tabeli i kliknij Zapisz wynik kalibracji."
    if s.dyn_result_pending:
        return "Trening > Dynamika serw: Zapisz jako dynamike stanowiska."
    if "identyfikacja" in s.jobs:
        return "Dopasowanie dynamiki do nagrania (7-11 s, ramie juz wolne) - poczekaj na wynik w zakladce Trening."
    if cur is None:
        return ("Procedura przejdziona. Po kazdym przestawieniu kamery: relokalizacja i Sim-Real; po identyfikacji "
                "douczaj polityki (Trening > Start z polityki).")
    return f"krok {cur.number}: {cur.action or cur.detail}."


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
    bg = [j for j in s.jobs if j not in ("fala kalibracyjna",)]
    if s.policy_running:
        bg.append(f"polityka {s.policy_running}" + (" (kostka z kamer)" if s.policy_from_cameras else ""))
    mode = ("symulacja - te same kroki i przyciski co na stanowisku" if s.simulated
            else f"prawdziwe ramie ({s.backend}) - reka przy STOP")
    return Guide(tuple(steps), _now(s, steps), tuple(blockers(s)), tuple(bg), mode)


def render(g: Guide) -> str:
    """Markdown przewodnika - ten sam tekst dla tego samego stanu (panel wysyla tylko zmiany)."""
    lines = [f"**Teraz:** {g.now}"]
    if g.blockers:
        lines.append("")
        lines += [f"- **Uwaga:** {b}" for b in g.blockers]
    lines.append("")
    lines.append(f"Tryb: {g.mode}" + (f" | w tle: {', '.join(g.background)}" if g.background else ""))
    lines.append("")
    for st in g.steps:
        # Lista ma byc krotka (panel nad zakladkami): opis tylko przy biezacym kroku
        # i krotki przy zrobionych.
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
