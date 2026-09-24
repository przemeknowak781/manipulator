"""Przewodnik panelu: kroki procedury (docs/TWIN.md 0-8) i jedna nastepna akcja z prawdziwego stanu.

Czesc bez visera (`guide.build` na migawkach stanu) i czesc na zywym panelu:
kazda kontrolka ma podpowiedz, a przewodnik zmienia sie po polaczeniu ramienia sim.
"""

from __future__ import annotations

import socket

import pytest

from lerobot_mp.twin.ui import guide as gd
from lerobot_mp.twin.ui.guide import CURRENT, DONE, SKIP, TODO, CameraSnap, GuideSnapshot

BUNDLED = (("reach-v3", "reach", True), ("lift-v3", "lift", True))


def _status(g: gd.Guide) -> dict[int, str]:
    return {s.number: s.status for s in g.steps}


def _real_cam(name="usb1", **kw) -> CameraSnap:
    base = dict(simulated=False, has_frame=True, intrinsics_ok=True, calibrated=True, trusted=True)
    base.update(kw)
    return CameraSnap(name, **base)


def test_fresh_sim_starts_at_connecting_and_skips_what_sim_cameras_do_not_need():
    g = gd.build(GuideSnapshot(backend="sim", simulated=True, policies=BUNDLED))
    st = _status(g)
    assert st[0] == CURRENT and g.current.number == 0
    assert "Ramie = sim, **Polacz**" in g.now and g.now.startswith("Krok 0/8")
    assert st[2] == SKIP and st[4] == SKIP                 # K znane, Sim-Real zastepuje prawda symulacji
    assert st[6] == DONE                                   # polityki bazowe sa od razu
    assert [s.number for s in g.steps] == list(range(9))
    assert sum(s.status == CURRENT for s in g.steps) == 1
    assert not g.blockers
    assert "symulacja" in g.mode


def test_sim_guides_through_the_same_steps_with_simulated_cameras():
    s = GuideSnapshot(connected=True, backend="sim", simulated=True, policies=BUNDLED)
    g = gd.build(s)
    assert g.current.number == 1 and "Dodaj symulowana przed ramieniem" in g.now
    cam = CameraSnap("sym1", simulated=True, has_frame=True, intrinsics_ok=True)
    g = gd.build(GuideSnapshot(connected=True, backend="sim", simulated=True, policies=BUNDLED, cameras=(cam,)))
    assert g.current.number == 3
    assert "Start fali" in g.now and "uznaj prawdziwa poze" in g.now
    assert "potwierdz" not in g.now.lower()               # w sim fala bez potwierdzenia karty


def test_real_arm_connected_without_cameras_asks_for_usb_cameras():
    g = gd.build(GuideSnapshot(connected=True, backend="feetech", simulated=False, policies=BUNDLED))
    st = _status(g)
    assert st[0] == DONE and st[1] == CURRENT
    assert "Szukaj kamer USB" in g.now and "Dodaj kamere USB" in g.now
    assert st[2] == TODO and st[4] == TODO                 # na sprzecie nic nie jest "nie dotyczy"
    assert "prawdziwe ramie" in g.mode and "STOP" in g.mode


def test_real_arm_without_connection_points_to_feetech():
    g = gd.build(GuideSnapshot(backend="feetech", simulated=False, arm_error="brak odpowiedzi serw"))
    assert g.current.number == 0 and "feetech" in g.now and "Wykryj porty" in g.now
    assert "brak odpowiedzi serw" in g.current.detail


def test_cameras_without_intrinsics_go_to_charuco_for_the_first_one():
    cams = (_real_cam("usb1", intrinsics_ok=False, calibrated=False, trusted=False,
                      intrinsics_problem="intrynsyki nominalne - najpierw krok 1 (tablica ChArUco)"),
            _real_cam("usb2", intrinsics_ok=False, calibrated=False, trusted=False))
    g = gd.build(GuideSnapshot(connected=True, backend="feetech", simulated=False, cameras=cams, policies=BUNDLED))
    assert g.current.number == 2
    assert "Kamera = usb1" in g.now and "Zbieraj kadry" in g.now
    assert "usb1: intrynsyki nominalne" in g.current.detail and "usb2" in g.current.detail
    # zbieranie kadrow w toku -> co robic z tablica i ktory przycisk potem
    g = gd.build(GuideSnapshot(connected=True, backend="feetech", simulated=False, cameras=cams, policies=BUNDLED,
                               intr_camera="usb1", jobs=("intrynsyki",)))
    assert "Oblicz i zapisz K" in g.now and "pochylona" in g.now


def test_trusted_K_then_card_wave_needs_confirmation_on_the_real_arm():
    cams = (_real_cam(calibrated=False, trusted=False),)
    g = gd.build(GuideSnapshot(connected=True, backend="feetech", simulated=False, cameras=cams, policies=BUNDLED))
    assert _status(g)[2] == DONE and g.current.number == 3
    assert "Karta w szczekach" in g.now and "Start fali" in g.now


def test_estop_blocks_everything_and_says_where_to_clear_it():
    cams = (_real_cam(),)
    g = gd.build(GuideSnapshot(connected=True, backend="feetech", simulated=False, estop=True, cameras=cams,
                               arm_error="STOP: serwo elbow_flex przeciazone", policies=BUNDLED))
    # STOP raz, w "Teraz", z przyczyna - bez drugiej takiej samej uwagi pod spodem
    assert "Skasuj STOP" in g.now and "aktywny STOP" in g.now and "elbow_flex przeciazone" in g.now
    assert not any("STOP" in b for b in g.blockers)
    # przy STOP-ie "Teraz" nie mowi, kto ma ramie - wtedy uwaga; jadacej polityki Uruchom nie odrzuca
    g = gd.build(GuideSnapshot(connected=True, backend="sim", estop=True, owner="polityka", policies=BUNDLED))
    b = next(b for b in g.blockers if b.startswith("ramie ma: polityka"))
    assert "Uruchom zastapi" in b and "i polityka sa odrzucane" not in b


def test_job_running_with_another_owner_is_named_with_how_to_stop_it():
    cams = (_real_cam(calibrated=False, trusted=False),)
    g = gd.build(GuideSnapshot(connected=True, backend="feetech", simulated=False, owner="kalibracja",
                               jobs=("fala kalibracyjna",), cameras=cams, policies=BUNDLED))
    # zadanie uruchomione przez operatora: raz, w "Teraz" (z tym, jak je przerwac), bez alarmu "Uwaga"
    assert "Fala kalibracyjna jedzie" in g.now and "Przerwij" in g.now
    assert not any(b.startswith("ramie ma") for b in g.blockers) and not g.background
    g = gd.build(GuideSnapshot(connected=True, backend="feetech", simulated=False, owner="kalibracja",
                               jobs=("fala kalibracyjna",), cameras=cams, policies=BUNDLED, calib_progress=0.43))
    assert "Fala kalibracyjna jedzie (40%)" in g.now      # postep co 10% - tekst zmienia sie rzadko
    g = gd.build(GuideSnapshot(connected=True, backend="sim", owner="polityka", policy_running="lift",
                               policy_from_cameras=True, policies=BUNDLED))
    assert "lift (kostka z kamer)" in g.now and "Zatrzymaj" in g.now
    assert not any(b.startswith("ramie ma") for b in g.blockers)
    assert not g.background                              # "w tle: polityka" powtarzalo "Teraz"
    g = gd.build(GuideSnapshot(connected=True, backend="sim", owner="identyfikacja dynamiki",
                               jobs=("identyfikacja",), policies=BUNDLED))
    assert "Identyfikacja nagrywa" in g.now


def test_results_waiting_for_save_come_before_the_next_step():
    cams = (_real_cam(calibrated=False, trusted=False),)
    g = gd.build(GuideSnapshot(connected=True, backend="feetech", simulated=False, cameras=cams, policies=BUNDLED,
                               calib_result_pending=True))
    assert "Zapisz wynik kalibracji" in g.now
    g = gd.build(GuideSnapshot(connected=True, backend="feetech", simulated=False, cameras=(_real_cam(),),
                               policies=BUNDLED, dyn_result_pending=True))
    assert "Zapisz jako dynamike stanowiska" in g.now


def test_moved_camera_and_bad_sim_real_are_flagged():
    cams = (_real_cam(moved=True),)
    g = gd.build(GuideSnapshot(connected=True, backend="feetech", simulated=False, cameras=cams, policies=BUNDLED))
    assert g.current.number == 3 and "Relokalizuj wybrana" in g.now and "usb1" in g.now
    assert any("przestawiona" in b for b in g.blockers)
    cams = (_real_cam(simreal_px=5.4),)
    g = gd.build(GuideSnapshot(connected=True, backend="feetech", simulated=False, cameras=cams, policies=BUNDLED))
    assert g.current.number == 4 and "5.5 px" in g.current.detail and "Krawedzie symulacji" in g.now
    # dziesiate czesci piksela nie zmieniaja tekstu (panel wysylalby przewodnik co pomiar)
    a = gd.render(gd.build(GuideSnapshot(connected=True, backend="feetech", simulated=False, policies=BUNDLED,
                                         cameras=(_real_cam(simreal_px=5.4),))))
    b = gd.render(gd.build(GuideSnapshot(connected=True, backend="feetech", simulated=False, policies=BUNDLED,
                                         cameras=(_real_cam(simreal_px=5.6),))))
    assert a == b
    cams = (_real_cam(simreal_px=float("nan")),)
    g = gd.build(GuideSnapshot(connected=True, backend="feetech", simulated=False, cameras=cams, policies=BUNDLED))
    assert g.current.number == 4
    assert "nie sprawdzone" in g.current.detail and "nan" not in g.current.detail
    # zaliczony Sim-Real: bez liczb w liscie
    cams = (_real_cam(simreal_px=2.4),)
    g = gd.build(GuideSnapshot(connected=True, backend="feetech", simulated=False, cameras=cams, policies=BUNDLED))
    assert _status(g)[4] == DONE and "px" not in g.steps[4].detail


def test_warnings_and_lerobot_backend_are_shown():
    g = gd.build(GuideSnapshot(connected=True, backend="lerobot", simulated=False, policies=BUNDLED,
                               warnings=("chwytak: kalibracja LeRobota rozni sie o 60 tikow",)))
    assert "chwytak: kalibracja LeRobota rozni sie o 60 tikow" in g.blockers
    assert any("uzywaj feetech" in b for b in g.blockers)


def test_all_done_on_the_real_arm():
    s = GuideSnapshot(connected=True, backend="feetech", simulated=False, cameras=(_real_cam(simreal_px=1.2),),
                      dynamics="identyfikacja 2026-09-20 (feetech); niepewnosc: kp +-5%",
                      dynamics_backend="feetech", policies=BUNDLED + (("lift-stanowisko", "lift", False),),
                      ran=frozenset({"reach", "lift-kamery"}))
    g = gd.build(s)
    assert all(st.status == DONE for st in g.steps), [(st.number, st.status) for st in g.steps]
    assert g.current is None and "Procedura przejdziona" in g.now
    assert not g.blockers


def test_render_is_stable_and_marks_the_current_step():
    s = GuideSnapshot(connected=True, backend="sim", policies=BUNDLED)
    a, b = gd.render(gd.build(s)), gd.render(gd.build(s))
    assert a == b                                           # ten sam stan = ta sama tresc = nic nie leci
    assert "**Teraz:**" in a and "`[>]` **1. Kamery**" in a and "`[x]` 0." in a
    c = gd.render(gd.build(GuideSnapshot(connected=True, backend="sim", policies=BUNDLED, estop=True)))
    assert c != a and c.startswith("**Teraz:** **aktywny STOP**")
    head, steps = gd.render_head(gd.build(s)), gd.render_steps(gd.build(s))
    assert a == head + "\n\n" + steps and "`[" not in head and "Teraz" not in steps


def test_dynamics_identified_in_sim_does_not_count_on_the_real_arm():
    src = "identyfikacja 2026-09-24T10:00:00 (sim); niepewnosc: kp +-4%"
    assert gd.dynamics_backend(src) == "sim"
    assert gd.dynamics_backend("identyfikacja 2026-09-24 (feetech); niepewnosc: x") == "feetech"
    assert gd.dynamics_backend("zmierzona") == ""
    base = dict(connected=True, cameras=(_real_cam(simreal_px=1.0),), policies=BUNDLED, dynamics=src,
                dynamics_backend="sim")
    g = gd.build(GuideSnapshot(backend="feetech", simulated=False, **base))
    assert g.current.number == 5 and "zmierzona na sim" in g.current.detail and "Identyfikuj" in g.now
    g = gd.build(GuideSnapshot(backend="sim", simulated=True, **base))
    assert _status(g)[5] == DONE                           # proba procedury w sim - wystarcza


def test_real_arm_with_bundled_policies_only_asks_for_fine_tuning():
    s = dict(connected=True, backend="feetech", simulated=False, cameras=(_real_cam(simreal_px=1.0),),
             dynamics="identyfikacja t (feetech); x", dynamics_backend="feetech", policies=BUNDLED)
    g = gd.build(GuideSnapshot(**s))
    assert g.current.number == 6
    assert "Start z polityki = reach-v3" in g.now and "**Ucz**" in g.now
    assert _status(gd.build(GuideSnapshot(**{**s, "simulated": True, "backend": "sim",
                                             "cameras": ()})))[6] == DONE


def test_trusted_pose_with_nominal_K_is_not_a_trusted_camera():
    # stara kalibracja bez zapisanego K: CameraRecord.trusted == True przy nominalnym K
    cams = (_real_cam(intrinsics_ok=False, intrinsics_problem="intrynsyki nominalne", simreal_px=1.0),)
    g = gd.build(GuideSnapshot(connected=True, backend="feetech", simulated=False, cameras=cams, policies=BUNDLED))
    st = _status(g)
    assert st[2] == CURRENT and st[3] == TODO and st[4] == TODO
    assert "niezaufanego K" in g.steps[3].detail


def test_policy_steps_are_per_session_checks():
    s = GuideSnapshot(connected=True, backend="sim", cameras=(CameraSnap("sym1", True, trusted=True,
                                                                          calibrated=True, intrinsics_ok=True),),
                      policies=BUNDLED, dynamics="identyfikacja t (sim); x", dynamics_backend="sim",
                      ran=frozenset({"reach"}))
    g = gd.build(s)
    assert g.current.number == 8 and "w tej sesji" in g.steps[7].detail
    assert "**lift (sim): poloz kostke losowo**" in g.now and "kamery" in g.now


# ------------------------------------------------------------------ na zywym panelu
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def panel(tmp_path_factory):
    pytest.importorskip("viser")
    pytest.importorskip("mujoco")
    from lerobot_mp.twin.ui.app import TwinApp

    app = TwinApp(tmp_path_factory.mktemp("przewodnik") / "twin.json", host="127.0.0.1", port=_free_port())
    yield app
    app.close()


def test_every_interactive_control_has_a_hint(panel):
    handles = list(panel.server.gui._gui_input_handle_from_uuid.values())
    # Kontrolka = uchwyt z podpowiedzia w propsach (przycisk, suwak, pole, lista, wektor...);
    # pasek postepu jej nie ma.
    controls = [h for h in handles if hasattr(h, "hint")]
    assert len(controls) >= 70
    missing = [(type(h).__name__, getattr(h, "label", "?")) for h in controls if not (h.hint or "").strip()]
    assert not missing
    # takze kontrolki z petli: suwaki stawow
    for name, s in panel.sliders.items():
        assert s.hint and ("Chwytak" in s.hint if name == panel.ws.spec().gripper else "st." in s.hint)


def test_guide_changes_when_the_sim_arm_connects(panel):
    panel.arm_backend.value = "sim"
    panel._tick_slow()
    before = panel.guide_md.content
    assert "Ramie = sim, **Polacz**" in before and "`[>]` **0." in panel.guide_steps_md.content
    try:
        panel.twin.connect("sim")
        panel._tick_slow()
        after = panel.guide_md.content
        assert after != before
        steps = panel.guide_steps_md.content
        assert "`[x]` 0." in steps and "polaczone: sim" in steps
        assert "`[x]` 0." not in after                      # lista krokow tylko w zwinietym folderze
    finally:
        panel.twin.disconnect()
    panel._tick_slow()
    assert panel.guide_md.content == before


def test_runner_success_counts_only_for_the_backend_it_started_on(panel):
    """Stary runner z sukcesem (lift z kamer w sim) nie zalicza kroku 8 ramieniu po nowym Polacz."""
    from types import SimpleNamespace as NS
    saved = panel.runner, panel._runner_backend, {k: set(v) for k, v in panel._ran.items()}
    try:
        panel.twin.connect("sim")
        panel._ran.clear()
        panel.runner = NS(status=NS(running=False, success=True), task=NS(name="lift"),
                          cube_provider=panel._vision_cube, stop=lambda *a, **k: None)
        panel._runner_backend = "feetech"                  # sukces z poprzedniego polaczenia, innego ramienia
        assert panel._guide_snapshot().ran == frozenset()
        assert "sim" not in panel._ran
        panel._runner_backend = "sim"                      # ruszyl na tym polaczeniu
        assert panel._guide_snapshot().ran == frozenset({"lift-kamery"})
    finally:
        panel.runner, panel._runner_backend = saved[0], saved[1]
        panel._ran.clear()
        panel._ran.update(saved[2])
        panel.twin.disconnect()


def test_camera_without_frame_is_flagged_only_after_a_few_seconds(panel, monkeypatch):
    import time

    from lerobot_mp.twin.ui.app import GUIDE_FRAME_LOST
    from lerobot_mp.twin.workspace import CameraRecord

    rec = CameraRecord(name="usb_przewodnik", source="7")
    monkeypatch.setattr(panel.ws, "cameras", [rec])
    with panel.frame_lock:
        frames = panel.frames
        panel.frames = {"usb_przewodnik": object()}
    try:
        assert panel._guide_snapshot().cameras[0].has_frame
        with panel.frame_lock:
            panel.frames = {}
        # przestoj USB krotszy niz prog: bez uwagi i bez przelaczania kroku 1
        panel._frame_seen["usb_przewodnik"] = time.monotonic() - (GUIDE_FRAME_LOST - 0.5)
        assert panel._guide_snapshot().cameras[0].has_frame
        panel._frame_seen["usb_przewodnik"] = time.monotonic() - (GUIDE_FRAME_LOST + 0.5)
        assert not panel._guide_snapshot().cameras[0].has_frame
    finally:
        with panel.frame_lock:
            panel.frames = frames
        panel._frame_seen.pop("usb_przewodnik", None)
