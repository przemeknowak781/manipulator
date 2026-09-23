# Handoff: cyfrowy bliźniak SO-101 → DGX Spark

Stan na 2026-09-23. Praca przenosi się z laptopa z Windows na NVIDIA DGX Spark.
Instrukcja uruchomienia: **[docs/TWIN.md](docs/TWIN.md)**.

## W skrócie

Aplikacja jest przerabiana na **uniwersalnego cyfrowego bliźniaka**: wiele
kamer skalibrowanych względem ramienia, scena MuJoCo odtwarzająca stanowisko
i środowisko do uczenia ze wzmocnieniem, z dobrym UI do zarządzania ramieniem
i kamerami. Kalibracja pochodzi z [galaxeo-manipulators PR #3](https://github.com/machinekind/galaxeo-manipulators/pull/3).

**Fundament jest gotowy i przetestowany** — model, kinematyka, scena z kamerami
o pełnym K, kolizje, kalibracja wielu kamer (0,15 mm / 0,055° w symulacji),
mapa stołu, zapis stanowiska, pętla sterowania. **Nie ma jeszcze** środowiska
Gymnasium, panelu webowego ani kalibracji na prawdziwych kamerach — dla
wszystkich trzech projekt jest rozpisany niżej, łącznie ze sprawdzonym API.

Ostatnia sesja celowo **nie liczyła nic na GPU** (decyzja użytkownika) —
optymalizacja renderu i trening czekają na Sparka.

---

## 1. Pierwsze kroki na Sparku

```bash
git clone https://github.com/przemeknowak781/manipulator.git && cd manipulator
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[twin,dev]"
export MUJOCO_GL=egl
lerobot-twin check                     # bez renderu: pakiety, model, zgodność z Articulusem
pytest -q                              # PEŁNY zestaw, łącznie z renderem
lerobot-twin calib-sim --n 5 --cameras 2
```

Wszystkie zależności bliźniaka mają koła dla Linux aarch64 (sprawdzone na PyPI).

**Do sprawdzenia na Sparku, bo na laptopie nie było jak:**

- [ ] 4 testy renderujące (`test_sim_camera_renders_where_its_intrinsics_project` ×3,
      `test_full_session_in_simulation_calibrates_the_camera`) — w ostatnim
      przebiegu odznaczone na prośbę użytkownika; przechodziły przed zmianą
      domyślnego położenia stołu, po niej nie były uruchamiane.
- [ ] Czas renderu z EGL na GB10 (tabela w sekcji 4) — to zdecyduje o tym, jak
      zbudować obserwacje obrazowe w RL.
- [ ] PyTorch z CUDA na aarch64 — dobrać właściwe koło/kontener NVIDIA dla Sparka;
      nie weryfikowane.
- [ ] Czy MJX / MuJoCo Warp działa na GB10 — to byłaby droga do tysięcy
      równoległych środowisk; nie weryfikowane.

Stan testów na laptopie: **258 przechodzi, 4 odznaczone (render), 1 pada** —
`test_mapping.py::test_direct_and_ik_move_the_tip_the_same_way`, porażka
starsza niż bliźniak i z nim niezwiązana (zbyt sztywny próg w samym teście).

---

## 2. Co jest gotowe

| moduł | co robi | dowód |
|---|---|---|
| `twin/robots.py` | opis ramienia: MJCF, stawy, TCP, osie chwytaka, zakresy fali | test osi na geometrii modelu |
| `twin/kinematics.py` | FK/IK na MuJoCo; IK z **priorytetem pozycji** (orientacja w przestrzeni zerowej) | zgodność z Articulusem **2 µm**; IK trafia 23+/25 osiągalnych celów |
| `twin/scene.py` | scena `MjSpec`: stół, ramię, kamery z **pełnym K**, karta z tagami, obiekty | render vs rzut K: **< 0,15 px** systematycznie |
| `twin/collision.py` | kolizje pozy i **drogi** do niej | stół i samokolizje wykrywane |
| `twin/calib/handeye.py` | solver **wielu kamer ze wspólną kartą** | 1 i 2 kamery z szumu 0,2 px |
| `twin/calib/session.py` | fala: szukanie → celowanie → pilnowanie kadru i rozrzutu → bramki | ~15–19 poz na sesję |
| `twin/calib/simulate.py` | sesja w symulacji oceniana względem prawdy | 6/6 kamer: **0,15 mm, 0,055°** |
| `twin/calib/topdown.py` | mapa stołu z jednej i wielu kamer, z dystorsją | kostka trafia **0,3 px** od miejsca |
| `twin/calib/card.py` | karta: geometria, poza w szczękach, arkusz A4 | oba tagi wykrywane z arkusza |
| `twin/workspace.py` | stanowisko w JSON: ramię, port, stół, karta, kamery + wynik kalibracji | testy zapisu/odczytu |
| `twin/cameras.py` | wykrywanie kamer, strumienie **bez lustra**, kamery symulowane | **nietestowane na sprzęcie** — laptop nie ma kamery |
| `twin/runtime.py` | pętla sterowania w wątku, nadzór bezpieczeństwa, sim albo sprzęt | sprawdzone ręcznie na symulacji |
| `twin/cli.py` | `lerobot-twin check / card / calib-sim / workspace` | test arkusza |
| `control/safety.py` | nowy parametr `start(..., go_home=False)` — łączenie bez ruchu | domyślne zachowanie bez zmian |

---

## 3. Decyzje i dlaczego

- **MuJoCo + MuJoCo Menagerie** (`robotstudio_so101`, Apache-2.0) zamiast
  własnego modelu: pochodzi z `so101_new_calib.xml`, czyli tej samej konwencji
  zera co backend `feetech` — kąty z serw idą do symulacji bez przesunięć.
  Ma proste kształty kolizyjne i dostrojone parametry chwytu.
- **Scena przejmuje opcje fizyki ramienia** (stożek eliptyczny, `impratio`, krok
  5 ms). `MjSpec.attach` domyślnie zostawia opcje sceny i tylko ostrzega.
- **IK z priorytetem pozycji.** SO-101 ma 5 osi; ważenie błędu obrotu i pozycji
  w jednej sumie odciągało ramię o 20 cm od celu przy nieosiągalnej orientacji.
- **Fala kalibracyjna losuje w przestrzeni STAWÓW**, a nie kartezjańsko jak
  galaxeo (A1X ma 6 osi) — każda poza jest osiągalna z definicji; obrót
  nadgarstka dobierany tak, żeby karta patrzyła na kamerę.
- **Fala pilnuje kadru.** Pierwsza detekcja daje pełną pozę kamery z PnP;
  kandydaci, dla których karta wypadłaby z kadru, są odrzucani. Bez tego 30
  z 40 poz było nad kadrem, a sesja zbierała 8 obserwacji w 90 pozach.
- **Pełne K w symulacji** (`focal_pixel`, `principal_pixel`), nie samo `fovy`.
- **Panel webowy na viserze** (jeszcze niezbudowany): przeglądarka, więc działa
  lokalnie i zdalnie; ma piramidy kamer z obrazem na żywo i uchwyt do
  przeciągania końcówki.
- **Gymnasium** jako API środowiska: od razu pasuje do SB3, LeRobota, CleanRL.

---

## 4. Liczby, które warto znać

| | |
|---|---|
| budowa sceny (`scene.build`) | **293 ms** — nie przebudowywać przy każdym resecie |
| 1 s fizyki | 6,9 ms (145× szybciej niż czas rzeczywisty) |
| render 640×480 na Intel HD 620 | 164 ms domyślnie · 79 ms bez cieni · 70 ms bez cieni i MSAA · **8 ms bez siatek ramienia** |
| trójkąty siatek ramienia | **348 164** — to one są wąskim gardłem renderu, nie karta |
| pętla sterowania na Windows | 28 Hz zamiast zadanych 50 |

---

## 5. Następne kroki (w kolejności)

### 5.1 Środowisko Gymnasium — `twin/env.py`

Projekt ustalony, kodu jeszcze nie ma.

- Scena budowana **raz** w `__init__`; randomizacja przy `reset` przez pola
  skompilowanego modelu: `model.cam_pos` / `cam_quat` (drżenie w granicach
  niepewności kalibracji), `light_pos` / `light_diffuse`, `geom_rgba`,
  położenia obiektów przez `qpos`. Zmiana rozmiaru obiektu wymaga przebudowy.
- **Akcja**: przyrosty celów stawów w [-1, 1] × maks. krok na takt (domyślnie),
  opcjonalnie cele bezwzględne.
- **Obserwacja**: `state` (stawy + TCP + dane zadania), `images` z każdej
  skalibrowanej kamery (render w natywnej rozdzielczości, potem zmniejszenie —
  geometria kalibracji zostaje), opcjonalnie **zszyta mapa stołu** z
  `calib/topdown.fused` — w galaxeo to ona dała pierwsze udane polityki.
- **Zadania**: `reach` (nagroda −odległość, sukces < 2 cm) i `lift` (dojazd +
  kontakt obu szczęk + wysokość kostki).
- Rejestracja: `LeRobotMP/TwinReach-v0`, `LeRobotMP/TwinLift-v0`.
- Testy: `gymnasium.utils.env_checker.check_env` + skryptowy ekspert z IK, który
  rozwiązuje `reach` (dowód, że zadanie jest wykonalne).
- Render dla RL: cienie wyłączone (`model.light_castshadow[:] = 0`),
  `model.vis.quality.offsamples = 0`; rozważyć zdziesiątkowane siatki wizualne.

### 5.2 Panel webowy (viser 1.1.1) — `twin/ui/`

API sprawdzone na zainstalowanej wersji:

- `viser.ViserServer(host, port, label)`, `server.scene.set_up_direction("+z")`.
- **Most MuJoCo → viser**: jeden węzeł na geom wizualny (grupy 0–2), siatka
  z `model.mesh_vert` / `mesh_face` przez `geom_dataid`, poza co takt z
  `data.geom_xpos` / `geom_xmat` w `server.atomic()`.
- **Kamery w 3D**: `scene.add_camera_frustum(name, fov, aspect, image=..., wxyz, position)`;
  `handle.image` da się podmieniać — miniatury na żywo na piramidach.
  `fov = 2·atan(H / 2 / fy)`.
- **Uchwyt końcówki**: `scene.add_transform_controls` → `on_update` → `kin.ik` →
  `twin.set_target` (tylko przy włączonym sprzęgle).
- **Zakładki**: Ramię (sim / port szeregowy, wykrywanie CH343 po VID `0x1A86`,
  sprzęgło, STOP, dom, suwaki stawów z blokadą sprzężenia zwrotnego, chwytak) ·
  Kamery (`probe_devices`, dodaj/usuń, podgląd, stan kalibracji) · Kalibracja
  (zmierzony bok taga, arkusz przez `send_file_download`, sesja w wątku z paskiem
  postępu, `apply_fit` + zapis + przebudowa sceny) · Sim↔Real (nałożenie kadru
  i renderu z suwakiem przezroczystości, mapa stołu) · RL.
- Pod spodem jest gotowy `runtime.Twin`: `connect(backend, port, go_home=False)`,
  `set_engaged`, `set_target`, `move`, `home`, `estop`, `status`, `lock`.

### 5.3 Kalibracja na prawdziwym ramieniu

Kawałki istnieją; brakuje polecenia i intrynsyk.

1. **Najpierw intrynsyki** — `twin/calib/intrinsics.py` (ChArUco w OpenCV).
   Do tej pory kamera ma nominalne K z 65° pola widzenia; z takim K pozy kamer
   będą obciążone.
2. Potem sesja:
   ```python
   ws = Workspace.load(); tw = Twin(ws); tw.connect("feetech", "COM12")
   tw.cameras.sync()
   card = ws.card_obj(); nominal = card.nominal(pinch_point(RobotKinematics(ws.spec())))
   checker = CollisionChecker(sc.build(ws.scene_config(with_card=True, card_pose=nominal, card_collider=0.012)))
   intr = {c.name: c.intrinsics() for c in ws.cameras if c.enabled}
   fit = Session(tw, tw.cameras, intr, tw.scene.kin, card, nominal, checker).run(print)
   ws.apply_fit(fit); ws.save()
   ```
   Karta musi być w szczękach **przed** startem — pierwszy ruch zamyka chwytak.
   Stół w bliźniaku musi się zgadzać z prawdziwym (wysokość podstawy = blat).

### 5.4 Rekonstrukcja 3D z biurka i dobór chwytu

Raport: **[docs/RESEARCH_3D.md](docs/RESEARCH_3D.md)**, prototypy z pomiarami:
[`research/3d/`](research/3d/). Najważniejsze:

- **Pierwszy krok bez sieci neuronowych, na CPU**: maski z różnicy względem
  pustego stołu + render ramienia z bliźniaka → wspólna otoczka wizualna ze
  znanych póz kamer (1,3–2,3 s, 94–100% pokrycia bryły) → CoACD → obiekt
  w scenie → chwyt analityczny sprawdzony w MuJoCo.
- **Piksele ramienia to „nie wiem", nie tło** — inaczej otoczka gubi do 85% obiektu.
- **SO-101 chwyta tylko w pionowej płaszczyźnie przez oś podstawy** (5 osi);
  z góry dla r = 15–30 cm, z boku dopiero od 25 cm. Szczęka na zawiasie:
  klocek 60 mm chwycony w środku wypada, przesunięty na stałą szczękę — trzyma.
- Z Mety: SAM 3.1 (maski z tekstu), MapAnything (jedyny, który bierze nasze
  pozy kamer i daje skalę; wariant wag na Apache), SAM 3D Objects (domykanie
  kształtu, ≥ 32 GB — na Sparku, ale kaolin trzeba zbudować pod sm_121).
- Do rozstrzygnięcia: komercyjność projektu (połowa modeli ma licencje NC).

### 5.5 Drobniejsze

- Wykrywanie portu ramienia po VID `0x1A86` — numer COM zmienia się między sesjami.
- Pętla 28 Hz na Windows zamiast 50 — zbadać.
- Kalibracja kamery na nadgarstku (oko-w-dłoni) — solver trzeba odwrócić:
  kamera jedzie z chwytakiem, karta stoi na stole.

---

## 6. Pytania do rozstrzygnięcia

1. **Licencja galaxeo-manipulators.** Repozytorium nie ma pliku licencji.
   `calib/tags.py`, `calib/handeye.py` i idea `calib/topdown.py` są przeniesione
   z atrybucją do tego repo (Apache-2.0). Zakładamy zgodę machinekind — do potwierdzenia.
2. Które zadania RL są priorytetem (reach / lift / pick-and-place / nalewanie jak w galaxeo)?
3. Jakie kamery trafią na stanowisko (model, rozdzielczość) i czy nadgarstkowa?
4. Trening na Sparku: MJX / MuJoCo Warp (tysiące środowisk na GPU) czy wektory CPU?

---

## 7. Sprzęt

| | ramię nr 1 (sierpień) | ramię nr 2 (wrzesień, bieżące) |
|---|---|---|
| przejściówka | CH343 `5AAF219949` | CH343 `5AAF220303` |
| port | COM11 | COM12 |
| limity serw (EEPROM) | były zawężone (bark −2…+84°); **rozszerzone do 0…4095** 2026-08-31, oryginał w [`docs/hardware/servo_limits_arm1.json`](docs/hardware/servo_limits_arm1.json) | wyglądają na prawdziwą kalibrację LeRobota (±100–125°) |
| uwagi | chwytak 1986…2670 tików (z jego EEPROM-u) | stał złożony (`shoulder_lift −101°`, poza limitem −95° z konfiguracji — przy połączeniu nadzór dociąga go do −95°); `wrist_flex` w serwie max +88° przy +95° w konfiguracji — backend przycina i ostrzega |

Przywrócenie oryginalnych limitów ramienia nr 1: dla każdego serwa zapis
`Lock`(55)=0 → `Min_Angle_Limit`(9) / `Max_Angle_Limit`(11) → `Lock`=1; przed
zmianą ustawić cel = bieżącą pozycję, inaczej zmiana limitu odblokuje stary cel
i ramię ruszy.

---

## 8. Pułapki, które cicho psują sim-2-real

Każda jest teraz pilnowana testem albo poprawiona w kodzie — ale warto je znać
przy kolejnym ramieniu, kamerze czy wersji bibliotek:

- **Znak punktu głównego w MuJoCo** — przy odwrotnym kadr przesuwa się o
  dwukrotność przesunięcia (62 px przy 30 px).
- **Półpiksel w pionie w renderze MuJoCo** — stale 0,5 px wyżej niż rzut K.
- **Półpiksel w detektorze AprilTag** — `CORNER_REFINE_APRILTAG` oddaje rogi
  w konwencji narożników pikseli; poprawka przepołowiła błąd obrotu kamery.
  Dotyczy też galaxeo.
- **`CameraStream` z aplikacji odbija obraz lustrzanie** (wygodne przy dłoni);
  bliźniak otwiera kamery z `mirror=False`.
- **`MjSpec.attach` gubi opcje fizyki ramienia**, jeśli scena ich nie przejmie.
- **Nadzór dociąga ramię do limitów z konfiguracji** przy połączeniu — ramię
  stojące poza nimi „skoczy” o kilka stopni.
