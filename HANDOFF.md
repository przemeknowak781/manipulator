# Handoff: cyfrowy bliźniak SO-101 na Shadow

Stan na 2026-09-24. Plan przeniesienia na NVIDIA DGX Spark zastąpiony
maszyną **Shadow** (Windows 11, RTX A4500 20 GB, EPYC 4 rdzenie / 8 wątków,
28 GB RAM). Instrukcja uruchomienia i procedura testu na sprzęcie:
**[docs/TWIN.md](docs/TWIN.md)**.

## W skrócie

System jest gotowy do pierwszego testu na fizycznym stanowisku. Wszystko,
co da się sprawdzić bez ramienia i kamer, jest sprawdzone na bliźniaku —
tą samą ścieżką kodu, którą pójdzie sprzęt:

- **panel webowy** (`lerobot-twin ui`) — ramię, kamery (także symulowane,
  stawiane i przeciągane w 3D), kalibracja, mapa na żywo, trening, polityki,
  porównanie sim-real; przećwiczony cały scenariusz w przeglądarce;
- **kalibracja**: intrynsyki z ChArUco, położenie wielu kamer z karty w
  chwytaku, szybka relokalizacja, wykrywanie przestawionej kamery;
- **trening na GPU**: MuJoCo Warp (tysiące światów) + PPO; `reach` w 2 min,
  `lift` w 28 min; polityki przechodzą na CPU (inny silnik) z 96–100% sukcesu;
- **kalibracja treningu sim-to-real**: identyfikacja dynamiki serw z nagrania
  ruchu, randomizacja wokół zmierzonej dynamiki;
- **percepcja**: kostka z kamer (część wspólna masek na wysokości górnej
  ściany), śledzenie w dłoni, gdy szczęki ją zasłaniają;
- **runner**: polityka na bliźniaku albo ramieniu przez `SafetySupervisor`,
  z zatrzymaniem przy rozjeździe i po wykonaniu zadania; `lift-v3` z kostką
  **wyłącznie z kamer** podnosi ją na bliźniaku 8/8 (`lift-v1`, uczona na
  prawdzie: 1/8);
- **polityki bazowe** w `assets/policies/` (`reach-v2`, `lift-v3`) — start do
  douczania na zmierzonej dynamice;
- **przegląd przed sprzętem**: 53 potwierdzone błędy (z czego 13 poważnych)
  poprawione i sprawdzone ponownie — sekcja 2a.

---

## 1. Co się zmieniło względem planu DGX

| DGX Spark (plan) | Shadow (jest) |
|---|---|
| Linux aarch64, `MUJOCO_GL=egl` | Windows, render przez GLFW/WGL na A4500 — 2 ms za kadr 640×480 |
| MJX / JAX na GPU | **MuJoCo Warp** — natywnie na Windows, 1,1 mln kroków fizyki/s; JAX z CUDA na Windows nie istnieje, WSL2 nie ruszy (brak zagnieżdżonej wirtualizacji) |
| PyTorch aarch64 do dobrania | koło `cu126` z indeksu PyTorcha; ze sterownikiem 565.90 (CUDA 12.7) działa |
| ramię przy laptopie | przepuszczenie USB w kliencie Shadow **albo** most `lerobot-mp-bridge` i port `socket://` (nowe) |

Pozycje z listy „do sprawdzenia na Sparku”: testy renderujące — przechodzą;
czas renderu — zmierzony; PyTorch z CUDA — działa; MJX/Warp — Warp działa.

---

## 2. Co jest gotowe

| moduł | co robi | dowód |
|---|---|---|
| `twin/scene.py` | + panele z obrazem, czujniki kontaktu szczęk; **punkt główny bez fałszywego półpiksela** | test na maskach segmentacji; kalibracja 4× dokładniejsza w obrocie |
| `twin/calib/intrinsics.py` | ChArUco: arkusz, zbieranie ujęć z pilnowaniem różnorodności, K + dystorsja, bramki | K z renderu do ~1 px, rogi 0,002 px od rzutu |
| `twin/perception.py` | mapa stołu na żywo, kostka z kamer, `CubeTracker` (w dłoni) | 1,2 mm / 0,6° z dwóch kamer |
| `twin/rl/task.py` | obserwacja/akcja/nagroda — numpy i torch, te same wzory | test równości numpy/torch |
| `twin/rl/env.py` | Gymnasium CPU, `check_env` | ekspert IK rozwiązuje `reach` 10/10 |
| `twin/rl/batch.py` | MuJoCo Warp, randomizacja per świat + `set_const` | zgodność z CPU 1e-7 → 4e-3 po 60 krokach losowych |
| `twin/rl/ppo.py`, `policy.py`, `evaluate.py` | PPO (także douczanie `--init`), plik polityki z zadaniem i wynikami, ewaluacja na CPU | `reach-v2`, `lift-v3` w `assets/policies` |
| `twin/rl/sysid.py` | nagranie ruchu przez nadzór, dopasowanie tłumienia/armatury/opóźnienia (kp i tarcie suche z tego ruchu niewyznaczalne — zostają z modelu) | prawdy spoza siatki startowej: tłumienie i armatura ~2%, opóźnienie 0–1,3 ms, dopasowanie na poziomie szumu |
| `twin/rl/runner.py` | polityka na `Twin`, stop przy rozjeździe > 25° (w zmierzonej pozie), stop po wykonaniu zadania, czekanie na kostkę | test na bliźniaku; pełny łańcuch z kamerami w czasie symulowanym (`test_twin_lift_from_cameras`) |
| `twin/ui/*` | panel (viser), zadania w tle, most sceny, strażnik kamer | test startu panelu + scenariusz w przeglądarce |
| `twin/runtime.py` | pętla ramienia: jeden właściciel ruchu, STOP/Dom w zmierzonej pozie, reakcja na błędy serw i utratę łącza, render bez blokady pętli | `tests/test_twin_runtime.py` (czas symulowany) |
| `robot/feetech.py` | + `socket://` (most, `TCP_NODELAY`), SYNC READ z powrotem do odczytów po kolei, suma kontrolna odpowiedzi, bity błędów serw (`faults()`), limity z EEPROM (`joint_limits()`) | testy na symulowanej magistrali |

Testy: `pytest -q` — wszystko poza znanym, starym `test_mapping.py::test_direct_and_ik_move_the_tip_the_same_way`
(za sztywny próg w samym teście, sprzed bliźniaka).

---

## 2a. Przegląd przed testem na sprzęcie

Przegląd całego kodu bliźniaka (wyszukiwanie → trzech sceptyków na każde
znalezisko → krytyk) potwierdził 53 błędy; poprawione na czterech gałęziach
rozłącznych plikowo (`fix-feetech`, `fix-runtime`, `fix-perception-rl`,
`fix-ui`) wobec spisanych kontraktów, potem sprawdzone ponownie przez
niezależnych weryfikatorów. Najważniejsze:

| było | teraz |
|---|---|
| uchwyt końcówki w 3D potrafił przerzucić IK na drugą gałąź — obrót o ~180° | IK tylko od bieżącej pozy, rozwiązanie z ruchem stawu > 15° odrzucone |
| *Połącz* i *Dom* nie zatrzymywały fali / identyfikacji / polityki; po rampie do domu ramię wracało skokiem do starego celu | jeden właściciel ramienia (`Twin.claim`), STOP/Dom/Połącz/Rozłącz go odbierają, ramię trzyma zmierzoną pozę |
| stop przy kolizji trzymał zablokowany cel — serwo dalej dociskało | stop trzyma **zmierzoną** pozę |
| odpowiedzi serw bez sprawdzenia sumy kontrolnej, bity błędów gubione | uszkodzona ramka odrzucona, błąd serwa → STOP awaryjny z powodem |
| po zerwaniu łącza zaległe cele szły seriami (skok 95°) | przy utracie łącza nic nie jest wysyłane, po powrocie nadzór startuje od świeżego odczytu |
| chwytak 0..100 w bliźniaku 1,8× szerzej niż na serwie | te same tiki co backend `feetech` |
| jedna kamera „kładła” na blacie kostkę uniesioną w dłoni | wykrycie z jednej kamery przy dłoni pomijane |
| intrynsyki z tablicy trzymanej płasko do obiektywu uznawane za zaufane | wymagany rozrzut pochyleń ≥ 35° i wiarygodna ogniskowa |
| identyfikacja „wyznaczała” kp, którego ruch nie rozróżnia | dopasowuje tylko to, co wyznaczalne, i mówi to wprost |
| TCP i kostka w obserwacji o krok fizyki za kątami stawów | kinematyka liczona po fizyce (CPU i GPU) |
| render trzymał blokadę pętli ramienia | render na kopii stanu |

Polityki uczone przed poprawkami (inny chwytak, inne obserwacje, epizod `lift`
bez końca) zastąpione douczonymi `reach-v2` (CPU 100% / 98%, 1,5 mm) i
`lift-v3` (CPU 100% / 100%, z kamer 8/8, kostka ~11 cm zamiast 17). Przy tym
wyszedł błąd douczania — krytyk od zera psuł dobrą politykę; pliki polityk
niosą teraz krytyka, a stare dostają rozgrzewkę (TWIN.md, *Polityki bazowe*).

---

## 3. Decyzje i dlaczego

- **Jedna definicja zadania dla CPU, GPU i ramienia** (`rl/task.py` z `xp` =
  numpy albo torch). Dwie kopie „prawie tej samej” obserwacji to ciche źródło
  sim-2-real.
- **Akcja jako przyrost celu stawów, chwytak bezwzględnie**, oba ograniczone
  tak jak `SafetySupervisor` — w treningu polityka nie może robić tego, czego
  nadzór na ramieniu nie przepuści.
- **Trening w osobnym procesie** — CUDA przez kilkanaście minut; padnięcie
  sterownika nie może zabrać panelu ani pętli ramienia. Zatrzymanie plikiem
  `STOP` w katalogu przebiegu, polityka z tego miejsca zostaje zapisana.
- **Kamera symulowana ma prawdziwą pozę (`sim_pose`) osobno od kalibracji** —
  kalibruje się ją jak prawdziwą i widać błąd względem prawdy.
- **Kostka na mapie górnej ściany, część wspólna masek** — średnia kamer
  rozmazywała boki (12–17 mm), część wspólna daje 1–2 mm.
- **Śledzenie kostki w dłoni** — wizja gubi kostkę dokładnie przy chwycie;
  szczęka zatrzymana na obiekcie + kostka ostatnio przy szczękach = jedzie z TCP.
- **Jeden wątek renderujący** w `Twin` — kontekst GL może być bieżący tylko w
  jednym wątku; wcześniej każdy wątek (panel, fala, mapa) renderował sam.
- **Trening widzi kostkę tak jak percepcja, nie jak fizyka.** Zmierzone na
  CPU: `lift` uczona na prawdzie — 20/20 na prawdzie, 19/20 ze złożonym
  obrotem, 19/20 z szumem 3 mm, **7/20** z odświeżaniem 10 Hz i opóźnieniem
  150 ms. Stąd `Randomization.cube_*`; `lift-v2` jest uczona już z nimi.
- **Kostka z kamer = dopasowanie sylwetki bryły**, a nie sama część wspólna
  masek — ta druga potrzebuje dwóch kamer, a ramię zasłania kostkę jednej
  z nich w połowie epizodów.
- **Jeden właściciel ruchu ramienia** (`panel`, `polityka`, `kalibracja`,
  `identyfikacja`) zamiast wielu wątków piszących cel na zmianę. STOP, Dom,
  Połącz i Rozłącz odbierają ramię każdemu przez `Twin.preempt`.
- **Błąd serwa = STOP.** Także przeciążenie chwytaka; jeśli na sprzęcie zwykły
  mocny chwyt będzie je wyzwalał, trzeba rozdzielić chwytak od reszty (decyzja
  po teście, nie na zapas).
- **Koniec epizodu po sukcesie liczony jak limit czasu** — PPO dalej
  bootstrapuje wartość stanu; zero na końcu uczyło polityki wisieć tuż pod
  progiem sukcesu.

---

## 4. Następne kroki

1. **Test na sprzęcie** według [docs/TWIN.md](docs/TWIN.md) — kroki 0–8.
   Najpierw połączenie i suwaki, potem intrynsyki, położenie, sim-real,
   identyfikacja, dopiero potem polityki.
2. **Po identyfikacji douczyć polityki** (`--init`, w panelu *Start z
   polityki*) — domyślne są uczone wokół modelu Menagerie. Douczanie zamiast
   uczenia od zera: `lift` z modelem percepcji od zera po 220 iteracjach miała
   0% sukcesu, a douczana z `lift-v1` startuje od umiejętności chwytu.
   **Przy pierwszym połączeniu sprawdzić zero chwytaka**: chwytak 0 w bliźniaku
   to −5,4° kąta szczęki (tiki z konfiguracji); jeśli na ramieniu przy 0
   szczęki się nie stykają, zero szczęki w MJCF nie leży w `center_ticks`.
3. **Obserwacje obrazowe w RL** — renderer MuJoCo Warp (wsadowy, na GPU)
   zgadza się z OpenGL do 0,05 px; środowisko obrazowe to kolejny krok
   (mapa stołu z kamer jako wejście, jak w galaxeo).
4. **Kalibracja oko-w-dłoni** (`wrist_cam`) — solver odwrócony: karta na stole.
5. **Rekonstrukcja 3D przedmiotów** — [docs/RESEARCH_3D.md](docs/RESEARCH_3D.md);
   MuJoCo Warp ma render gaussian splats, co pozwala wstawić zeskanowany
   przedmiot do renderu treningowego.
6. Solver kalibracji bez odpornej funkcji straty — jedna zła obserwacja potrafi
   podnieść residuum do 1 px (ziarno 4 w `calib-sim`); warto dodać odrzucanie.

---

## 5. Pytania do rozstrzygnięcia

1. **Licencja galaxeo-manipulators** — `calib/tags.py`, `calib/handeye.py`
   i idea `calib/topdown.py` przeniesione z atrybucją; zakładamy zgodę machinekind.
2. Które zadania po `reach` i `lift`: pick-and-place, nalewanie, sortowanie?
3. Jakie kamery trafią na stanowisko i czy przepuszczanie USB kamer przez
   Shadow wystarcza przepustowością (kilka strumieni MJPEG) — jeśli nie,
   kamery trzeba będzie puścić siecią tak jak ramię.
4. Czy Shadow widzi komputer przy ramieniu (Tailscale?) — od tego zależy
   wybór: przepuszczenie USB czy most.

---

## 6. Sprzęt

| | ramię nr 1 (sierpień) | ramię nr 2 (wrzesień, bieżące) |
|---|---|---|
| przejściówka | CH343 `5AAF219949` | CH343 `5AAF220303` |
| port | COM11 | COM12 |
| limity serw (EEPROM) | były zawężone (bark −2…+84°); **rozszerzone do 0…4095** 2026-08-31, oryginał w [`docs/hardware/servo_limits_arm1.json`](docs/hardware/servo_limits_arm1.json) | wyglądają na prawdziwą kalibrację LeRobota (±100–125°) |
| uwagi | chwytak 1986…2670 tików (z jego EEPROM-u) | stał złożony (`shoulder_lift −101°`, poza limitem −95° z konfiguracji — ramię zostaje, gdzie stoi, i wraca w zakres 15°/s dopiero po włączeniu sprzęgła); `wrist_flex` w serwie max +88° przy +95° w konfiguracji — limity bliźniaka zawężane do EEPROM (`joint_limits()`) |

Przywrócenie oryginalnych limitów ramienia nr 1: dla każdego serwa zapis
`Lock`(55)=0 → `Min_Angle_Limit`(9) / `Max_Angle_Limit`(11) → `Lock`=1; przed
zmianą ustawić cel = bieżącą pozycję, inaczej zmiana limitu odblokuje stary cel
i ramię ruszy.

---

## 7. Pułapki, które cicho psują sim-2-real

Każda jest pilnowana testem albo poprawiona w kodzie:

- **Znak punktu głównego w MuJoCo** — przy odwrotnym kadr przesuwa się o
  dwukrotność przesunięcia (62 px przy 30 px).
- **Fałszywy półpiksel w pionie** — dawna „kompensacja” wynikała z pomiaru
  na środkach czerwonych plam, które podnosi cień dolnej połowy kulki. Mierzyć
  geometrię trzeba na **maskach segmentacji**, nie na kolorze.
- **Półpiksel w detektorze AprilTag** — `CORNER_REFINE_APRILTAG` oddaje rogi
  w konwencji narożników pikseli. Dotyczy też galaxeo.
- **Zmiana masy w skompilowanym modelu bez `mj_setConst`** — stałe kontaktu
  (`body_invweight0`) zostają od starej masy; kostka ściśnięta między szczęką
  a blatem wylatywała z prędkością 10–180 m/s. Na GPU: `mjw.set_const`.
- **Torch z PyPI na Windows nie ma CUDA** — koło z indeksu PyTorcha.
- **`CameraStream` z aplikacji odbija obraz lustrzanie**; bliźniak otwiera
  kamery z `mirror=False`.
- **`MjSpec.attach` gubi opcje fizyki ramienia**, jeśli scena ich nie przejmie.
- **Nadzór dociągał ramię do limitów z konfiguracji** przy połączeniu (skok
  przy `shoulder_lift −101°`) — teraz ramię zostaje, a w zakres wraca powoli
  po włączeniu sprzęgła.
- **Chwytak 0..100 przeliczany inaczej w bliźniaku i w backendzie** — ta sama
  szczęka była w symulacji 1,8× szerzej otwarta; jedno przeliczenie przez tiki.
- **Obserwacja z kinematyki sprzed kroku fizyki** — TCP 7 mm od FK(q); po
  `mj_step` trzeba `mj_kinematics` (na GPU `mjw.kinematics`).
- **`cv2.phaseCorrelate` z oknem (OpenCV 5.0) mnoży wejście przez okno w
  miejscu** — kadr odniesienia strażnika kamer blakł co 2 s; przekazywać kopie.
- **Test w czasie zegarowym na obciążonej maszynie** — `lift` z kamer bywał
  czerwony pod obciążeniem; testy pętli idą teraz w czasie symulowanym
  (`Twin.step`, `PolicyRunner.step_once`).
- **Panel: poza każdego geomu 30×/s zrywała połączenie przeglądarki** (~3000
  komunikatów/s) — most sceny wysyła pozy ciał i tylko przy zmianie, a siatki
  do widoku są uproszczone (348 → 43 tys. trójkątów, 0,3 mm obrysu).
- **Viser w ukrytej karcie przeglądarki nic nie rysuje** — klient przetwarza
  komunikaty w pętli animacji; „pusty panel” w tle to nie błąd serwera.
- **Polityka nauczona na prawdziwej pozycji obiektu nie przeżywa opóźnienia
  percepcji** — patrz wyżej; każdy nowy obiekt z kamer musi mieć swój model
  percepcji w treningu.
- **Dopasowanie modelu „obiekt leży na blacie” do obiektu w powietrzu daje
  pewne siebie bzdury** — bramka na jakość dopasowania (IoU), a obiekt w dłoni
  z kinematyki.
