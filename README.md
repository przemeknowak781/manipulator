# Cyfrowy bliźniak SO-101

Symulacja MuJoCo, która wie o biurku to samo, co prawdziwe stanowisko: gdzie
stoi ramię, gdzie stoją kamery, jaką mają optykę i jak naprawdę odpowiadają
serwa. Polityki uczone w niej na GPU jadą potem na prawdziwym **SO-101**
tą samą ścieżką kodu — przez nadzór bezpieczeństwa i z tą samą percepcją
z kamer.

W repozytorium jest też aplikacja do [sterowania ramieniem ruchem dłoni](#sterowanie-dłonią--lerobot-101--mediapipe)
(MediaPipe), od której projekt się zaczął.

```
  prawdziwe stanowisko                        bliźniak (MuJoCo)
 ┌──────────────────────┐   kalibracja      ┌──────────────────────────┐
 │ SO-101 (COM/socket)  │ ────────────────► │ ramię z MuJoCo Menagerie │
 │ kamery USB dookoła   │  karta w dłoni,   │ kamery w tych samych     │
 │ biurko, kostka       │  ChArUco, sysid   │ pozach, z tym samym K,   │
 └──────────▲───────────┘                   │ serwa jak zmierzone      │
            │                               └────────────┬─────────────┘
            │  polityka przez nadzór                     │  tysiące światów MuJoCo Warp
            │  (te same wzory obserwacji)                ▼  PPO na GPU
            └────────────────────────────────── polityka reach / lift
```

## Jaki problem rozwiązuje

Robot nauczony w zwykłej symulacji zwykle zawodzi na prawdziwym biurku.
Symulacja nie wie, gdzie naprawdę stoją kamery i jaką mają optykę, jak
reagują konkretne serwa ani jak spóźniona i zaszumiona jest percepcja —
polityka działa w sim, a na ramieniu nie. Każda nowa umiejętność to wtedy
godziny prób i błędów na sprzęcie, z ryzykiem uszkodzeń.

## Podejście

Bliźniak jest **zmierzony ze stanowiska**, a nie modelowany na oko:

- **Kamery** trafiają do symulacji w swojej zmierzonej pozie i z własną
  macierzą K — intrynsyki z tablicy ChArUco, położenie z karty z tagami
  w chwytaku, wszystkie kamery naraz.
- **Serwa** — identyfikacja dynamiki z nagrania ruchu (tłumienie, armatura,
  opóźnienie); trening losuje warunki *wokół zmierzonego* ramienia.
- **Percepcja** — trening widzi kostkę tak, jak widzą ją kamery (spóźnioną,
  rzadko odświeżaną, zaszumioną), a nie tak, jak zna ją fizyka.
- **Jedna ścieżka kodu dla symulacji i ramienia** — te same wzory obserwacji,
  ten sam nadzór bezpieczeństwa, ta sama percepcja; przejście sim → ramię to
  zmiana backendu (`sim` → `feetech`).

## Co umożliwia

- **Trening polityk na GPU** (`reach`, `lift`) i uruchamianie ich na
  prawdziwym SO-101 przez nadzór bezpieczeństwa.
- **Chwytanie obiektów widzianych wyłącznie z kamer** — bez znaczników na
  obiekcie; kostka lokalizowana z dokładnością 1–3 mm.
- **Dowolne rozstawianie wielu kamer** — przestawienie kamery wykrywane samo
  (przybliżenie o 1 %, obrót o 1°), szybka relokalizacja jednej kamery.
- **Douczanie gotowych polityk** na zmierzonej dynamice swojego ramienia
  zamiast uczenia od zera.
- **Porównanie Sim-Real na żywo** — kadr prawdziwej kamery z nałożonymi
  krawędziami renderu bliźniaka z tej samej pozy.
- **Bezpieczną pracę z ramieniem** — jeden właściciel ruchu; STOP i Dom
  odbierają ramię każdemu; reakcja na błędy serw, utratę łącza i kolizje;
  ściskający chwytak nie puszcza przy STOP.
- **Pracę zdalną** — ramię przez most sieciowy (`socket://`), panel
  w przeglądarce z przewodnikiem krok po kroku i podpowiedzią przy każdej
  kontrolce.

## Co przyspiesza

Zmierzone na RTX A4500:

| | czas |
|---|---|
| Fizyka na GPU (MuJoCo Warp) | **1,1 mln kroków/s**, tysiące światów naraz |
| Trening `reach` | **ok. 2 min** |
| Trening `lift` od zera / douczanie | ok. 28 min / ok. 16 min |
| Kalibracja wszystkich kamer | jedna fala, ok. 20 póz; w symulacji 0,16 mm / 0,013° błędu |
| Identyfikacja dynamiki serw | ok. 20 s ruchu + kilka sekund dopasowania |
| Od świeżego klona do panelu | ok. 6 min instalacji → `lerobot-twin demo` → `lerobot-twin ui` |

Większość iteracji — projekt zadania, trening, strojenie percepcji,
kalibracja, próba całego łańcucha — dzieje się w bliźniaku; na sprzęcie
zostaje sprawdzenie.

## Stan

W symulacji: polityki `reach-v3` i `lift-v3` mają 100 % sukcesu na CPU
(inny silnik niż w treningu), `lift` z kostką **wyłącznie z kamer** podnosi
8/8. Kod przeszedł trzy rundy przeglądu z niezależną weryfikacją poprawek;
567 testów. **Na prawdziwym ramieniu całość nie była jeszcze testowana** —
to następny krok, procedura krok po kroku w [docs/TWIN.md](docs/TWIN.md).
Decyzje, historia i pułapki: [HANDOFF.md](HANDOFF.md).

---

## Szybki start na nowym komputerze

Repozytorium zawiera wszystko, czego potrzeba do uruchomienia — także modele
i dane:

| Katalog | Co w nim jest | W repozytorium? |
|---|---|---|
| `models/` | modele MediaPipe: dłoń i poza `lite` ([models/README.md](models/README.md)) | tak; brakujący model pobiera się sam |
| `assets/robots/so101/` | model ramienia z MuJoCo Menagerie z siatkami STL | tak |
| `assets/so101_preview.npz` | model podglądu 3D aplikacji `lerobot-mp` | tak |
| `assets/policies/` | polityki bazowe `reach-v3` i `lift-v3` (start do douczania) | tak |
| `examples/twin.sim.json` | przykładowe stanowisko w symulacji: dwie kamery z zaufaną kalibracją | tak |
| `workspace/twin.json` | **Twoje** stanowisko: kamery i ich kalibracja, port, stół, zmierzona dynamika | nie — per biurko, powstaje przy pierwszym zapisie w panelu albo przez `lerobot-twin demo` |
| `workspace/policies/` | polityki, które sam wytrenujesz | nie |
| `configs/local.yaml` | Twoja konfiguracja (kopia `configs/default.yaml`), np. tiki chwytaka | nie (`.gitignore`) |

Przy uruchomieniu z klonu domyślne ścieżki (`models/…`, `workspace/…`) liczą
się od katalogu repozytorium, więc `lerobot-twin ui` odpalone z innego
katalogu trafia w to samo stanowisko. Ścieżki podane w flagach liczą się jak
zwykle od katalogu bieżącego. Poza repozytorium zostają tylko rzeczy tworzone
na danej maszynie: kalibracja LeRobota (tylko backend `lerobot`) i pamięć
podręczna kerneli MuJoCo Warp (ok. 100 s kompilacji przy pierwszym treningu).

**Wymagania:** Python **3.12** (sprawdzony 3.12.10), Git, OpenGL (każda karta
graficzna, także zintegrowana). Trening polityk wymaga karty **NVIDIA z CUDA**
(sterownik z obsługą CUDA 12). Bez niej działa wszystko poza treningiem: panel,
kalibracja, percepcja, identyfikacja dynamiki i ewaluacja polityk na CPU.
Miejsce na dysku: ok. 5 GB na `.venv` z torchem CUDA (sam torch ok. 3,9 GB),
ok. 1 GB bez NVIDIA (torch CPU); przy instalacji z CUDA dodatkowo ok. 3 GB
pamięci podręcznej pip (`pip install --no-cache-dir …` jej nie tworzy).
Sprawdzone wersje pakietów są w [`constraints.txt`](constraints.txt).

### 1. Instalacja

**Windows + karta NVIDIA** (PowerShell):

```powershell
git clone https://github.com/machinekind/digital_twin_training.git
cd digital_twin_training
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass   # tylko to okno; bez tego Activate.ps1 jest zablokowany
.venv\Scripts\activate
python -m pip install --upgrade pip
# torch z CUDA NAJPIERW - torch z PyPI na Windows jest bez CUDA
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[twin,train,feetech,dev]" -c constraints.txt
```

Świeży Windows blokuje skrypty PowerShella („running scripts is disabled on
this system”), stąd `Set-ExecutionPolicy` przed aktywacją — w każdym nowym
oknie przed `.venv\Scripts\activate`, albo raz na stałe:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. W `cmd.exe` aktywacja to
`.venv\Scripts\activate.bat` i nic więcej nie trzeba. Można też nie aktywować
wcale i pisać `.venv\Scripts\python -m pip …`, `.venv\Scripts\lerobot-twin …`.

Karty RTX 50xx: `cu128` zamiast `cu126`. Mało miejsca na `C:`? Przed
instalacją (w tym samym oknie PowerShella) przenieś pliki tymczasowe i pamięć
podręczną pip na inny dysk:

```powershell
$env:TMP='D:\tmp'; $env:PIP_CACHE_DIR='D:\tmp\pipcache'   # cmd.exe: set TMP=D:\tmp & set PIP_CACHE_DIR=D:\tmp\pipcache
```

**Linux + karta NVIDIA** (x86_64, glibc ≥ 2.28, np. Ubuntu 22.04/24.04):

```bash
sudo apt install python3.12 python3.12-venv libgl1 libglib2.0-0 libegl1   # Ubuntu 22.04: python3.12 z PPA deadsnakes
git clone https://github.com/machinekind/digital_twin_training.git && cd digital_twin_training
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[twin,train,feetech,dev]" -c constraints.txt
sudo usermod -aG dialout $USER    # dostęp do portu ramienia (po tym wyloguj się i zaloguj)
```

Torch z PyPI na Linuksie jest z CUDA 13 i wymaga sterownika ≥ 580 — dlatego
też tu indeks `cu126`. Na serwerze bez ekranu: `export MUJOCO_GL=egl`.

**Bez karty NVIDIA** — te same kroki, tylko torch z indeksu `cpu` i bez `train`:

```powershell
# Windows (PowerShell)
git clone https://github.com/machinekind/digital_twin_training.git
cd digital_twin_training
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[twin,feetech,dev]" -c constraints.txt
```

```bash
# Linux (pakiety systemowe jak wyżej) i macOS z Apple Silicon
git clone https://github.com/machinekind/digital_twin_training.git && cd digital_twin_training
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu   # macOS: pip install torch==2.11.0
pip install -e ".[twin,feetech,dev]" -c constraints.txt
```

Linux bez ekranu i bez GPU: dodatkowo `sudo apt install libosmesa6` i
`export MUJOCO_GL=osmesa`. Sama aplikacja sterowania dłonią (bez bliźniaka,
bez torcha): `pip install -e ".[feetech]" -c constraints.txt`.

### 2. Sprawdzenie

```bash
lerobot-twin check
```

Wypisuje wersje (mujoco, viser, torch, cv2…), model SO-101 i porty USB-serial,
a osobno gotowość do treningu: torch z CUDA (`cuda.is_available`, karta), warp
i mujoco_warp — `OK` przy każdej gotowej pozycji, `--` przy niedostępnej.
„Trening na GPU: niedostępny” na komputerze bez NVIDIA to nie błąd.

### 3. Panel bliźniaka z przykładowym stanowiskiem

```bash
lerobot-twin demo     # examples/twin.sim.json -> workspace/twin.json (istniejącego nie nadpisze bez --force)
lerobot-twin ui       # http://localhost:8080
```

Port 8080 zajęty (inny serwer, proxy)? `lerobot-twin ui --port 8765` i adres
`http://localhost:8765`. Na starcie panel wypisuje w konsoli pełną ścieżkę pliku
stanowiska (`istnieje` albo `nowe, puste`) i katalog polityk — tak widać, że
użyty jest `workspace/twin.json` z klonu, także gdy panel odpalono z innego
katalogu. To samo pokazuje `lerobot-twin workspace`.

W panelu: zakładka **Ramię** → `sim` → **Połącz**; zakładka **Polityki** →
`lift-v3`, `lift: skad polozenie kostki` = **kamery**, kostkę kładzie przycisk
`lift (sim): poloz kostke losowo`. Bez `demo` bliźniak startuje z pustym
stanowiskiem (ramię `sim`, bez kamer) — kamery dodaje się w zakładce Kamery
(`Dodaj symulowana przed ramieniem`, potem kalibracja albo
`Symulowana: uznaj prawdziwa poze za kalibracje`).
Karta z panelem musi być widoczna: viser w karcie w tle nic nie rysuje.

### 4. Testy

```bash
pytest -q                     # ok. 3-4 min na RTX A4500 (+ ok. 100 s kompilacji kerneli Warp za 1. razem)
pytest -q -m "not render"     # komputer bez OpenGL
```

Testy GPU pomijają się same bez CUDA albo bez `mujoco_warp`; testy bliźniaka —
bez `mujoco`/`torch`. Jeden stary test (`test_mapping.py::test_direct_and_ik_move_the_tip_the_same_way`)
jest oznaczony jako znany problem (`xfail`).

Czysty wynik bez NVIDIA i bez LeRobota: same `passed` poza **7 skipped**
(4 testy GPU, 3 testy LeRobota) i **1 xfailed**; ok. 2–3 min na CPU. Z kartą
NVIDIA zostają tylko 3 pominięcia LeRobota (albo żadne, gdy jest zainstalowany).

### 5. Trening (tylko NVIDIA)

```bash
lerobot-twin train --task reach --iters 160
lerobot-twin train --init assets/policies/reach-v3/policy.pt --iters 150    # douczanie polityki bazowej
lerobot-twin eval workspace/policies/<nazwa>/policy.pt --rand               # ewaluacja, działa na CPU
```

Wyniki trafiają do `workspace/policies/<nazwa>/`. Na karcie z mniej niż 20 GB
pamięci dodaj `--envs 2048` albo `1024`.

### 6. Prawdziwe SO-101

Do bliźniaka służy backend `feetech` (sam `pyserial`, bez LeRobota i bez jego
kalibracji). Port pokaże `lerobot-twin check` albo „Wykryj porty” w panelu
(Windows `COMx`, Linux `/dev/ttyACM0`). Tiki chwytaka w konfiguracji
(`robot.gripper_closed_ticks` 1986 / `gripper_open_ticks` 2670) to wartości
ramienia autora — sprawdź swoje i wpisz je w kopię konfiguracji:

```bash
cp configs/default.yaml configs/local.yaml      # Windows: copy configs\default.yaml configs\local.yaml
lerobot-twin --config configs/local.yaml ui
```

Dalej: [docs/TWIN.md](docs/TWIN.md), „Pierwszy test na prawdziwym stanowisku”, od kroku 0.

### Naprawa OpenCV po LeRobocie

Opcjonalny backend `lerobot` (`pip install -e ".[robot]" -c constraints.txt`,
tylko Python 3.12) ciągnie `opencv-python-headless`, który podmienia `cv2` na
wersję bez okien — testy przechodzą, ale okno podglądu `lerobot-mp` się nie
otwiera. Naprawa:

```bash
pip uninstall -y opencv-python-headless opencv-python
pip install --force-reinstall --no-deps opencv-contrib-python==5.0.0.93
python -c "import cv2; print([l.strip() for l in cv2.getBuildInformation().splitlines() if 'GUI' in l])"
```

Ostatnie polecenie ma pokazać `WIN32UI` (Windows) albo `GTK`/`QT` (Linux), a nie `NONE`.

---

---

# Sterowanie dłonią — LeRobot 101 × MediaPipe

Sterowanie ramieniem **SO-101 (LeRobot 101)** ruchem dłoni przed kamerą.
MediaPipe śledzi 21 punktów dłoni, a aplikacja zamienia je na zadane pozycje
sześciu stawów — z filtracją, limitami i zatrzymaniem awaryjnym.

Działa **bez robota**: wbudowany symulator i podgląd 3D pozwalają nauczyć się
gestów, zanim cokolwiek podłączysz.

```
   dłoń przed kamerą              podgląd na żywo                  ramię
  ┌────────────────┐        ┌──────────────────────┐        ┌──────────────┐
  │  21 punktów    │──────► │  HUD: stan, stawy,   │──────► │  SO-101      │
  │  MediaPipe     │        │  limity, chwytak     │        │  lub symulator│
  └────────────────┘        │  + model 3D ramienia │        └──────────────┘
                            └──────────────────────┘
```

## Szybki start

### Windows

Kliknij dwukrotnie **`start.bat`**. Przy pierwszym uruchomieniu utworzy
środowisko (Python 3.12, jeśli jest) i zainstaluje zależności w wersjach
z `constraints.txt`, potem pokaże menu: symulator, prawdziwe ramię, tryb IK,
demo z pliku wideo, diagnostyka, panel bliźniaka.

### Linux / macOS

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[feetech]" -c constraints.txt
lerobot-mp                      # symulator + kamera 0
```

Modele MediaPipe są w repozytorium (`models/`); gdyby ich zabrakło, pobiorą
się same przy pierwszym starcie.

### Prawdziwe ramię

```bash
pip install -e ".[feetech]" -c constraints.txt
lerobot-mp --port /dev/ttyACM0        # Linux
lerobot-mp --port COM5                # Windows
```

Bez zainstalowanego LeRobota `--port` używa backendu `feetech` — rozmowy wprost
z serwami, bez kalibracji LeRobota. Z LeRobotem (`.[robot]`) domyślny jest
backend `lerobot`: ramię musi być wtedy skalibrowane jego narzędziami, a
`--robot-id` wskazuje plik kalibracji. Backend wybiera się jawnie flagą
`--robot feetech|lerobot`.

> **Zanim podłączysz robota:** zrób wokół niego miejsce. Sterowanie rusza
> dopiero po naciśnięciu **SPACJI**, **ESC** to zatrzymanie awaryjne, a **X**
> zamyka aplikację (i odsyła ramię do pozycji domowej).

### Kiedy system nie widzi ramienia

Ramię nie jest urządzeniem USB samo w sobie — zgłasza się jako **przejściówka
USB-serial** na płytce sterownika (najczęściej CH340, rzadziej CP210x lub FTDI)
i dopiero ona tworzy port COM. Aplikacja nie ma jak go „poszukać": jeśli portu
nie ma w systemie, nie ma czego otworzyć.

Sprawdzenie, co widzi system:

```bash
python -c "import serial.tools.list_ports as l; print([p.device for p in l.comports()] or 'BRAK')"
```

Pusta lista oznacza jedną z trzech rzeczy, w tej kolejności prawdopodobieństwa:

1. **Kabel** — sporo kabli USB ma tylko żyły zasilania. Ramię się zaświeci,
   a portu nie będzie. Zamień kabel na taki, o którym wiesz, że przenosi dane.
2. **Sterownik przejściówki** — układ dotarł do systemu, ale bez sterownika
   siedzi w Menedżerze urządzeń jako nieznane urządzenie zamiast portu COM.
   W Menedżerze urządzeń szukaj pozycji z żółtym wykrzyknikiem, sprawdź jej
   `VID`/`PID` i doinstaluj sterownik: `1A86:7523` to CH340, `10C4:EA60` to
   CP210x, `0403:6001` to FTDI.
3. **Maszyna wirtualna albo pulpit zdalny** — urządzenie wpięte do fizycznego
   komputera **nie trafia** do systemu goszczonego samo z siebie. Trzeba je
   jawnie przepuścić:

| Środowisko | Gdzie to włączyć |
|---|---|
| Shadow / Blade | w kliencie Shadow: menu urządzeń USB → zaznacz przejściówkę ramienia |
| VirtualBox | Urządzenia → USB → wybierz urządzenie (plus Extension Pack) |
| VMware | VM → Removable Devices → urządzenie → Connect |
| Hyper-V | brak przekierowania USB — użyj trybu rozszerzonej sesji albo `usbipd-win` |
| WSL2 | `usbipd bind --busid <id>` i `usbipd attach --wsl --busid <id>` |
| Pulpit zdalny (RDP) | w kliencie: Zasoby lokalne → Więcej → zaznacz urządzenie |

Ta sama zasada dotyczy kamery. Jeśli w systemie goszczonym kamera jest
widoczna, ale nie oddaje żadnej klatki (OpenCV otwiera ją i zaraz dostaje
`ERROR_OPERATION_ABORTED`), to zwykle nie jest wina sterownika — to samo
przekierowanie USB, które trzeba włączyć.

---

## Ramię na jednej maszynie, aplikacja na drugiej

Przekierowanie całego USB przez sieć działa dobrze na LAN-ie i na lokalnym
hypervisorze, a źle przez internet — i nie dlatego, że „jest wolniej".
Protokół Feetech to pytanie–odpowiedź: aplikacja wysyła ramkę i **czeka** na
odpowiedź serwa, zanim wyśle następną. Lokalnie jedna taka transakcja kosztuje
**0,29 ms**. Przy przekierowaniu USB kosztuje tyle, ile wynosi RTT łącza —
razy liczba przebiegów URB, których sterownik potrzebuje na jedną ramkę.
`feetech-servo-sdk` daje na odpowiedź około **34 ms** przy 1 Mbaud, więc to
wyścig, którego nie warto zaczynać na łączu o RTT rzędu 20 ms.

Most przenosi przez sieć **sam strumień bajtów**, a nie transakcje USB, więc
jedna ramka to jeden przebieg zamiast kilku. Na maszynie z ramieniem:

```bash
pip install -e ".[bridge]"
lerobot-mp-bridge --port COM11 --listen 0.0.0.0:5555 --allow <IP-maszyny-zdalnej>
```

Na maszynie zdalnej wystarczy dowolny sterownik wirtualnego portu szeregowego
po TCP — **HW VSP3** albo **com0com + com2tcp** na Windowsie, `socat` na
Linuksie. Powstaje tam zwykły `COM3`, a aplikacja nie wie, że port jest gdzie
indziej:

```bash
lerobot-mp --port COM3
```

Zmierzone: ping wszystkich sześciu serw przez most po pętli zwrotnej to
**0,40 ms** wobec 0,29 ms bezpośrednio — sam most kosztuje więc **0,11 ms**,
a cała reszta budżetu to RTT łącza. `--stats 5` pokazuje ruch na żywo, kiedy
trzeba sprawdzić, czy coś w ogóle płynie.

Dwie rzeczy warte zapamiętania:

* `--allow` nie jest ozdobnikiem. Otwarty port TCP po drugiej stronie rusza
  fizycznym ramieniem — bez listy adresów most ostrzega w logu i słucha
  wszystkich.
* Rozłączenie sieci **nie upuszcza ramienia**: serwa trzymają ostatnią zadaną
  pozycję. Most zamyka wtedy port, żeby kolejny klient nie zastał go zajętego.

Zostaje kamera — musi być na tej maszynie, na której liczy się wizja. Sieć
przenosi tu port szeregowy, nie obraz.

---

## Jak się tym steruje

| Ruch dłoni | Co robi ramię |
|---|---|
| w lewo / w prawo | obrót podstawy `shoulder_pan` |
| w górę / w dół | podnoszenie ramienia `shoulder_lift` |
| bliżej / dalej od kamery | wysuwanie przedramienia `elbow_flex` |
| pochylenie dłoni | pochylenie nadgarstka `wrist_flex` |
| obrót dłoni | obrót nadgarstka `wrist_roll` |
| szczypnięcie kciuk–wskazujący | chwytak `gripper` |
| **zwinięcie trzech ostatnich palców** | **pauza** — ramię stoi, dłoń można przełożyć |

To jest tryb `direct`. Są jeszcze trzy — `ik`, `arm` i `keys` — opisane niżej.
Ostatni z nich nie potrzebuje ani dłoni, ani kamery.

Gest pauzy działa jak podniesienie myszy z podkładki: zwijasz środkowy,
serdeczny i mały palec, przenosisz rękę w wygodne miejsce, prostujesz palce —
i jedziesz dalej. Robot nigdy nie przeskakuje, bo ruch liczy się od pozycji
z chwili załączenia, a nie od bezwzględnego położenia dłoni w kadrze.

### Klawisze

| | | | |
|---|---|---|---|
| `SPACJA` | włącz/wyłącz sterowanie | `H` | powrót do pozycji domowej |
| `ESC` | stop awaryjny i jego kasowanie | `C` | nowe zaczepienie dłoni |
| `O` / `P` | kalibracja chwytaka: otwarty / zamknięty | `M` | tryb mapowania: `direct` → `ik` → `arm` |
| `J` `L` `I` `K` | obrót kamery podglądu | `,` `.` | przybliżenie |
| `-` / `=` | limit prędkości | `V` | podgląd ramienia wł./wył. |
| `X` | wyjście | | |

W trybie `keys` dochodzą klawisze jazdy: `W` / `S` prowadzą chwytak w górę i w
dół, `A` / `D` w bok, `Q` / `E` cofają i wysuwają, strzałki w bok obracają
nadgarstek, a strzałki góra/dół rozwierają i zaciskają szczękę.

Dlatego właśnie **wyjście siedzi na `X`, a stop awaryjny na `ESC`** — `Q` jest
zajęte przez jazdę. Stop awaryjny celowo *nie* trafił na klawisz sąsiadujący z
wyjściem: pomyłka przy panice miałaby zatrzymać ramię, a nie zamknąć aplikację,
bo zamknięcie odsyła ramię do pozycji domowej, czyli nim rusza.

**Kalibracja chwytaka pod własną dłoń** zajmuje dwie sekundy: rozstaw palce
i naciśnij `O`, złącz je i naciśnij `P`.

---

## Cztery tryby mapowania

Przełączasz je klawiszem `M` albo flagą `--mode`.

**`direct`** (domyślny) — każda oś dłoni steruje jednym stawem. Nie wymaga
znajomości wymiarów ramienia, więc działa poprawnie przy dowolnej kalibracji.
Zacznij od niego.

**`ik`** — pozycja dłoni wyznacza punkt w przestrzeni, a kąty stawów liczy
odwrotna kinematyka. Ruch jest bardziej „kartezjański": dłoń w bok przesuwa
chwytak w bok, a nie obraca całe ramię wokół podstawy.

**`arm`** — śledzi **całe Twoje ramię**, nie samą dłoń. MediaPipe Pose podaje
bark, łokieć i nadgarstek; aplikacja liczy z nich trzy kąty i przekłada je
wprost na trzy pierwsze stawy robota:

| Twój ruch | Staw robota |
|---|---|
| unosisz rękę | `shoulder_lift` |
| przenosisz ją w bok / do przodu | `shoulder_pan` |
| **zginasz łokieć** | **`elbow_flex`** |
| obracasz dłoń | `wrist_roll` |
| szczypiesz palcami | `gripper` |

Domyślne przełożenie to **1:1** — zginasz łokieć o 60°, robot zgina o 60°.

```bash
lerobot-mp --mode arm
lerobot-mp --mode arm --arm-side Right     # wymuś konkretną rękę
lerobot-mp --mode arm --no-arm-hand        # bez śledzenia dłoni (szybciej)
```

Kąty liczą się w **układzie Twojego tułowia**, a nie kamery: obrócenie się
bokiem albo przechylenie na krześle nie zmienia zadanej pozy robota, bo
geometrycznie ramię wobec tułowia się nie zmieniło.

W tym trybie dłoń jest **dodatkiem**, a nie warunkiem — kiedy wypadnie z kadru,
nadgarstek i chwytak po prostu stoją, a bark z łokciem jadą dalej. Zniknięcie
całego ramienia zatrzymuje wszystko, tak jak zniknięcie dłoni w pozostałych
trybach.

Dwie rzeczy zmierzone, nie założone:

* po ustabilizowaniu śledzenia ten sam obraz daje **0,4° rozrzutu na łokciu**
  (model `lite`, 17 ms/klatkę) albo **0,2°** (`pose_landmarker_full`, 22 ms) —
  znacznie poniżej tego, co ma znaczenie przy prowadzeniu ręką;
* MediaPipe jest trenowane na ludziach mniej więcej pionowych. Przechylenie do
  ±10° kosztuje kilka stopni błędu, ale przy ±25° estymata psuje się mocno
  (kilkadziesiąt stopni). Trzymaj się z grubsza prosto.

Oba modele śledzenia naraz to około 30 ms na klatkę, czyli ~20 FPS wizji.
Pętla sterowania i tak tyka niezależnie z 30 Hz, więc ramię jedzie płynnie —
po to jest to rozdzielenie.

**`keys`** — sterowanie z klawiatury, **bez kamery i bez dłoni**. Klawisze
prowadzą końcówkę chwytaka po przestrzeni, a kąty stawów liczy ta sama odwrotna
kinematyka co w trybie `ik`:

| Klawisze | Co robi ramię |
|---|---|
| `W` / `S` | góra i dół |
| `A` / `D` | ruch w bok |
| `Q` / `E` | cofnięcie i wysunięcie końcówki |
| `←` / `→` | obrót nadgarstka `wrist_roll` |
| `↑` / `↓` | rozwarcie i zaciśnięcie chwytaka |

```bash
lerobot-mp --mode keys
```

Osie są zaczepione w **bieżącym kierunku ramienia**, a nie w układzie świata:
`W` zawsze wysuwa chwytak dalej od podstawy, niezależnie od tego, gdzie ramię
akurat patrzy. Przy osiach światowych to samo `W` raz by wysuwało, a raz
prowadziło bokiem — zależnie od obrotu podstawy.

Wciśnięcie nie robi kroku, tylko **nadaje osi prędkość na 0,18 s**
(`keyboard.hold_timeout`). Klawiatura nie wysyła zdarzenia „puszczono" —
trzymany klawisz to seria powtórzeń z autopowtarzania systemu. Gdyby jedno
zdarzenie znaczyło jeden krok, ruch szarpałby na starcie, bo pierwsze
powtórzenie przychodzi dopiero po ~0,5 s. Tak wychodzi jazda ciągła przy
trzymaniu i natychmiastowy stop po puszczeniu.

Trzymanie klawisza poza zasięgiem **nie nakręca** zapamiętanego punktu: po
przycięciu przez IK cel wraca na osiągalny, więc powrót nie trwa tyle, ile
trwało wyjście.

Ten tryb działa też jako awaryjne wyjście, kiedy kamery po prostu nie ma —
`M` nie wyprowadzi z niego aplikacji uruchomionej bez obrazu, bo pozostałe
tryby nie miałyby z czego liczyć ruchu.

### Czym steruje się szczęką chwytaka

`mapping.gripper_source` (albo `--gripper`):

* `pinch` (domyślnie) — odległość kciuk–wskazujący, znormalizowana rozmiarem
  dłoni, więc niezależna od odległości od kamery. Działa we wszystkich trzech
  trybach, także w `arm`.
* `none` — chwytak nie rusza się sam.

Kciuk i wskazujący są celowo **wyłączone** z gestu pauzy, żeby szczypanie nie
było mylone z zaciskaniem pięści.

---

## Podgląd 3D — prawdziwe złożenie, nie ilustracja

Panel obok obrazu z kamery pokazuje **rzeczywistą geometrię SO-101**: bryły
producenta (STEP z `TheRobotStudio/SO-ARM100`) złożone według jego URDF-a,
zaimportowane z repozytorium [Articulus](https://github.com/przemeknowak781/articulus).
Model MJCF ramienia — MuJoCo Menagerie (Apache-2.0), modele MediaPipe — Google
(Apache-2.0). Pełna lista plików osób trzecich: [NOTICE](NOTICE).

Kinematyka podglądu nie jest przepisana drugi raz — łańcuch jest wczytywany
z eksportu Articulusa jako ciąg kroków `pre @ ruch @ post`. Test
`test_kinematics_matches_the_articulus_reference` porównuje wynik z
transformacjami referencyjnymi zapisanymi przy eksporcie: **zgodność poniżej
mikrometra dla każdego członu w każdej nazwanej pozie**.

Model (~160 kB) jest w repozytorium. Odtworzenie go od zera:

```bash
pip install "build123d>=0.11,<0.12"
git clone https://github.com/przemeknowak781/articulus ../articulus
python scripts/import_articulus_model.py --articulus ../articulus
```

Bez tego pliku aplikacja działa dalej — pokazuje uproszczony rysunek
schematyczny zamiast modelu.

---

## Skąd wzięły się liczby w konfiguracji

Stałe geometryczne **nie są oszacowane**. Wyprowadza je
`scripts/derive_geometry.py` z tego samego modelu 3D — długości ogniw,
wysokość barku, położenie osi obrotu podstawy, przesunięcia i znaki wszystkich
stawów — i od razu mierzy, jak bardzo uproszczony model płaski rozjeżdża się
z pełną kinematyką:

```
$ python scripts/derive_geometry.py --check
  sprawdzono 625 poz; blad mediana 1.06 mm, najgorszy 1.07 mm
  domkniecie FK->IK->FK: najgorszy blad 0.0000 mm
```

Milimetr błędu przy zasięgu 45 cm — znacznie poniżej precyzji sterowania ręką.
Kilka rzeczy, które ten pomiar wykrył i które inaczej zostałyby błędem:

* oś obrotu podstawy **nie leży w początku układu** — pominięcie tego dawało
  30 mm błędu przy obrocie o 45°,
* wszystkie cztery stawy ramienia mają **ujemny** zwrot względem kąta ogniwa,
* gałąź rozwiązania IK „łokieć w górę" wypada poza zakresy stawów SO-101 —
  domyślna jest druga.

---

## Bezpieczeństwo

Nic nie trafia do serw z pominięciem nadzoru (`control/safety.py`):

* **limity pozycji** każdego stawu, ciaśniejsze niż zakres z kalibracji,
* **limit prędkości** zadanej — gwałtowny ruch ręki nie daje szarpnięcia,
* **start od zmierzonej pozycji** i płynne dojście do pozycji domowej,
  zamiast skoku przy pierwszym rozkazie,
* **watchdog dłoni** — zniknięcie ręki zamraża ruch po 0,4 s, a po 6 s
  odsyła ramię do pozycji domowej,
* **stop awaryjny** klawiszem `ESC` (drugie `ESC` go kasuje; `X` zamyka aplikację),
* **`max_relative_target`** przekazywany do LeRobot jako dodatkowy limit
  sprzętowy skoku.

Sterowanie startuje **wyłączone** — robot nie ruszy, dopóki świadomie nie
naciśniesz SPACJI.

---

## Konfiguracja

Wszystkie parametry są w `configs/default.yaml`, opisane komentarzami:

```bash
lerobot-mp --config configs/default.yaml
```

Plik można skracać — brakujące klucze biorą wartości domyślne, a nieznany
klucz kończy się czytelnym błędem zamiast cichego zignorowania.

Najczęściej strojone rzeczy:

| Objaw | Co zmienić |
|---|---|
| staw jedzie w złą stronę | `joints.<nazwa>.invert: true` |
| ruch zbyt czuły / zbyt leniwy | `joints.<nazwa>.gain` |
| drżenie obrazu przenosi się na ramię | `filters.position.min_cutoff` w dół |
| wyczuwalne opóźnienie | `filters.*.beta` w górę |
| ramię rusza się za szybko | `safety.velocity_scale: 0.5` |
| pauza włącza się przypadkiem | `clutch.curl_threshold` w dół |
| tryb `arm` gubi rękę | `arm.min_visibility` w dół |
| tryb `arm` reaguje za mocno | `arm.pan_gain` / `lift_gain` / `elbow_gain` |

Uwaga na jednostki `gain`: dla `shoulder_pan`, `shoulder_lift` i `elbow_flex`
to **stopnie na jednostkę znormalizowaną** (ruch dłoni przez pół kadru ≈ 0,5),
a dla `wrist_flex` i `wrist_roll` **bezwymiarowe przełożenie** (1,0 = ruch 1:1).

---

## Bez kamery i bez ekranu

```bash
lerobot-mp --source nagranie.mp4 --no-view --clutch always --record wynik.mp4
```

Odtwarza plik wideo (własne nagranie dłoni — w repozytorium nie ma przykładowego) zamiast kamery i zapisuje cały podgląd — z HUD-em i modelem
3D — do pliku. Przydaje się do demonstracji i do zgłaszania błędów.

---

## Jak to jest zbudowane

```
src/lerobot_mp/
├── config.py              parametry + wczytywanie YAML
├── app.py                 pętla główna
├── cli.py                 wiersz poleceń
├── vision/                kamera, MediaPipe (dłoń + sylwetka), cechy sterujące
├── control/               filtry, kinematyka, mapowanie, nadzór
├── robot/                 symulator i adapter LeRobot
├── preview/               model 3D, renderer programowy i jego wątek
└── ui/                    HUD i podgląd schematyczny
```

Wizja i sterowanie chodzą w **różnym tempie**: pętla sterowania tyka ze stałą
częstotliwością niezależnie od tego, czy kamera zdążyła z nową klatką. Dzięki
temu limity prędkości i watchdog działają tak samo, gdy detekcja chwilowo
zwolni — ramię dojeżdża, zamiast szarpać.

Podgląd 3D rysuje **osobny wątek**. Złożenie ~22 tys. trójkątów kosztuje ok.
30 ms, więc robiony wprost w pętli sterowania zabierał jej ten czas co drugą
iterację — pętla zadana na 30 Hz schodziła do 20 Hz, a model i tak przeskakiwał.
Teraz pętla zostawia rendererowi najnowszą pozę i zabiera ostatnią gotową
klatkę, nigdy na niego nie czekając. Poz pośrednich nie odrabiamy, więc
zaległości nie narastają: przy wolnym rysowaniu podgląd gubi klatki, a ramię
i tak jedzie równo.

MediaPipe ma dwa niekompatybilne API — nowe `tasks` (1.x) i stare `solutions`
(0.10.x). Aplikacja wykrywa dostępne automatycznie i w obu przypadkach zwraca
ten sam format, więc działa na obu wersjach.

### Testy

```bash
pip install -e ".[dev]" -c constraints.txt && pytest -q
```

567 testów pokrywa matematykę sterowania (filtry, kinematyka, mapowanie,
nadzór), kąty ramienia w układzie tułowia, wątek podglądu 3D i zgodność modelu
ze źródłem, cały łańcuch od cech dłoni do symulowanego ramienia oraz bliźniaka
(kalibracja, percepcja, RL, panel). Bez extra `[twin]` testy bliźniaka się
pomijają; bez OpenGL: `pytest -q -m "not render"`.

Kilka z nich sprawdza **skutek fizyczny, a nie wartość stawu** — i to nie jest
formalność: w kalibracji SO-101 rosnący `shoulder_lift` *opuszcza* ramię, więc
test „wartość rośnie" przepuszczał odwrócony kierunek. Test liczący wysokość
końcówki go nie przepuszcza.

---

## Wymagania

* Python 3.12 (sama aplikacja `lerobot-mp` działa od 3.10, ale sprawdzany
  i wymagany przez LeRobota jest 3.12)
* kamera internetowa
* dla prawdziwego ramienia: SO-101 na USB i `.[feetech]` (albo `.[robot]`
  z LeRobotem i jego kalibracją)
* bliźniak (`.[twin]`): OpenGL do renderu kamer; trening (`.[train]`):
  karta NVIDIA z CUDA — szczegóły w „Szybki start na nowym komputerze”

## Licencja

Apache-2.0. Geometria SO-101 pochodzi z
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
(Apache-2.0) przez repozytorium
[Articulus](https://github.com/przemeknowak781/articulus).
