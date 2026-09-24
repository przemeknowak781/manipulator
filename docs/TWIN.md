# Cyfrowy bliźniak stanowiska SO-101

Symulacja MuJoCo, która wie o biurku to samo, co prawdziwe stanowisko: gdzie
stoi ramię, gdzie stoją kamery, jaką mają ogniskową i jak naprawdę odpowiadają
serwa. Kamera skalibrowana na biurku trafia do symulacji **dokładnie w swojej
pozie i ze swoją macierzą K**, polityka uczy się na tysiącach światów na GPU
wokół **zmierzonej** dynamiki ramienia, a potem jedzie na prawdziwym ramieniu
tą samą ścieżką — przez nadzór bezpieczeństwa — i widzi kostkę tą samą
percepcją, którą widziała w symulacji.

```
  prawdziwe stanowisko                        bliźniak (MuJoCo)
 ┌──────────────────────┐   kalibracja      ┌──────────────────────────┐
 │ SO-101 (COM/socket)  │ ────────────────► │ ramię z MuJoCo Menagerie │
 │ kamery USB dookoła   │  karta w dłoni,   │ kamery w tych samych     │
 │ biurko, kostka       │  ChArUco, sysid   │ pozach, z tym samym K,   │
 └──────────▲───────────┘                   │ serwa jak zmierzone      │
            │                               └────────────┬─────────────┘
            │  polityka przez nadzór                     │  4096 światów MuJoCo Warp
            │  (te same wzory obserwacji)                ▼  PPO na GPU (A4500)
            └────────────────────────────────── polityka reach / lift
```

## Stan (2026-09-24)

| | |
|---|---|
| **Gotowe i przetestowane w symulacji** | model, kinematyka, scena z kamerami o pełnym K, kolizje, **kalibracja wielu kamer z karty w chwytaku**, **intrynsyki z ChArUco**, mapa stołu na żywo, **wykrywanie kostki z kamer**, **wykrywanie przestawionej kamery**, **środowiska RL CPU i GPU**, **PPO na GPU**, **identyfikacja dynamiki serw**, **runner polityk przez nadzór**, **panel webowy** |
| **Do sprawdzenia na sprzęcie** | wszystko, co dotyka prawdziwego ramienia i prawdziwych kamer — procedura krok po kroku niżej |
| **Rozpoznanie** | rekonstrukcja 3D przedmiotów z biurka — [RESEARCH_3D.md](RESEARCH_3D.md) |

Szczegóły, decyzje i pułapki: **[HANDOFF.md](../HANDOFF.md)**.

---

## Instalacja

Skrót dla każdej platformy jest w README, „Szybki start na nowym komputerze”.
Najważniejsze zasady:

- **Torch z CUDA instaluj NAJPIERW, z indeksu PyTorcha**, a dopiero potem
  projekt. Torch z PyPI na Windows jest bez CUDA (`torch.version.cuda == None`),
  a na Linuksie wymaga CUDA 13 (sterownik ≥ 580). Gdy torch z PyPI trafi do
  środowiska pierwszy, `pip install torch --index-url …` odpowie „already
  satisfied” i CPU-owy torch zostanie — wtedy `pip uninstall torch` i od nowa.
- Wersje sprawdzone razem są w `constraints.txt` (`-c constraints.txt`);
  `mujoco` i `mujoco-warp` muszą mieć tę samą wersję (extra je wiążą).
- Trening potrzebuje karty NVIDIA z CUDA. Bez niej instaluj bez `train`
  (torch z indeksu `cpu`): panel, kalibracja, percepcja, identyfikacja
  dynamiki i ewaluacja polityk na CPU działają.

```bash
# Windows + NVIDIA (PowerShell); Linux: python3.12 i source .venv/bin/activate
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126   # bez NVIDIA: .../whl/cpu
pip install -e ".[twin,train,feetech,dev]" -c constraints.txt                 # bez NVIDIA: ".[twin,feetech,dev]"
lerobot-twin check
lerobot-twin demo    # przykładowe stanowisko w symulacji -> workspace/twin.json
```

`lerobot-twin check` wypisuje osobno gotowość do treningu: wersję torcha, jego
kompilację CUDA, `cuda.is_available()` i kartę, oraz `warp` (czy widzi GPU)
i `mujoco_warp`.

- Linux bez ekranu: `export MUJOCO_GL=egl` (NVIDIA) albo `osmesa` (CPU,
  `sudo apt install libosmesa6`). Port szeregowy: `sudo usermod -aG dialout $USER`.
- Po instalacji LeRobota (`.[robot]`) sprawdź, czy `cv2` ma GUI — LeRobot
  podmienia OpenCV na wersję bez okien; naprawa w README („Naprawa OpenCV po
  LeRobocie”). Do bliźniaka LeRobot nie jest potrzebny (backend `feetech`).
- Pierwsza kompilacja kerneli MuJoCo Warp trwa ~100 s; potem są w cache
  (Windows `%LOCALAPPDATA%\NVIDIA\warp\Cache`, Linux `~/.cache/warp`).

**Pliki stanowiska.** `workspace/twin.json` (kamery, kalibracja, port, stół,
dynamika) i `workspace/policies/` są lokalne, poza repozytorium. Z klonu ich
domyślne ścieżki liczą się od katalogu repozytorium, niezależnie od katalogu
uruchomienia; inny plik: `--workspace <plik>`. Bez pliku bliźniak startuje
z pustym stanowiskiem (ramię `sim`, bez kamer). `lerobot-twin demo` kopiuje
`examples/twin.sim.json` — dwie kamery symulowane, których kalibracja to ich
prawdziwa poza w scenie (zaufane), więc „kostka z kamer” działa od razu.
Istniejącego stanowiska nie nadpisze (`--force`, albo `--path <plik>`).

**Konfiguracja ramienia.** Bliźniak czyta plik konfiguracji tylko wtedy, gdy
się go wskaże: `lerobot-twin --config configs/local.yaml ui` (także `train`,
`eval`; opcja ustawia zmienną `LEROBOT_MP_CONFIG`, którą dziedziczy proces
treningu uruchamiany z panelu). Klucze sekcji `robot:` ważne dla bliźniaka
i backendu `feetech` (opisane w `configs/default.yaml`): `center_ticks` (tik
zera stawu, 2048), `gripper_closed_ticks` / `gripper_open_ticks` (szczęka 0
i 100 w skali aplikacji; domyślne 1986 / 2670 to ramię autora), `baudrate`
(1 000 000).

### Na Shadow (maszyna autora)

Shadow to Windows x86-64 z **RTX A4500 (20 GB)**, sterownik 565.90 (CUDA 12.7)
— `cu126` na nim działa. WSL2 nie ruszy (wirtualka nie udostępnia
zagnieżdżonej wirtualizacji), ale nie jest potrzebne: MuJoCo Warp i PyTorch
z CUDA działają natywnie. Na `C:` zostaje niewiele miejsca — duże pakiety
instaluj z `TMP` i `PIP_CACHE_DIR` na `D:` (`set TMP=D:\tmp`) albo
z `--no-cache-dir`. Czasy w tym dokumencie bez innego opisu są zmierzone
na Shadow.

## Panel

```bash
lerobot-twin ui                  # http://localhost:8080
lerobot-twin ui --host 0.0.0.0   # także z sieci — panel nie ma hasła, tylko w zaufanej sieci
```

Stały **STOP** i pasek stanu nad zakładkami. Każdy ruch — z suwaków, z uchwytu
w 3D, z polityki, z fali kalibracyjnej, z identyfikacji — idzie przez
`runtime.Twin` i jego `SafetySupervisor`.

**Ramię ma zawsze jednego właściciela**: panel (sprzęgło i uchwyt), polityka,
fala kalibracyjna albo identyfikacja. Drugi nie ruszy, dopóki pierwszy nie
skończy — panel mówi, kto ma ramię. **STOP**, **Dom**, **Połącz** i **Rozłącz**
odbierają ramię każdemu, a ramię zostaje w **zmierzonej** pozie: nie dociska
do przeszkody, nie wraca skokiem do starego celu i nie cofa się (przy szybkim
ruchu staje między zmierzoną pozą a ostatnim rozkazem). **Chwytak, który
ściska, dalej ściska** — STOP ani Dom nie upuszczają trzymanej kostki; otwiera
się go ręcznie suwakiem. Każde uruchomienie na prawdziwym ramieniu wymaga
świeżego potwierdzenia.

Pętla sama zatrzymuje ramię (STOP awaryjny, powód w pasku stanu), gdy:
- serwo ramienia zgłosi błąd (przeciążenie, przegrzanie, napięcie, czujnik kąta);
- staw ramienia jest ponad 25° od rozkazu dłużej niż 0,5 s — serwo, które
  zwiotczało bez bitu błędu (każdy właściciel ruchu, nie tylko polityka);
  jeśli prawdziwe serwa pod obciążeniem nie nadążą, podnieś
  `Twin.track_err_deg` / `track_err_s`;
- serwa nie odpowiadają przez 5 odczytów z rzędu, albo pętla stanie na > 0,5 s.

Krótsza przerwa łącza nie jest błędem: nadzór stoi, nic nie idzie do serw, a
po pierwszym świeżym odczycie rusza od zmierzonej pozy (po 0,9 s przerwy
największy krok celu 4,2° — wcześniej 16,7°). **Przeciążenie samego chwytaka**
(mocny chwyt) nie zatrzymuje ramienia: docisk jest zmniejszany do 8 jednostek
(~5°) ciaśniej niż zmierzona szczęka, w pasku stanu pojawia się ostrzeżenie;
STOP dopiero, gdy błąd chwytaka trwa 2 s.

Połączenie potrzebuje dwóch zgodnych odczytów (±2 jednostki) — przekłamana
ramka nie zostaje pierwszym rozkazem. Ramię stojące po włączeniu poza limitami
zostaje, gdzie jest; po włączeniu sprzęgła wraca w zakres (najwyżej 90°/s).
Limity z konfiguracji są zawężane do limitów zapisanych w EEPROM serw.
Niefatalne zastrzeżenia (inne tiki chwytaka w backendzie, zero stawów w
kalibracji LeRobota, zmniejszony docisk) panel pokazuje jako **Uwaga** w pasku
stanu.

| zakładka | co robi |
|---|---|
| **Ramię** | połączenie: `sim` (bliźniak jest ramieniem), `feetech` (port `COMx` / `/dev/ttyACM0` albo `socket://adres:5555` przez most), `lerobot`; wykrywanie przejściówki CH343; sprzęgło; suwaki stawów; uchwyt końcówki w 3D (IK) |
| **Kamery** | kamery USB i **symulowane** — symulowaną stawiasz w dowolnym miejscu widoku 3D i przeciągasz uchwytem; piramidy z obrazem na żywo (zielona = zaufana, pomarańczowa = niezaufana, czerwona = **przestawiona**); podgląd; błąd kalibracji względem prawdy dla symulowanych |
| **Kalibracja** | arkusze do druku (tablica ChArUco, karta z tagami); **1.** intrynsyki — tablica w ręku, kadry zapisują się same, gdy wnoszą nowe ujęcie; **2.** położenie wszystkich kamer naraz — fala z kartą w chwytaku; **szybka relokalizacja** jednej przestawionej kamery |
| **Mapa** | kadry wszystkich zaufanych kamer zszyte w mapę blatu (także na stole w 3D), wykrywanie kostki (kolor do wyboru) |
| **Trening** | PPO na GPU w osobnym procesie, z wykresem sukcesu; **identyfikacja dynamiki serw** na połączonym ramieniu i zapis jako środka randomizacji |
| **Polityki** | lista z wynikami (GPU, CPU z randomizacją i bez), ewaluacja na CPU, **uruchomienie na bliźniaku** — sim albo prawdziwe ramię; `reach`: cel przeciągany w 3D; `lift`: kostka z symulacji albo **z kamer** |
| **Sim-Real** | kadr prawdziwej kamery z nałożonymi krawędziami renderu bliźniaka z tej samej pozy i K, mediana rozjazdu w pikselach |

---

## Pierwszy test na prawdziwym stanowisku — krok po kroku

Wszystko poniżej jest przećwiczone na bliźniaku (`sim`) — ta sama ścieżka,
te same przyciski. Przed każdym krokiem na sprzęcie: ramię na stole, wolne
miejsce dookoła, zasilacz podłączony, ręka przy STOP.

**0. Ramię do komputera z bliźniakiem.** Zależnie od tego, gdzie działa bliźniak:
- *lokalnie* — ramię wpięte do tego komputera: port `COMx` (Windows) albo
  `/dev/ttyACM0` (Linux, użytkownik w grupie `dialout`); pokaże go
  `lerobot-twin check` albo „Wykryj porty” w panelu;
- *maszyna wirtualna albo zdalna* (np. Shadow) — *przepuszczenie USB w kliencie*
  (przejściówka pojawia się jako `COMx`), albo
- *most*: na komputerze przy ramieniu `lerobot-mp-bridge --port COMx --listen 0.0.0.0:5555 --allow <adres maszyny z bliźniakiem>`,
  w panelu port `socket://<adres komputera przy ramieniu>:5555`. Odczyt ramienia
  idzie jednym pakietem SYNC READ — przez sieć jeden przebieg na odczyt zamiast
  sześciu. Maszyna zdalna (Shadow stoi w centrum danych) musi **widzieć**
  komputer przy ramieniu: sieć prywatna (Tailscale, ZeroTier) albo przekierowany
  port. Most działa dobrze do ~20 ms RTT; `--stats` pokazuje, ile faktycznie schodzi.

Przed pierwszym połączeniem sprawdź tiki chwytaka swojego ramienia (akapit
**Chwytak** niżej) i wpisz je w `configs/local.yaml`; panel uruchamiaj wtedy
z `--config configs/local.yaml`.

W zakładce Ramię: `feetech`, port, **Połącz** (bez jazdy do domu). Suwaki mają
pokazywać to, co ramię. Sprzęgło + mały ruch jednym suwakiem.

**Do bliźniaka używaj backendu `feetech`.** Backend `lerobot` liczy kąty od
środka zakresu swojej kalibracji, a nie od tiku 2048 jak model — na typowej
kalibracji SO-101 to −2° na `shoulder_pan` i −6,4° na `elbow_flex`, przy
jednostronnym zakresie nawet ~35°; jego chwytak 0..100 to tiki 2031…3524
zamiast 1986…2670. Panel pokazuje oba rozjazdy jako ostrzeżenie. Błędy serw
i limity z EEPROM działają na obu backendach (`lerobot`: rejestr stanu co 3.
odczyt).

**1. Kamery.** Na maszynie wirtualnej/zdalnej (np. Shadow) najpierw przepuść
kamery USB w kliencie. W panelu *Szukaj kamer USB* → *Dodaj*. Kamera, która się otwiera, ale nie daje klatek, to prawie
zawsze brak przepuszczenia (patrz README).

**2. Intrynsyki każdej kamery.** *Pobierz arkusz tablicy*, wydrukuj w skali
100%, **zmierz bok kwadratu linijką**, wpisz. *Zbieraj kadry* i pokazuj tablicę
w rogach kadru, bliżej i dalej, pochyloną — 12+ kadrów, pokrycie 55%+.
*Oblicz i zapisz K*. Residuum ≤ 0,6 px. Tablica trzymana równolegle do
obiektywu nie wyznacza ogniskowej — sesja bez pochyleń (rozrzut < 35°) albo z
ogniskową nie do wiary wychodzi jako **niezaufana**, a kamera z takim K nie
dostanie zaufanej pozy w kroku 3. *Przerwij* niczego nie liczy ani nie zapisuje.

**3. Położenie kamer.** *Pobierz arkusz karty*, wydrukuj, zmierz bok taga,
zegnij kartę, włóż wolny koniec w szczęki (ok. 70 mm ma wystawać). Zaznacz
potwierdzenie, *Start fali*. Ramię zbiera ~20 póz; kamera, której fala się nie
udała, wraca jako niezaufana z powodem — także kamera bez zaufanego K z kroku 2.
*Zapisz wynik kalibracji* (zapisuje też użyty bok taga — zmieniony po fali nie
przejdzie). Wyjmij kartę.

**4. Sprawdzenie Sim-Real.** Zakładka Sim-Real, każda kamera: krawędzie renderu
mają leżeć na krawędziach ramienia i stołu. Mediana > 3 px = coś jest źle
(zmierzony tag? stół w bliźniaku nie na wysokości podstawy?).

**5. Dynamika serw.** Zakładka Trening → *Identyfikuj na połączonym ramieniu*
(~20 s ruchu po 12°; do pozy startowej ramię jedzie 30°/s) → *Zapisz jako
dynamikę stanowiska*. Od teraz trening randomizuje wokół zmierzonego ramienia,
nie katalogu. Z tego ruchu wyznaczalne są **tłumienie, armatura i opóźnienie**
— na ruchu z panelu (5 stawów, 20 s), dla 13 syntetycznych ramion spoza siatki
startowej i szumu 0,05°: tłumienie do 10,5%, armatura do 8,2%, opóźnienie do
1,9 ms; podawany przedział niepewności zawsze obejmował prawdziwy błąd.
Wzmocnienia serw i tarcia suchego ten ruch nie rozróżnia (kp 0,63 dopasowuje
się tak samo dobrze jak 1,0) — zostają z modelu, panel pisze przy nich
„z modelu”, a trening po identyfikacji losuje je szerzej: kp ×0,6–1,5, tarcie
×0,5–2,0. Dopasowanie trwa 7–11 s; STOP, *Połącz* i *Rozłącz* przerywają tylko
nagrywanie — gotowe nagranie dopasowuje się do końca. Nagranie z martwą pętlą
albo nieświeżymi odczytami jest odrzucane.

**6. Trening.** Najszybciej: **douczanie** gotowych polityk na zmierzonej
dynamice — zakładka Trening, *Start z polityki* `reach-v3` / `lift-v3`,
100–300 iteracji (`lerobot-twin train --init …`). Od zera: `reach` 160
iteracji (~2,5 min), `lift` 600 (~30 min). Po treningu polityka sama przechodzi
ewaluację na CPU.

**7. `reach` na ramieniu.** Zakładka Polityki, polityka `reach`, potwierdzenie,
*Uruchom*. Przeciągaj żółtą kulkę celu w 3D — ramię za nią jedzie. Polityka
staje sama, gdy ramię nie nadąża za celem o więcej niż 25° (kolizja, blokada),
i zostaje w zmierzonej pozie. Cel spoza obszaru treningu jest rzutowany na jego
brzeg (kulka wraca na rzut).

**8. `lift` z kamerami.** Kostka 3 cm w kolorze z listy na blacie przed ramieniem,
zakładka Mapa ma ją pokazywać. Polityka `lift-v3` (albo jej douczona wersja),
źródło kostki **kamery**, *Uruchom*. Gdy szczęki zasłonią kostkę, śledzenie
przejmuje ją „w dłoni”. Polityka kończy sama („zadanie wykonane”), gdy kostka
jest 6 cm nad blatem przez 10 taktów — liczonych tylko ze świeżej pozy (kamery
albo „w dłoni”), nigdy z ostatnio widzianej.

**Chwytak.** 0..100 w panelu to te same tiki serwa co w backendzie `feetech`
(`gripper_closed_ticks` … `gripper_open_ticks`, zero w `center_ticks`); w
bliźniaku 0 = −5,4°, 100 = 54,7° kąta szczęki. Domyślne 1986 … 2670 to ramię
autora (z EEPROM-u jego serwa chwytaka). Sprawdź na ramieniu, że przy 0
szczęki się stykają — jeśli nie, zero szczęki w MJCF nie leży w `center_ticks`
i trzeba to poprawić w konfiguracji, zanim zaufa się `lift`: skopiuj
`configs/default.yaml` do `configs/local.yaml` (jest w `.gitignore`), popraw
w sekcji `robot:` klucze `center_ticks`, `gripper_closed_ticks`,
`gripper_open_ticks` (i ewentualnie `baudrate`), a panel uruchamiaj przez
`lerobot-twin --config configs/local.yaml ui`. Wystarczy sama sekcja `robot:`
z tymi kluczami — reszta bierze wartości domyślne.

---

## Polecenia

| | |
|---|---|
| `lerobot-twin check` | czy środowisko jest gotowe (bez renderu); osobno gotowość GPU do treningu |
| `lerobot-twin demo` | przykładowe stanowisko w symulacji → `workspace/twin.json` (`--force`, `--path`) |
| `lerobot-twin ui` | panel w przeglądarce |
| `lerobot-twin --config configs/local.yaml <polecenie>` | konfiguracja ramienia z pliku (tiki chwytaka, baudrate) |
| `lerobot-twin card --tag-mm 50` | arkusz karty kalibracyjnej (PNG, A4) |
| `lerobot-twin board --square-mm 28` | tablica ChArUco do intrynsyk (PNG, A4) |
| `lerobot-twin calib-sim --n 5 --cameras 2` | kalibracja w symulacji, oceniana względem prawdy |
| `lerobot-twin train --task reach --iters 160` | PPO na GPU + ewaluacja na CPU; `--no-rand`, `--spread`, `--init <policy.pt>` (douczanie, np. `--init assets/policies/reach-v3/policy.pt --iters 150`) |
| `lerobot-twin eval <policy.pt> --rand` | ewaluacja polityki w zwykłym MuJoCo |
| `lerobot-twin policies` | zapisane polityki i ich wyniki |
| `lerobot-twin workspace` | co wiadomo o stanowisku |

```bash
pytest -q                                          # całość, z renderem i GPU
pytest -q -m "not render"                          # maszyna bez OpenGL/GPU (testy CUDA pomijają się same, także bez mujoco_warp)
```

---

## Liczby zmierzone na Shadow (RTX A4500, EPYC 4 rdzenie)

| | Shadow | laptop (Intel HD 620) |
|---|---|---|
| budowa sceny | 111 ms | 293 ms |
| 1 s fizyki (CPU) | 2,6 ms | 6,9 ms |
| render 640×480 (cienie, MSAA) | **2,1 ms** | 164 ms |
| render 640×480 bez cieni i MSAA | 1,25 ms | 70 ms |
| render 224×224 / 128×128 | 0,8 / 0,7 ms | – |
| pętla sterowania | 48–50 Hz | 28 Hz |
| MuJoCo Warp, 8192 światy | **1,1 mln kroków fizyki/s** | – |
| środowisko GPU `reach` / `lift`, 4096 światów | 190 / 129 tys. kroków polityki/s | – |
| trening `reach` (15 mln kroków) | **2 min** → CPU 100%, 1,5 mm od celu | – |
| trening `lift` (59 mln kroków) | **28 min** → CPU 100% (96% z randomizacją) | – |
| douczanie `lift` z modelem percepcji (29 mln kroków) | 16 min → CPU 98% z randomizacją, z kamer 8/8 | – |
| douczanie `reach` / `lift` po przeglądzie (13 / 29 mln kroków) | 1,5 / 16 min → CPU 100% / 100% | – |

MuJoCo Warp zgadza się z MuJoCo na CPU: stawy po 2 s fizyki różnią się o
0,001°, obserwacje środowisk GPU i CPU po 30 krokach o 6·10⁻⁵.

---

## Kalibracja kamer

Metoda z [galaxeo-manipulators](https://github.com/machinekind/galaxeo-manipulators/pull/3):
**jedna karta z dwoma AprilTagami, ściśnięta w chwytaku**. Ramię nią macha,
kamery patrzą, a solver wyznacza *jednocześnie* pozę każdej kamery względem
podstawy i pozę karty w dłoni — więc nie trzeba wiedzieć, jak krzywo ktoś ją
włożył. Karta jest wspólna dla wszystkich kamer.

Przed kalibracją położenia każda prawdziwa kamera dostaje **intrynsyki z
tablicy ChArUco** (`calib/intrinsics.py`) — inaczej poza jest dopasowywana do
nominalnego K z 65° pola widzenia i wychodzi pewna siebie i przesunięta.
Tablica to ArUco 5×5, inny słownik niż tagi karty (36h11).

### Wynik w symulacji (`lerobot-twin calib-sim --n 5 --cameras 2`)

| | przed poprawką półpiksela | **po** |
|---|---|---|
| zaufane kamery | 10/10 | 10/10 |
| błąd położenia kamery (mediana) | 0,345 mm | **0,161 mm** |
| błąd obrotu kamery (mediana) | 0,053° | **0,013°** |
| residuum (mediana) | 0,22 px | 0,21 px |

Z panelu, na żywym bliźniaku, dwie kamery: 0,19 mm / 0,009° i 0,31 mm / 0,005°
względem prawdy. Intrynsyki z symulowanej tablicy: ogniskowe z dokładnością
~0,1–1 px, residuum 0,1 px.

### Przestawiona kamera

Po zapisaniu kalibracji panel zapamiętuje kadr odniesienia każdej kamery
i co 2 s porównuje z nim bieżący — korelacją fazową, z **ramieniem wyciętym
maską z bliźniaka** (ruch ramienia daje 0,4 px „przesunięcia”, obrót kamery
o 1° — 9,8 px). Oprócz przesunięcia łapie **obrót wokół osi kamery i zmianę
skali** (dopasowanie podobieństwa na piramidzie ¼–½ kadru); miarą jest
największe przesunięcie narożnika kadru. Ruch ramienia daje najwyżej 0,76 px,
przybliżenie o 1% — 3,7–4,4 px (prawda 4), obrót o 1° — 6,8–7,5 px (prawda 7).
Sprawdzenie kosztuje 10–20 ms na kamerę. Powyżej 3 px piramida robi się czerwona i panel
proponuje szybką relokalizację (krótka fala tylko dla tej kamery). Klatka
starsza niż 1 s nie jest kadrem — zamrożony albo martwy strumień panel pokazuje
jako błąd kamery i otwiera go na nowo.

---

## Dokładność symulowanej kamery

Symulowana kamera dostaje pełną macierz K — ogniskowe w pikselach i punkt
główny. Test renderuje kulki w znanych punktach stołu i porównuje środki ich
**masek segmentacji** z rzutem przez K: przesunięcie systematyczne poniżej
0,15 px, dla trzech różnych K.

| | objaw | poprawka |
|---|---|---|
| znak punktu głównego w MuJoCo | cały kadr przesunięty o 62 px | „środek minus K” w obu osiach |
| ~~półpiksel w pionie~~ | render 0,5 px **niżej** niż rzut K | dawna „kompensacja” była błędem pomiaru: środki czerwonych plam podnosił cień dolnej połowy kulki; maski segmentacji (czysta geometria) pokazały, że to ona przesuwała kadr. Po usunięciu błąd obrotu w kalibracji spadł 4× |
| półpiksel w detektorze AprilTag | rogi o 0,5 px dalej niż w konwencji `projectPoints` | korekta w `tags.detect` |

Renderer MuJoCo Warp (wsadowy, na GPU) liczy promienie przez środki pikseli
z tych samych intrynsyk i zgadza się z OpenGL co do 0,05 px na maskach.

---

## Mapa stołu i kostka

Każdy kadr przerysowany na płaszczyznę blatu, mapy zszyte z wagami — do
oglądania stanowiska (także na blacie w 3D).

Kostka (`perception.CubeDetector.detect_frames`):

1. **start** — część wspólna masek koloru z kamer na wysokości górnej ściany
   (ściana leży w tym samym miejscu dla każdej kamery, boki rozjeżdżają się);
2. **dopasowanie sylwetki** — (x, y, obrót), przy których rzut bryły kostki
   najlepiej pokrywa maski wszystkich kamer; działa z jedną kamerą i z wieloma,
   liczone w wycinku kadru (~17 ms);
3. **ramię to „nie wiem”** — piksele zasłonięte ramieniem (segmentacja z
   bliźniaka, w pozie z serw) nie głosują;
4. **bramka IoU ≥ 0,75** — model zakłada kostkę leżącą; kostkę w powietrzu da
   się „wcisnąć” w jakąś pozę na blacie z IoU 0,36–0,64 i błędem do 1 m;
5. **jedna kamera przy dłoni to za mało** — wykrycie potwierdzone przez jedną
   kamerę (`n_cameras`) blisko chwytaka albo przy kostce w dłoni jest pomijane:
   jedna kamera potrafi „położyć” na blacie kostkę uniesioną w szczękach.
   `n_cameras = -1` (z mapy) i `0` (ramię zasłania kostkę wszystkim) też są
   jednym świadkiem. Jeden świadek przy dłoni jest przyjmowany tylko przy
   pustej szczęce, gdy 3 wykrycia zgadzają się co do 1 cm, a TCP przesunął się
   o ≥ 1,5 cm (kostka potrącona). Kostka, która wypadła z dłoni, opada na blat
   („upuszczona”). Wykrycie ma czas kadru (`t`); stare i powtórzone nie
   przedłużają życia kostki. Zamknięta pętla z `lift-v3`: sama kamera `a` —
   17/17 ułożeń, kamery `a`+`b` — 14/14.

Na blacie w symulacji: **1–3 mm, ~1°**. `CubeTracker` przejmuje kostkę, gdy
kamery jej nie widzą: szczęka **zablokowana** (nie dojeżdża do rozkazu i stoi)
→ kostka jedzie z TCP w pozie zapisanej w chwili chwytu.

### Percepcja w treningu

Polityka `lift` uczona na **prawdziwej** pozycji kostki podnosiła ją 20/20
razy — i tylko **7/20**, gdy pozycja przychodziła jak z kamer: co 100 ms,
150 ms spóźniona. Podczas podnoszenia widziała kostkę „pod sobą” i wracała po
nią. Dlatego trening widzi kostkę tak, jak ją da percepcja
(`Randomization.cube_*`): opóźnienie 0–3 takty, odświeżanie co 1–3 takty,
szum 2 mm, obrót złożony do ±45° — a nagroda liczy się z prawdy.

| pełny łańcuch z kamerami na bliźniaku (8 ułożeń kostki) | podniesione |
|---|---|
| `lift-v1` — uczona na prawdziwej pozycji kostki | 1/8 |
| `lift-v2` — `lift-v1` douczona z modelem percepcji (300 iteracji, 16 min) | **8/8** |
| `lift-v3` — `lift-v2` douczona po przeglądzie (300 iteracji); kończy sama, kostka 9,6–11,8 cm | **8/8** |

Od zera z modelem percepcji uczenie szło bardzo wolno (0% po 220 iteracjach) —
douczanie z polityki, która już umie chwytać, doszło do 98% w 300.
Pilnuje tego `tests/test_twin_lift_from_cameras.py` — w czasie symulowanym,
więc wynik nie zależy od obciążenia maszyny.

## Polityki bazowe

W repozytorium są dwie gotowe polityki (`assets/policies/`), widoczne w panelu
i w `lerobot-twin policies` obok własnych ze stanowiska:

| | zadanie | CPU bez / z randomizacją |
|---|---|---|
| `reach-v3` | dojazd TCP do punktu, nadgarstek blisko pozy domowej | 100% / 100%, 1,2 mm od celu |
| `lift-v3` | chwyt i podniesienie kostki z kamer | 100% / 100% |

Obie uczone wokół modelu Menagerie — po identyfikacji dynamiki na swoim ramieniu
douczyć je (`--init`, 100–300 iteracji), zamiast uczyć od zera. Pliki niosą też
wagi **krytyka** z treningu: douczanie z krytykiem od zera psuło politykę
(`reach-v1`: 98% → 9% sukcesu po 100 iteracjach, dojeżdżała i odpływała od celu,
bo niedouczony krytyk przy małej eksploracji nie odróżnia „przy celu” od „kilka
mm obok”). Polityka bez zapisanego krytyka dostaje 30 iteracji rozgrzewki, w
których uczy się tylko krytyk. „Nagroda” w logu treningu zawiera wartość stanu
doliczaną na końcu epizodu — rośnie razem z krytykiem, o jakości mówi *sukces*.

`lift-v3` pod limitem stawu bywa tylko tam, gdzie zadanie tego wymaga (kostka
blisko podstawy — łokieć zgięty do +92°): 10 taktów na 20 epizodów, `lift-v2`
26 i wymachy po chwycie. `reach-v2` kręciła nadgarstkiem do +150° w każdym
epizodzie (obrót nie zmienia TCP, więc nagroda go nie widziała); `reach-v3`
uczona z karą za obrót od pozy domowej (`roll_penalty`): mediana 13°, najwyżej
48°, zero taktów przy limicie. Pilnuje tego `tests/test_twin_policies.py` —
zachowanie **dostarczanych** plików polityk, nie tylko środowiska.

---

## Uczenie ze wzmocnieniem

| | |
|---|---|
| zadania | `reach` — TCP do punktu (sukces < 2 cm); `lift` — chwyt i podniesienie kostki 3 cm o 6 cm; epizod `lift` kończy się po 10 taktach sukcesu (liczone jak limit czasu — wartość stanu dalej jest bootstrapowana), kara za jazdę pod limity stawów i za ruch po sukcesie |
| akcja | 6 liczb w [−1, 1]: przyrost celu stawów ramienia (≤ 0,05 rad na takt, 20 Hz), cel chwytaka bezwzględnie, ograniczony jak w nadzorze |
| obserwacja | stawy i cele stawów (znormalizowane), TCP, cel albo kostka (położenie, obrót 6D), poprzednia akcja — **jedna definicja** (`rl/task.py`) dla CPU, GPU i ramienia; TCP i kostka liczone po krokach fizyki, zgodne z kątami stawów |
| randomizacja | wzmocnienie, tłumienie, armatura i tarcie serw, masa i tarcie kostki, opóźnienie akcji, szum kątów — wokół `Workspace.dynamics` (identyfikacja); **percepcja kostki** jak z kamer (opóźnienie, odświeżanie, szum, symetria). `--no-rand` = dokładnie zmierzona dynamika i jej opóźnienie, model percepcji zostaje; ewaluacja „CPU bez randomizacji” też idzie na zmierzonym ramieniu, nie na Menagerie |
| środowiska | `LeRobotMP/TwinReach-v0`, `LeRobotMP/TwinLift-v0` (Gymnasium, CPU); `rl.batch.BatchEnv` (MuJoCo Warp, GPU) |

---

## Jak to jest zbudowane

```
src/lerobot_mp/twin/
├── robots.py        opis ramienia: MJCF, stawy, TCP, osie chwytaka, szczęki, zakresy fali
├── kinematics.py    FK/IK przez MuJoCo; IK z priorytetem pozycji (5 osi SO-101)
├── scene.py         scena z MjSpec: stół, ramię, kamery z pełnym K, karta, obiekty, panele, czujniki chwytu
├── collision.py     kolizje pozy i drogi do niej
├── workspace.py     stanowisko w JSON: ramię, port, stół, karta, kamery (+ prawda symulowanych), dynamika
├── cameras.py       wykrywanie kamer, strumienie bez lustra, kamery symulowane
├── runtime.py       pętla sterowania w wątku, nadzór, wątek renderujący
├── perception.py    mapa stołu na żywo, kostka z kamer, śledzenie kostki w dłoni
├── cli.py           lerobot-twin
├── calib/
│   ├── tags.py        AprilTag 36h11 + korekta półpiksela
│   ├── handeye.py     solver wielu kamer ze wspólną kartą
│   ├── card.py        karta: geometria, poza w szczękach, arkusz
│   ├── intrinsics.py  ChArUco: arkusz, zbieranie ujęć, K i dystorsja, sesja w symulacji
│   ├── session.py     fala: szukanie, celowanie, kadr, rozrzut, bramki
│   ├── simulate.py    sesja w symulacji względem prawdy
│   └── topdown.py     rzut kadru na blat
├── rl/
│   ├── task.py        obserwacja, akcja, nagroda, starty - numpy i torch
│   ├── env.py         Gymnasium na CPU
│   ├── batch.py       tysiące światów MuJoCo Warp na GPU
│   ├── ppo.py         PPO
│   ├── policy.py      plik polityki (sieć + normalizacja + zadanie + wyniki)
│   ├── evaluate.py    ewaluacja w zwykłym MuJoCo
│   ├── randomize.py   dynamika zmierzona i zakresy randomizacji
│   ├── sysid.py       identyfikacja dynamiki serw
│   └── runner.py      polityka na żywym bliźniaku / ramieniu
└── ui/
    ├── app.py         panel (viser)
    ├── bridge.py      scena MuJoCo w przeglądarce (węzeł na ciało)
    ├── jobs.py        zadania w tle: trening (proces), kalibracje, identyfikacja
    └── watch.py       wykrywanie przestawionej kamery

assets/robots/so101/   model z MuJoCo Menagerie (Apache-2.0), commit ac6b2b0
```
