# Cyfrowy bliźniak stanowiska SO-101

Symulacja MuJoCo, która wie o biurku to samo, co prawdziwe stanowisko: gdzie
stoi ramię, gdzie stoją kamery, jaką mają ogniskową. Kamera skalibrowana na
biurku trafia do symulacji **dokładnie w swojej pozie i ze swoją macierzą K** —
i to jest cały most sim-2-real: polityka uczona w symulacji widzi ten sam kadr,
który zobaczy na prawdziwym stole.

```
  prawdziwe stanowisko                     bliźniak (MuJoCo)
 ┌──────────────────────┐   kalibracja   ┌──────────────────────────┐
 │ SO-101 na COM12      │ ─────────────► │ ramię z MuJoCo Menagerie │
 │ kamery USB dookoła   │  karta w dłoni │ kamery w tych samych     │
 │ biurko               │                │ pozach, z tym samym K    │
 └──────────────────────┘                └────────────┬─────────────┘
            ▲                                         │
            │  te same kąty stawów                    ▼
            └──────────────────────────── środowisko RL (Gymnasium)
                                          mapa stołu z wielu kamer
```

## Stan

| | |
|---|---|
| **Gotowe i przetestowane** | model ramienia, kinematyka prosta i odwrotna, scena z kamerami o pełnym K, sprawdzanie kolizji, **kalibracja wielu kamer z karty w chwytaku** (sprawdzona w symulacji względem prawdy), mapa stołu z wielu kamer, zapis stanowiska, pętla sterowania ramieniem (symulowanym i prawdziwym) |
| **W budowie** | środowisko Gymnasium, panel webowy (viser), kalibracja na prawdziwym ramieniu z prawdziwymi kamerami |
| **Rozpoznanie** | rekonstrukcja 3D przedmiotów z biurka i dobór chwytu — patrz [HANDOFF.md](../HANDOFF.md) |

Szczegóły, decyzje i następne kroki: **[HANDOFF.md](../HANDOFF.md)**.

---

## Instalacja

### Windows (laptop przy ramieniu)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[twin]"
lerobot-twin check
```

### Linux / NVIDIA DGX Spark

Spark to ARM64 (aarch64). Wszystkie zależności bliźniaka mają gotowe koła dla
Linux aarch64 albo są czystym Pythonem — sprawdzone na PyPI we wrześniu 2026:
MuJoCo 3.14 (cp312–cp315), OpenCV 5.0, NumPy 2.5, msgspec; viser, Gymnasium
i pyserial to czysty Python.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[twin]"
export MUJOCO_GL=egl          # render bez ekranu, na GPU
lerobot-twin check
```

`MUJOCO_GL=egl` jest potrzebne, gdy nie ma okna (SSH, kontener). Na pulpicie
działa też domyślne `glfw`.

`lerobot-twin check` niczego nie renderuje — sprawdza pakiety, wczytuje model
ramienia, porównuje jego kinematykę z niezależnym modelem Articulusa i wypisuje
przejściówki USB-serial:

```
  OK   mujoco                       3.14.0
  OK   model SO-101 (menagerie)     9 cial, 348164 trojkatow
  OK   zgodnosc z Articulusem       2.3 um (musi byc < 10 um)
  OK   porty USB-serial             COM12 (USB-Enhanced-SERIAL CH343 (COM12))
```

---

## Polecenia

| | |
|---|---|
| `lerobot-twin check` | czy środowisko jest gotowe (bez renderu) |
| `lerobot-twin card --tag-mm 50` | arkusz karty kalibracyjnej do druku (PNG, A4 w poziomie) |
| `lerobot-twin calib-sim --n 5 --cameras 2` | kalibracja w symulacji, oceniana względem prawdy (**renderuje**) |
| `lerobot-twin workspace` | co wiadomo o stanowisku: kamery, ich kalibracja, karta |

Testy:

```bash
pytest -q                                   # wszystko, łącznie z renderem
pytest -q -k "not renders and not full_session"   # bez renderu (np. na maszynie bez GPU)
```

---

## Kalibracja kamer

Metoda pochodzi z [galaxeo-manipulators](https://github.com/machinekind/galaxeo-manipulators/pull/3):
**jedna karta z dwoma AprilTagami, ściśnięta w chwytaku**. Ramię nią macha,
kamery patrzą, a solver wyznacza *jednocześnie* pozę każdej kamery względem
podstawy i pozę karty w dłoni — więc nie trzeba wiedzieć, jak krzywo ktoś ją
włożył. Przy wielu kamerach karta jest wspólna dla wszystkich, więc kamera,
która widzi ją dobrze, pomaga tej, która widzi ją gorzej.

### Wynik w symulacji

Losowe stanowiska, kamery 45–75 cm od stołu, karta przekrzywiona w dłoni do
12 mm i 10°. Sesja widzi tylko to, co widziałaby na biurku: kadry, zmierzone
kąty stawów, K kamer i *nominalną* pozę karty.

| | 1 kamera (3 stanowiska) | 2 kamery (3 stanowiska) |
|---|---|---|
| zaufane kamery | 3/3 | 6/6 |
| błąd położenia kamery | mediana 0,20 mm | mediana **0,15 mm**, najgorzej 0,25 mm |
| błąd obrotu kamery | 0,115°¹ | mediana **0,055°**, najgorzej 0,068° |
| residuum | 0,17 px | 0,16 px |
| poz na sesję | ok. 15 | ok. 19 dla obu kamer razem |

¹ przed poprawką konwencji rogów tagów — patrz niżej. Galaxeo dla porównania:
0,21 mm i 0,06° na jednej kamerze i ramieniu z sześcioma osiami.

```bash
lerobot-twin calib-sim --n 5 --cameras 2
```

### Na prawdziwym ramieniu

1. **Wydrukuj kartę** w skali 100%:
   ```bash
   lerobot-twin card --tag-mm 50 --out karta.png
   ```
2. **Zmierz linijką** bok czarnego kwadratu na wydruku. Drukarki skalują, a ta
   liczba ustala skalę całej kalibracji — to jedyna rzecz, której symulacja
   nie sprawdzi za Ciebie.
3. Wytnij pasek, zegnij po linii przerywanej tagami na zewnątrz, wklej w środek
   sztywną tekturę. **Zgięcie to koniec karty** — dzięki temu drugi tag po
   złożeniu leży dokładnie za pierwszym.
4. Ściśnij wolny koniec w szczękach, ok. 70 mm ma wystawać. Dokładna poza nie
   ma znaczenia — kalibracja ją wyznacza.
5. Postaw kamery tak, żeby widziały przestrzeń nad stołem przed ramieniem.
6. Uruchom sesję *(polecenie na prawdziwe ramię jest w budowie — przepis
   w [HANDOFF.md](../HANDOFF.md))*.
7. Wyjmij kartę. Nic innego na stanowisku nie ma znacznika.

Kamera, której fala się nie udała, wraca jako **niezaufana z podanym powodem**
(za duże residuum, za mały rozrzut obrotów, za mało obserwacji), zamiast
z pewną siebie złą pozą.

---

## Dokładność symulowanej kamery

Symulowana kamera dostaje pełną macierz K — ogniskowe w pikselach i punkt
główny — a nie samo pole widzenia. Test renderuje kulki w znanych punktach
stołu i porównuje środki plam z rzutem przez K, dla trzech różnych K (w tym
z punktem głównym przesuniętym i fx ≠ fy): **przesunięcie systematyczne
poniżej 0,15 px, pojedyncze błędy poniżej 0,4 px**.

Po drodze wyszły trzy rzeczy, które cicho psułyby sim-2-real. Wszystkie są
teraz pilnowane testami:

| | objaw | poprawka |
|---|---|---|
| znak punktu głównego w MuJoCo | cały kadr przesunięty o 62 px w poziomie | przesunięcie liczone jako „środek minus K” w obu osiach |
| półpiksel w pionie w MuJoCo | render stale 0,5 px wyżej niż rzut K, dla każdego K | kompensacja w `scene.py` |
| półpiksel w detektorze AprilTag | rogi o 0,5 px dalej niż w konwencji `projectPoints` | korekta w `tags.detect`; **błąd obrotu kamery spadł o połowę** |

Ostatnia dotyczy też galaxeo, z którego pochodzi detektor: `CORNER_REFINE_APRILTAG`
oddaje rogi w konwencji narożników pikseli, a model otworkowy OpenCV liczy
w konwencji środków.

---

## Mapa stołu z wielu kamer

Każdy kadr przerysowany na płaszczyznę blatu, mapy zszyte z wagami (kamera
patrząca z góry liczy się bardziej niż ta z boku). Przedmiot na stole ląduje
na mapie w swoim prawdziwym (x, y), niezależnie od tego, skąd patrzy kamera —
galaxeo zmierzyło, że właśnie ta zmiana dała pierwsze udane nalania w zamkniętej
pętli. W teście podstawa kostki trafia **0,3 px** od swojego miejsca z każdej
z dwóch kamer, a dwie kamery pokrywają 99% blatu. W odróżnieniu od oryginału
rzut liczony jest przez `projectPoints`, więc uwzględnia dystorsję obiektywu.

---

## Jak to jest zbudowane

```
src/lerobot_mp/twin/
├── robots.py        opis ramienia: MJCF, stawy, TCP, osie chwytaka, zakresy fali
├── kinematics.py    FK/IK przez MuJoCo; IK z priorytetem pozycji (5 osi SO-101)
├── scene.py         scena z MjSpec: stół, ramię, kamery z pełnym K, karta, obiekty
├── collision.py     kolizje pozy i drogi do niej, na scenie bliźniaka
├── workspace.py     stanowisko w JSON: ramię, port, stół, karta, kamery + kalibracja
├── cameras.py       wykrywanie kamer, strumienie bez lustra, kamery symulowane
├── runtime.py       pętla sterowania w wątku: nadzór bezpieczeństwa, sim albo sprzęt
├── cli.py           lerobot-twin
└── calib/
    ├── tags.py      AprilTag 36h11 (z galaxeo) + korekta półpiksela, render taga
    ├── handeye.py   solver wielu kamer ze wspólną kartą (z galaxeo, uogólniony)
    ├── card.py      geometria karty, poza w szczękach, arkusz do druku
    ├── session.py   fala: szukanie, celowanie, pilnowanie kadru i rozrzutu, bramki
    ├── simulate.py  sesja w symulacji oceniana względem prawdy
    └── topdown.py   mapa stołu z jednej i z wielu kamer (z galaxeo, z dystorsją)

assets/robots/so101/   model z MuJoCo Menagerie (Apache-2.0), commit ac6b2b0
```

Model ramienia to MJCF z [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101),
wyprowadzony z `so101_new_calib.xml` — konwencji nowej kalibracji LeRobota, tej
samej, w której pracuje backend `feetech`. Zgodność z niezależnym modelem
Articulusa: **2 µm** w 200 losowych pozach po pełnym zakresie stawów.
