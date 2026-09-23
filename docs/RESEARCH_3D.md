# Rozpoznanie: modele 3D przedmiotów z biurka i dobór chwytu

Stan na 2026-09-23. Cel: bliźniak sam uzupełnia scenę o przedmioty leżące na
biurku, a ramię wie, jak je najlepiej chwycić szczęką SO-101.

**Zastrzeżenia.** Sieć, z której robione było rozpoznanie, przechwytywała TLS,
więc strony nie otwierały się bezpośrednio — fakty o modelach pochodzą ze
streszczeń wyszukiwarki. **Licencje przed decyzją potwierdzić w plikach
LICENSE.** Liczby oznaczone jako *zmierzone* to wyniki prototypów z
[`research/3d/`](../research/3d/) na laptopie z i5-7200U i Intel HD 620
(4 kamery 640×480 wokół pola roboczego, scena bliźniaka z MuJoCo 3.14).

## W skrócie

Najmniejszy krok, który daje wartość, **nie potrzebuje żadnej sieci
neuronowej i działa na CPU**: maski z różnicy obrazu względem pustego stołu,
wspólna otoczka wizualna ze znanych póz kamer, dekompozycja wypukła, obiekt
w scenie, chwyt liczony analitycznie i sprawdzony w MuJoCo. Modele Mety
wchodzą później: do rozdzielania i nazywania obiektów (SAM 3.1) oraz do
domykania niewidocznych części kształtu (SAM 3D Objects z metryczną pointmapą
z MapAnything).

Z Mety naprawdę przydają się:

| model | po co |
|---|---|
| **SAM 3.1** | maski obiektów z promptu tekstowego, śledzenie |
| **MapAnything** | jedyny model Mety, który przyjmuje **nasze K i pozy kamer** i oddaje skalę metryczną |
| **SAM 3D Objects** | domyka niewidoczne części kształtu (siatka + poza) |
| **DINOv2** | rozpoznawanie tego samego obiektu w kolejnych ujęciach |
| **EdgeTAM** | lekkie śledzenie masek |

VGGT pasuje gorzej: ignoruje znane pozy i nie daje skali. CoTracker3 i ShapeR
mają licencje niekomercyjne.

---

## 1. Rekomendowany potok

1. **Maski.** Różnica względem zdjęcia pustego stołu plus maska ramienia
   renderowana z bliźniaka (40–100 ms na kamerę, *zmierzone*). Do rozdzielenia
   stykających się obiektów i nadania im nazw: SAM 3.1 z promptem tekstowym;
   lokalnie zamiennie EfficientSAM3 (maski) i EdgeTAM (śledzenie).
2. **Geometria metryczna ze znanych póz: wspólna otoczka wizualna** (visual
   hull) dla wszystkich obiektów naraz. Każdy woksel przypisany do obiektu
   głosowaniem widoków. **Piksele ramienia to „nie wiem", a nie tło.**
   Bez sieci neuronowych i bez głębi. *Zmierzone* przy wokselu 2,5 mm:
   - czas 1,3–2,3 s;
   - pokrycie prawdziwej bryły 94–100%;
   - objętość 1,1–1,4× prawdziwej dla brył wypukłych, ok. 5× dla kubka
     (sylwetki nie odtworzą wnętrza);
   - szerokość zawyżona o 7–12 mm;
   - liczenie każdego obiektu **osobno** gubi 53–70% kubka, puszki i długopisu
     przez wzajemne zasłanianie, a ramię potraktowane jako tło gubi do 85% —
     stąd wspólna otoczka i „nie wiem" dla ramienia.
3. **Pełny kształt na żądanie.** SAM 3D Objects z jednego obrazu i maski daje
   siatkę i pozę (R, t, s) w układzie kamery (konwencja PyTorch3D — przeliczyć
   na OpenCV). Model przyjmuje zewnętrzną pointmapę: zamiast monokularnej
   podajemy metryczną z MapAnything, potem dopasowujemy s, R, t do sylwetek
   ze wszystkich kamer i przechodzimy przez `T_cam2base`. Wariant wielowidokowy:
   MV-SAM3D.
4. **Biblioteka obiektów** (jak w SyncTwin): dla znanego obiektu tylko nowa
   poza w `qpos` jego wolnego stawu, bez kompilacji sceny; rozpoznanie po
   embeddingu DINOv2.
5. **Wstawienie do MJCF.** Siatka wizualna plus kolizyjna z CoACD, każdy
   kawałek jako osobny mesh (MuJoCo zastępuje mesh jego otoczką wypukłą).
   Masa = objętość × założona gęstość, tarcie losowane. Po dodaniu lub
   usunięciu obiektu przebudowa sceny z configu, jak dziś `scene.build`:
   0,25–0,47 s z 5 obiektami (*zmierzone*). `spec.recompile` jest szybszy
   (0,13–0,27 s), ale ma zgłoszone błędy w 3.14
   ([mujoco#3585](https://github.com/google-deepmind/mujoco/issues/3585)).
6. **Chwyt liczony analitycznie w tym, co SO-101 faktycznie osiągnie**
   (*zmierzone* IK projektu, [`grasp_reachability.py`](../research/3d/grasp_reachability.py)):
   - przy 5 osiach kierunek podejścia musi leżeć w **pionowej płaszczyźnie
     przez oś podstawy** — podejście z boku daje błąd orientacji 83–92°;
   - chwyt z góry działa dla r = 15–30 cm przy TCP 2–5 cm nad blatem, przy
     10 cm już nie; przy r = 35 cm orientacja odchyla się o 9–27°;
   - chwyt poziomy tylko od r ≥ 25 cm;
   - więc próbkujemy trzy parametry — punkt, pochylenie, obrót — zamiast całego
     SE(3); kandydaci antypodalnie na siatce, jak w ACRONYM;
   - **szczęka jest na zawiasie**: stała 20 mm od TCP, ruchoma przy pełnym
     otwarciu cofa się o 76 mm wzdłuż podejścia
     ([`jaw_geometry.py`](../research/3d/jaw_geometry.py));
   - walidacja w MuJoCo z losowaniem pozy (±3–5 mm) i tarcia. Próbne chwyty:

     | obiekt | ustawienie | wynik |
     |---|---|---|
     | klocek 60 mm | TCP na środku | porażka |
     | klocek 60 mm | przesunięcie na stałą szczękę | sukces z obu stron |
     | klocek 40 mm | przesunięcie na stałą szczękę | sukces |
     | długopis 11 mm | zależnie od otwarcia | raz sukces, raz porażka — stąd losowanie |

   - etap 2: **GraspGen-X**, jedyny generator uczony, który przyjmuje dowolny
     chwytak (URDF + ruch zamykania).
7. **Odświeżanie zdarzeniowe**: detektor zmian (kadr kontra render bliźniaka,
   z pominięciem maski ramienia), po każdym chwycie lub odłożeniu, na żądanie
   z UI. Nie rekonstruować, gdy ramię jest nad stołem.

---

## 2. Modele

*n/d* — nie dotyczy. Licencje wg streszczeń wyszukiwarki, **do potwierdzenia**.

| model | co daje | znane pozy | skala metr. | sprzęt | licencja / komercja | link |
|---|---|---|---|---|---|---|
| SAM 3 / 3.1 (Meta) | maski i ID instancji z tekstu, śledzenie | n/d | n/d | 848M param. | SAM License: komercja z wyłączeniami; wagi bramkowane | [repo](https://github.com/facebookresearch/sam3) |
| EfficientSAM3 | destylat SAM 3 | n/d | n/d | brzegowy | kod Apache 2.0; wagi niepotwierdzone | [repo](https://github.com/SimonZeng7108/efficientsam3) |
| EdgeTAM (Meta) | śledzenie masek | n/d | n/d | 16 FPS na telefonie | Apache 2.0 | [repo](https://github.com/facebookresearch/EdgeTAM) |
| SAM 3D Objects (Meta) | siatka + R, t, s | nie | przybliżona, z pointmapy | NVIDIA ≥ 32 GB; ok. 31 s/obiekt | SAM License; bramkowane | [repo](https://github.com/facebookresearch/sam-3d-objects) |
| MapAnything (Meta + CMU) | metryczne pointmapy, głębia, pozy | **tak** | **tak** | GPU | kod Apache; wagi `map-anything` NC, **`map-anything-apache` Apache** | [repo](https://github.com/facebookresearch/map-anything) |
| VGGT (Meta) | pozy, głębia, pointmapy | nie | nie | ok. 1,9 GB/klatkę | VGGT-1B NC; wariant komercyjny po formularzu | [repo](https://github.com/facebookresearch/vggt) |
| ShapeR (Meta) | metryczna siatka obiektu | tak | tak | GPU | CC-BY-NC | [repo](https://github.com/facebookresearch/ShapeR) |
| CoTracker3 (Meta) | śledzenie punktów | n/d | n/d | GPU | CC-BY-NC | [repo](https://github.com/facebookresearch/co-tracker) |
| DINOv2 / v3 (Meta) | cechy do rozpoznawania | n/d | n/d | małe warianty na CPU | v2 Apache; v3 własna (komercja z bramką) | [v2](https://github.com/facebookresearch/dinov2) |
| Depth Anything 3 | głębia spójna między widokami | tak | wariant Metric | 0,08–1,4 mld param. | Small/Base/Metric Apache; Large/Giant NC | [repo](https://github.com/ByteDance-Seed/depth-anything-3) |
| MoGe-2 | metryczna pointmapa z 1 obrazu | nie | tak | ViT-S/B/L | MIT | [repo](https://github.com/microsoft/moge) |
| FoundationPose | poza 6D | n/d | z głębi, której nie mamy | GPU | NC; wersja Isaac na licencji NVIDIA | [repo](https://github.com/NVlabs/FoundationPose) |
| GraspGen / GraspGen-X | chwyty 6-DoF; X dla dowolnego chwytaka | n/d | n/d | CUDA | kod Apache, wagi NVIDIA OML | [repo](https://github.com/NVlabs/GraspGen) |
| Contact-GraspNet / M2T2 / AnyGrasp | chwyty dla szczęk typu Franka | n/d | n/d | GPU | CGN niekomercyjna; M2T2 MIT; AnyGrasp licencja na maszynę | [CGN](https://github.com/NVlabs/contact_graspnet), [M2T2](https://github.com/NVlabs/M2T2) |
| CoACD | dekompozycja wypukła | n/d | n/d | CPU, dziesiątki sekund/siatkę | MIT | [repo](https://github.com/SarahWeiii/CoACD) |

---

## 3. CPU, GPU i DGX Spark

**Na laptopie (i5-7200U, 8 GB)** — szacunki z FLOPs i zmierzonej
przepustowości 34–72 GFLOPS fp32, w praktyce wolniej:

| | czas |
|---|---|
| SAM 3 | ok. 1–2,5 min/obraz (3,4 GB wag przy 8 GB RAM) |
| MapAnything | ok. 1,5–3 min na 4 widoki |
| VGGT | ok. 3–6 min |
| SAM 3D Objects | niewykonalne |

*Zmierzone*: próba chwytu to 0,1–0,45 s symulacji plus 0,03–0,3 s IK;
nieudane IK trwa 1–5 s, więc przyda się analityczne IK dla 5 osi albo mapa
osiągalności.

**DGX Spark**: 128 GB pamięci wspólnej zmieści wszystkie te modele. To jednak
aarch64 z architekturą sm_121 — rozszerzenia CUDA (np. kaolin, potrzebny SAM 3D)
trzeba budować ze źródeł. Są relacje z uruchamiania SAM 3D na Sparku; czasów
nie potwierdzono.

**Chmura**: endpointy SAM 3 ok. 0,005 $/obraz i SAM 3D ok. 0,02 $/obiekt;
wynajem A100 80 GB ok. 1,4 $/h. Karta 24 GB nie wystarczy dla SAM 3D.

---

## 4. Ryzyka

- **Licencje.** Niekomercyjne: główne wagi MapAnything (jest wariant Apache),
  VGGT-1B, CoTracker3, ShapeR, DA3 Large/Giant, Contact-GraspNet.
- **Wnętrza i spody** — otoczka wizualna ich nie widzi. Chwyt kubka za ściankę
  z góry i tak jest dla SO-101 nieosiągalny (błąd orientacji 16°).
- **Przezroczyste i błyszczące przedmioty** psują głębię; sylwetki są
  odporniejsze, ale segmentacja szkła niepewna.
- **Zasłanianie przez ramię** — ramię musi być renderowane z dokładnymi kątami;
  przy ramieniu nad stołem objętość zasłoniętych obiektów rośnie 1,6–2,6×.
- **Dryf kalibracji** — poruszona kamera „zjada" obiekt przy rzeźbieniu;
  potrzebna okresowa kontrola.
- **Synchronizacja** — kadry ze wszystkich kamer z tej samej chwili, scena w bezruchu.
- **MuJoCo 3.14** — trzymać się przebudowy sceny z configu zamiast `recompile`.
- **Prywatność** — wysyłanie zdjęć biurka do chmurowych endpointów to decyzja,
  nie szczegół techniczny.
- **Masa i tarcie nieznane** — pomysł do sprawdzenia: odczyt obciążenia serw
  STS3215 po podniesieniu (serwa raportują obciążenie; rejestr `Present_Load`
  już czytamy w diagnostyce).

## 5. Pytania do rozstrzygnięcia

1. Czy projekt ma być komercyjny? To odcina połowę tabeli modeli.
2. Ile kamer i jaka rozdzielczość na docelowym stanowisku?
3. Czy kamera na nadgarstku ma skanować nowe obiekty (pozy z kinematyki, dużo
   więcej widoków)?
4. Lista przedmiotów testowych — szkło, metal, cienkie?
