"""Kalibracja kamer wzgledem ramienia z jednej karty trzymanej w chwytaku.

Metoda i wiekszosc kodu pochodzi z `sim/calib/` w
[machinekind/galaxeo-manipulators](https://github.com/machinekind/galaxeo-manipulators)
(PR #3, commit 02641c4), gdzie zostala zmierzona wzgledem prawdy z symulacji:
0,21 mm i 0,06 stopnia mediany polozenia kamery na 20 losowych stanowiskach.
Tutaj jest uogolniona z jednego ramienia (Galaxea A1X) i jednej kamery na
dowolne ramie z `twin.robots` i dowolnie wiele kamer naraz.

    tags      - detekcja AprilTag 36h11 i poza pojedynczego taga (OpenCV aruco)
    handeye   - pozy kamer i poza karty w dloni, lacznie, z bledu reprojekcji
    card      - geometria karty, jej nominalna poza w szczekach, arkusz do druku
    topdown   - kadr przerysowany na plaszczyzne stolu; mapa z wielu kamer
"""
