"""Cyfrowy blizniak: ramie, kamery i scena w jednym ukladzie wspolrzednych.

Warstwy, od dolu:

* `robots`      - opis ramienia (MJCF, stawy, TCP) niezalezny od tego, czy jest prawdziwe,
* `kinematics`  - kinematyka prosta i odwrotna liczona przez MuJoCo na tym opisie,
* `calib`       - kalibracja kamer z karty trzymanej w chwytaku (za galaxeo-manipulators),
* `workspace`   - to, co sesja wie o stanowisku: ramie, kamery, stol - zapisywane na dysk,
* `scene`       - scena MuJoCo zlozona z workspace'u: ramie, stol, kamery tam, gdzie stoja,
* `env`         - srodowisko Gymnasium na tej scenie, do uczenia ze wzmocnieniem,
* `ui`          - panel webowy (viser) do zarzadzania ramieniem i kamerami.

Symulacja i rzeczywistosc dziela jeden workspace. Kamera skalibrowana na
prawdziwym stanowisku trafia do sceny dokladnie w swojej pozie i ze swoja
ogniskowa - i to jest caly most sim-2-real: polityka widzi w symulacji ten sam
kadr, ktory zobaczy na biurku.
"""
