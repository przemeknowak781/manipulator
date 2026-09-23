# Prototypy: rekonstrukcja z sylwetek i chwyty SO-101

Pomiary do [docs/RESEARCH_3D.md](../../docs/RESEARCH_3D.md). To prototypy
badawcze, nie kod aplikacji — działają na scenie bliźniaka z `lerobot_mp.twin`.

| | |
|---|---|
| `hull_and_grasp_bench.py` | kompilacja sceny z obiektami, `recompile`, szybkość symulacji, maski z renderu segmentacji, **wspólna otoczka wizualna** z 4 kamer (ramię ukryte / ramię jako „nie wiem" / ramię jako tło), próbne chwyty z góry w MuJoCo |
| `grasp_reachability.py` | gdzie SO-101 chwyci z góry, a gdzie z boku — siatka punktów, IK projektu, błąd pozycji i orientacji |
| `jaw_geometry.py` | położenie czubków obu szczęk względem TCP przy różnym otwarciu |
| `*_out.json` | wyniki z laptopa (i5-7200U, Intel HD 620) |

Skrypty zapisują wyniki i rendery w **bieżącym katalogu** — uruchamiaj je
z katalogu tymczasowego, a nie z korzenia repozytorium:

```bash
mkdir -p /tmp/r3d && cd /tmp/r3d
python -B /sciezka/do/repo/research/3d/hull_and_grasp_bench.py
```

`hull_and_grasp_bench.py` renderuje (OpenGL) — na maszynie bez ekranu ustaw
`MUJOCO_GL=egl`.
