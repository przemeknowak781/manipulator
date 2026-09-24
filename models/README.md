# Modele MediaPipe

Aplikacja `lerobot-mp` śledzi dłoń i ramię operatora modelami Google MediaPipe.
Dwa modele używane w domyślnej konfiguracji leżą w repozytorium, więc świeży
klon działa bez internetu:

| Plik | Do czego | Rozmiar | SHA-256 |
|---|---|---|---|
| `hand_landmarker.task` | śledzenie dłoni (wszystkie tryby) | 7 819 105 B | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |
| `pose_landmarker_lite.task` | tryb `arm` (całe ramię operatora) | 5 777 746 B | `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a` |

Źródła (te same adresy są w `src/lerobot_mp/config.py`, pola `model_url`):

- https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
- https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

## Licencja

Modele są dziełem Google i są udostępniane na licencji **Apache-2.0**
(karty modeli MediaPipe: *Hand Landmarker* i *Pose Landmarker*,
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker i
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker).
Pliki są skopiowane bez zmian. Zob. też [NOTICE](../NOTICE).

## Skąd aplikacja bierze model

1. Ścieżka z konfiguracji (`tracker.model_path`, `arm.model_path`). Domyślna
   `models/<plik>.task` przy uruchomieniu z klonu repozytorium wskazuje na ten
   katalog, niezależnie od katalogu, z którego wystartowano program. Ścieżkę
   względną wpisaną samodzielnie (inną niż domyślna) liczy się od katalogu
   bieżącego.
2. Jeśli pliku nie ma, aplikacja pobiera go z adresu `model_url` przy
   pierwszym użyciu (`vision/tracker.py`, `download_model`). Bez internetu
   kończy się to komunikatem z gotowym poleceniem do ręcznego pobrania, np.:

   ```
   # Linux / macOS
   curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   # Windows: w PowerShellu samo `curl` to Invoke-WebRequest (bez -L), stąd curl.exe
   curl.exe -L -o models\hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   # albo
   Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -OutFile models\hand_landmarker.task
   ```

Dokładniejszy wariant pozy (`pose_landmarker_full.task`, ok. 0,2° zamiast 0,4°
rozrzutu na łokciu) nie jest w repozytorium: wpisz go w oba pola `arm.model_path`
i `arm.model_url` (adres jak wyżej, z `full` zamiast `lite`), a pobierze się sam
i zostanie lokalny (`.gitignore` przepuszcza tylko dwa pliki z tabeli).

Sprawdzenie sumy po pobraniu:

```
# Windows (PowerShell)
Get-FileHash models\hand_landmarker.task -Algorithm SHA256
# Linux / macOS
sha256sum models/*.task
```
