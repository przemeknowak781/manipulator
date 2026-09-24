"""Gdzie leza dane aplikacji: modele, stanowisko blizniaka, polityki.

Domyslne sciezki (`models/*.task`, `workspace/twin.json`, `workspace/policies`)
przy uruchomieniu z klonu repozytorium licza sie od jego korzenia - tak samo
jak zasoby w `twin/robots.resolve_asset` - zeby `lerobot-twin ui` odpalone
z innego katalogu nie zaczynalo od pustego stanowiska i nie pobieralo modeli
drugi raz. Po zwyklej instalacji pakietu (bez klonu) zostaje katalog biezacy.

Sciezki podane przez uzytkownika (flagi, YAML) NIE przechodza przez te funkcje:
wzgledne licza sie od katalogu biezacego, jak w kazdym programie.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Korzen repozytorium: `src/lerobot_mp/paths.py` -> dwa poziomy nad `src`.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Zmienna srodowiskowa z plikiem YAML konfiguracji (patrz `config.load_config`).
CONFIG_ENV = "LEROBOT_MP_CONFIG"


def source_checkout() -> bool:
    """Czy pakiet dziala z klonu repozytorium (`pip install -e .` albo PYTHONPATH=src)?"""
    return (REPO_ROOT / "pyproject.toml").is_file() and (REPO_ROOT / "src" / "lerobot_mp").is_dir()


def data_path(rel: str | Path) -> Path:
    """Domyslna sciezka danych: w klonie wzgledem korzenia repozytorium, inaczej wzgledem CWD.

    Uruchomienie z korzenia repozytorium daje te sama sciezke wzgledna co
    dotad (`workspace/twin.json`), wiec komunikaty i zachowanie sie nie zmieniaja.
    """
    rel = Path(rel)
    if rel.is_absolute() or not source_checkout():
        return rel
    try:
        if Path.cwd().resolve() == REPO_ROOT:
            return rel
    except OSError:  # pragma: no cover - usuniety katalog biezacy
        pass
    return REPO_ROOT / rel


def config_from_env() -> Path | None:
    """Plik konfiguracji z `LEROBOT_MP_CONFIG` (pusty albo brak = brak pliku)."""
    value = os.environ.get(CONFIG_ENV, "").strip()
    return Path(value).expanduser() if value else None
