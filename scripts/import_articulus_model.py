#!/usr/bin/env python3
"""Import prawdziwego zlozenia SO-101 z eksportu Articulusa do podgladu 3D.

Zrodlem geometrii jest repozytorium `przemeknowak781/articulus`, ktore odtwarza
SO-101 z brył STEP producenta (TheRobotStudio/SO-ARM100) i z jego URDF.
Eksport `articulus web` daje trzy rzeczy, ktorych potrzebuje podglad:

* `chain`  - lancuch kinematyczny jako kroki `pre @ ruch @ post`,
* `meshes` - siatki wszystkich czlonow w ukladach lokalnych czesci,
* `reference` - dokladne transformacje czlonow dla nazwanych poz, dzieki
  ktorym mozna *sprawdzic liczbowo*, czy nasza kinematyka zgadza sie
  z kinematyka Articulusa, zamiast zakladac, ze sie zgadza.

Pelny eksport ma ~11 MB siatek (535 tys. trojkatow) - o dwa rzedy wielkosci
za duzo na podglad rysowany programowo w petli sterowania. Ten skrypt
upraszcza siatki przez *grupowanie wierzcholkow* (vertex clustering): wierzcholki
sa zaokraglane do siatki przestrzennej o zadanym boku, powtorzenia scalane,
a trojkaty zdegenerowane usuwane. Sylwetka i proporcje zostaja, liczba
trojkatow spada kilkudziesieciokrotnie.

Uzycie:
    python scripts/import_articulus_model.py \
        --articulus /sciezka/do/articulus \
        --out assets/so101_preview.npz

Skrypt uruchamia `articulus web` samodzielnie (potrzebny `build123d`), albo
korzysta z gotowego eksportu przez `--export-dir`.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

#: Kolory czlonow (RGB 0-255) wg materialow z URDF producenta:
#: czesci drukowane sa zolte, serwa STS3215 czarne.
COLOR_PRINTED = (232, 196, 58)
COLOR_VENDOR = (48, 48, 52)

#: Bok siatki grupowania wierzcholkow [mm]. 2 mm daje ~8 tys. trojkatow
#: dla calego ramienia - plynnie renderuje sie programowo, a ksztalt
#: kazdej czesci pozostaje rozpoznawalny.
DEFAULT_CLUSTER_MM = 2.0


def run_articulus_export(articulus_root: Path, out_dir: Path) -> Path:
    """Uruchamia `articulus web` w sklonowanym repozytorium Articulusa."""
    robot_dir = articulus_root / "robots" / "so101-follower"
    if not robot_dir.is_dir():
        raise SystemExit(f"Nie znalazlem {robot_dir} - czy to na pewno repo Articulus?")

    env_src = articulus_root / "src"
    cmd = [
        sys.executable,
        "-m",
        "articulus.cli",
        "web",
        str(robot_dir),
        "--no-verify",
        "--out",
        str(out_dir),
    ]
    print(f"[1/4] articulus web {robot_dir.name} ...")
    result = subprocess.run(
        cmd,
        cwd=articulus_root,
        env={**_env(), "PYTHONPATH": str(env_src)},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            "Eksport z Articulusa nie powiodl sie:\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}\n"
            "Czy zainstalowano build123d?  pip install 'build123d>=0.11,<0.12'"
        )
    return out_dir / "so101-follower"


def _env() -> dict[str, str]:
    import os

    return dict(os.environ)


def load_export(export_dir: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """Wczytuje `robot.json` i `mesh.bin` (float32 XYZ + uint32 indeksy)."""
    model = json.loads((export_dir / "robot.json").read_text(encoding="utf-8"))
    blob = (export_dir / "mesh.bin").read_bytes()

    vertex_count, index_count = struct.unpack_from("<II", blob, 0)
    v_start = 8
    i_start = v_start + vertex_count * 3 * 4
    vertices = np.frombuffer(blob, dtype="<f4", count=vertex_count * 3, offset=v_start)
    indices = np.frombuffer(blob, dtype="<u4", count=index_count, offset=i_start)
    return model, vertices.reshape(-1, 3), indices.reshape(-1, 3)


def cluster_decimate(
    vertices: np.ndarray, faces: np.ndarray, cell_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Upraszcza siatke przez grupowanie wierzcholkow w siatce przestrzennej.

    Kazdy wierzcholek trafia do komorki o boku `cell_mm`; wszystkie wierzcholki
    jednej komorki zastepuje ich srednia. Trojkaty, ktorych dwa wierzcholki
    wpadly do tej samej komorki, znikaja (maja zerowe pole).
    """
    if len(vertices) == 0 or len(faces) == 0:
        return vertices, faces

    keys = np.floor(vertices / cell_mm).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)

    # Reprezentantem komorki jest srednia jej wierzcholkow - lagodniej niz
    # branie pierwszego z brzegu, bo nie przesuwa powierzchni o pol komorki.
    merged = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(merged, inverse, vertices)
    merged /= counts[:, None]

    new_faces = inverse[faces]
    degenerate = (
        (new_faces[:, 0] == new_faces[:, 1])
        | (new_faces[:, 1] == new_faces[:, 2])
        | (new_faces[:, 0] == new_faces[:, 2])
    )
    new_faces = new_faces[~degenerate]

    # Usuwamy wierzcholki, do ktorych nie odwoluje sie zaden trojkat.
    used, remap = np.unique(new_faces, return_inverse=True)
    return merged[used].astype(np.float32), remap.reshape(-1, 3).astype(np.int32)


def build_asset(model: dict, vertices: np.ndarray, indices: np.ndarray, cell_mm: float) -> dict:
    """Sklada zawartosc pliku .npz z eksportu Articulusa."""
    mesh_header = model["meshes"]["links"]
    roles = {link["id"]: link.get("role", "printed") for link in model["links"]}

    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    face_link: list[np.ndarray] = []
    link_names: list[str] = []
    colors: list[tuple[int, int, int]] = []
    vertex_base = 0
    before = after = 0

    print(f"[3/4] upraszczam siatki (siatka {cell_mm} mm) ...")
    for link_id in sorted(mesh_header):
        head = mesh_header[link_id]
        v = vertices[head["vertexOffset"] : head["vertexOffset"] + head["vertexCount"]]
        f = indices[head["indexOffset"] // 3 : (head["indexOffset"] + head["indexCount"]) // 3]
        # Indeksy w blobie sa LOKALNE dla czlonu (0..vertexCount-1), a nie
        # globalne - `vertexOffset` sluzy tylko do wycięcia wierzcholkow.
        f = f.astype(np.int64)
        if f.size and f.max() >= len(v):
            raise SystemExit(
                f"Czlon {link_id}: indeks {f.max()} poza zakresem {len(v)} wierzcholkow "
                "- format eksportu Articulusa sie zmienil."
            )

        before += len(f)
        v, f = cluster_decimate(v, f, cell_mm)
        after += len(f)
        if len(f) == 0:
            continue

        index = len(link_names)
        link_names.append(link_id)
        colors.append(COLOR_VENDOR if roles.get(link_id) == "vendor" else COLOR_PRINTED)

        all_v.append(v)
        all_f.append(f + vertex_base)
        face_link.append(np.full(len(f), index, dtype=np.int16))
        vertex_base += len(v)

    print(f"      trojkaty: {before} -> {after}  ({100 * after / max(before, 1):.1f}%)")

    # Milimetry -> metry: reszta aplikacji liczy wszystko w metrach.
    verts_m = np.concatenate(all_v).astype(np.float32) / 1000.0
    faces = np.concatenate(all_f).astype(np.int32)
    face_link_arr = np.concatenate(face_link)

    name_to_index = {name: i for i, name in enumerate(link_names)}
    chain = [step for step in model["chain"] if step["link"] in name_to_index]

    chain_link = np.array([name_to_index[s["link"]] for s in chain], dtype=np.int16)
    chain_parent = np.array(
        [name_to_index.get(s.get("parent"), -1) if s.get("parent") else -1 for s in chain],
        dtype=np.int16,
    )
    chain_joint = np.array([s.get("joint", "") for s in chain], dtype=object)
    chain_axis = np.array([s.get("axis", [0.0, 0.0, 1.0]) for s in chain], dtype=np.float32)
    chain_pre = np.array([_matrix(s.get("pre")) for s in chain], dtype=np.float32)
    chain_post = np.array([_matrix(s.get("post")) for s in chain], dtype=np.float32)
    chain_kind = np.array([s.get("kind", "fixed") for s in chain], dtype=object)

    dofs = model["dofs"]
    return {
        "link_names": np.array(link_names, dtype=object),
        "link_colors": np.array(colors, dtype=np.uint8),
        "vertices": verts_m,
        "faces": faces,
        "face_link": face_link_arr,
        "chain_link": chain_link,
        "chain_parent": chain_parent,
        "chain_joint": chain_joint,
        "chain_kind": chain_kind,
        "chain_axis": chain_axis,
        "chain_pre": chain_pre,
        "chain_post": chain_post,
        "dof_names": np.array([d["id"] for d in dofs], dtype=object),
        "dof_min": np.array([d["min"] for d in dofs], dtype=np.float32),
        "dof_max": np.array([d["max"] for d in dofs], dtype=np.float32),
        "source": np.array(
            json.dumps(
                {
                    "robot": model.get("robot"),
                    "title": model.get("title"),
                    "schema": model.get("schema"),
                    "cluster_mm": cell_mm,
                    "origin": "przemeknowak781/articulus -> vendor/lerobot-so101 (TheRobotStudio/SO-ARM100)",
                },
                ensure_ascii=False,
            ),
            dtype=object,
        ),
        "reference": np.array(json.dumps(model.get("reference", {})), dtype=object),
    }


def _matrix(values: list[float] | None) -> np.ndarray:
    """Zamienia 16 liczb (wiersz po wierszu) na macierz 4x4; None -> jednostkowa.

    Przesuniecia sa w milimetrach - przeliczamy je na metry od razu tutaj.
    """
    if not values:
        return np.eye(4, dtype=np.float32)
    matrix = np.array(values, dtype=np.float64).reshape(4, 4)
    matrix[:3, 3] /= 1000.0
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--articulus",
        type=Path,
        default=Path("../articulus"),
        help="katalog z klonem repozytorium Articulus",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="gotowy eksport `articulus web` (pomija ponowne generowanie)",
    )
    parser.add_argument("--out", type=Path, default=Path("assets/so101_preview.npz"))
    parser.add_argument("--cluster-mm", type=float, default=DEFAULT_CLUSTER_MM)
    args = parser.parse_args()

    if args.export_dir:
        export_dir = args.export_dir
        print(f"[1/4] uzywam gotowego eksportu: {export_dir}")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            export_dir = run_articulus_export(args.articulus.resolve(), Path(tmp))
            print("[2/4] wczytuje eksport ...")
            model, vertices, indices = load_export(export_dir)
            asset = build_asset(model, vertices, indices, args.cluster_mm)
            _save(asset, args.out)
            return 0

    print("[2/4] wczytuje eksport ...")
    model, vertices, indices = load_export(export_dir)
    asset = build_asset(model, vertices, indices, args.cluster_mm)
    _save(asset, args.out)
    return 0


def _save(asset: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **asset)
    size_kb = out.stat().st_size / 1024
    print(
        f"[4/4] zapisano {out}  ({size_kb:.0f} kB, "
        f"{len(asset['faces'])} trojkatow, {len(asset['link_names'])} czlonow)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
