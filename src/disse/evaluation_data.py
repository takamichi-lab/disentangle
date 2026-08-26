"""Create released 96-source by 96-spatial-condition evaluation inputs.

The repository ships identifiers and synthetic-room metadata, but no
YouTube-derived waveforms. Users provide lawful local copies of the 96 dry
clips. RIRs are regenerated from the fixed geometry, and inference performs
the dry-audio/RIR convolution on demand so that tens of gigabytes of
intermediate FOA audio do not need to be stored.
"""

from __future__ import annotations

import ast
import csv
import math
import os
import random
from pathlib import Path
from typing import Mapping

import numpy as np

from .captions import augment_caption


DRY_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def _three_floats(value: str, field: str) -> tuple[float, float, float]:
    try:
        parsed = ast.literal_eval(value)
        result = tuple(float(item) for item in parsed)
    except (SyntaxError, ValueError, TypeError) as error:
        raise ValueError(f"Invalid {field}: {value!r}") from error
    if len(result) != 3:
        raise ValueError(f"{field} must contain three numbers: {value!r}")
    return result


def _rir_output_path(row: Mapping[str, str], output_dir: Path) -> Path:
    value = str(row.get("rir_path") or "").strip()
    if not value:
        raise ValueError("RIR catalog row is missing rir_path")
    return output_dir / Path(value).name


def _unrounded_source_position(
    row: Mapping[str, str], dimensions: tuple[float, float, float]
) -> np.ndarray:
    """Recover the position used for simulation before CSV rounding."""
    center = np.asarray(dimensions, dtype=np.float64) / 2.0
    distance = float(row["source_distance_m"])
    azimuth = math.radians(float(row["azimuth_deg"]))
    elevation = math.radians(float(row["elevation_deg"]))
    position = center + distance * np.asarray(
        (
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        )
    )
    recorded = _three_floats(row["source_pos_xyz"], "source_pos_xyz")
    rounded = tuple(round(float(value), 3) for value in position)
    if rounded != recorded:
        raise ValueError(
            f"Polar source metadata does not match source_pos_xyz: "
            f"calculated {rounded}, recorded {recorded}"
        )
    return position


def generate_fixed_rirs(
    catalog_path: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> list[Path]:
    """Generate the fixed tetrahedral A-format RIRs from released metadata."""
    try:
        import pyroomacoustics as pra
        import soundfile as sf
        from pyroomacoustics.directivities import CardioidFamily, DirectionVector
        from tqdm.auto import tqdm
    except ImportError as error:
        raise RuntimeError(
            "RIR generation requires `pip install -e '.[evaluation-data]'`"
        ) from error

    rows = _read_csv(catalog_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    radius = 0.05
    vertex = radius / math.sqrt(3.0)
    tetrahedron = np.asarray(
        (
            (vertex, vertex, vertex),
            (vertex, -vertex, -vertex),
            (-vertex, vertex, -vertex),
            (-vertex, -vertex, vertex),
        ),
        dtype=np.float64,
    ).T

    for row in tqdm(rows, desc="Generating fixed RIRs", unit="RIR"):
        output = _rir_output_path(row, output_dir)
        outputs.append(output)
        if output.is_file() and not force:
            continue

        dimensions = _three_floats(row["dims"], "dims")
        source_position = _unrounded_source_position(row, dimensions)
        sample_rate = int(float(row["fs"]))
        room = pra.ShoeBox(
            list(dimensions),
            fs=sample_rate,
            materials=pra.Material(float(row["alpha"])),
            max_order=10,
        )
        room.add_source(list(source_position))
        center = np.asarray(dimensions, dtype=np.float64) / 2.0
        directions = []
        for x, y, z in tetrahedron.T:
            azimuth = math.degrees(math.atan2(y, x)) % 360.0
            colatitude = math.degrees(math.acos(z / radius))
            directions.append(
                CardioidFamily(
                    orientation=DirectionVector(
                        azimuth=azimuth, colatitude=colatitude, degrees=True
                    ),
                    p=0.5,
                    gain=1.0,
                )
            )
        microphones = pra.MicrophoneArray(
            center.reshape(3, 1) + tetrahedron,
            fs=sample_rate,
            directivity=directions,
        )
        room.add_microphone_array(microphones)
        room.compute_rir()

        responses = [np.asarray(room.rir[index][0]) for index in range(4)]
        length = max(response.size for response in responses)
        a_format = np.stack(
            [np.pad(response, (0, length - response.size)) for response in responses],
            axis=1,
        ).astype(np.float32)
        # SoundFile's original WAV default was PCM-16; make it explicit here.
        sf.write(output, a_format, sample_rate, subtype="PCM_16")

    return outputs


def _find_dry_audio(root: Path, source_id: str) -> Path | None:
    roots = (root, root / "test", root / "val")
    for candidate_root in roots:
        for extension in DRY_EXTENSIONS:
            candidate = candidate_root / f"{source_id}{extension}"
            if candidate.is_file():
                return candidate
    return None


def _relative_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def make_evaluation_manifest(
    audio_catalog: str | Path,
    rir_catalog: str | Path,
    dry_root: str | Path,
    rir_root: str | Path,
    output_path: str | Path,
    *,
    seed: int = 42,
    check_files: bool = True,
) -> Path:
    """Write the released source-major 9,216-row evaluation manifest."""
    audio_rows = _read_csv(audio_catalog)
    rir_rows = sorted(
        _read_csv(rir_catalog), key=lambda row: str(row["rir_path"])
    )
    if len(audio_rows) != 96 or len(rir_rows) != 96:
        raise ValueError(
            f"The released evaluation needs 96 audio rows and 96 RIR rows; "
            f"received {len(audio_rows)} and {len(rir_rows)}"
        )

    dry_root = Path(dry_root)
    rir_root = Path(rir_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dry_paths: dict[str, Path] = {}
    missing: list[str] = []
    for row in audio_rows:
        source_id = str(row["audiocap_id"])
        found = _find_dry_audio(dry_root, source_id)
        if found is None:
            found = dry_root / f"{source_id}.wav"
            missing.append(str(found))
        dry_paths[source_id] = found

    rir_paths: dict[str, Path] = {}
    for row in rir_rows:
        spatial_id = Path(row["rir_path"]).stem
        path = _rir_output_path(row, rir_root)
        if not path.is_file():
            missing.append(str(path))
        rir_paths[spatial_id] = path

    if check_files and missing:
        preview = "\n".join(f"  - {path}" for path in missing[:20])
        suffix = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise FileNotFoundError(f"Evaluation inputs are missing:\n{preview}{suffix}")

    rng = random.Random(seed)
    fieldnames = ("dry_path", "rir_path", "text", "source_id", "spatial_id")
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for audio_row in audio_rows:
            source_id = str(audio_row["audiocap_id"])
            for rir_row in rir_rows:
                spatial_id = Path(rir_row["rir_path"]).stem
                writer.writerow(
                    {
                        "dry_path": _relative_path(
                            dry_paths[source_id], output_path.parent
                        ),
                        "rir_path": _relative_path(
                            rir_paths[spatial_id], output_path.parent
                        ),
                        "text": augment_caption(
                            str(audio_row["caption"]), rir_row, rng=rng
                        ),
                        "source_id": source_id,
                        "spatial_id": spatial_id,
                    }
                )
    return output_path


def fixed_grid_summary(
    audio_catalog: str | Path, rir_catalog: str | Path
) -> dict[str, int]:
    """Return small sanity-check counts for the released fixed catalogs."""
    audio_rows = _read_csv(audio_catalog)
    rir_rows = _read_csv(rir_catalog)
    return {
        "sources": len(audio_rows),
        "spatial_conditions": len(rir_rows),
        "pairs": len(audio_rows) * len(rir_rows),
    }
