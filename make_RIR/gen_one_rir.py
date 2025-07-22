#gen_one_rir.py
import json, random, math, yaml, soundfile as sf, numpy as np
import pyroomacoustics as pra
from pyroomacoustics.directivities import CardioidFamily, DirectionVector
from pathlib import Path
from typing import Literal
import hashlib
# ───────────────── 設定ロード
cfg = yaml.safe_load(Path("spatial_ranges.yml").read_text())

# ──────────────────── 量子化グリッド & ヘルパ ──────────────────────
# 距離: 1 cm (=0.01 m) 単位  /  角度: 1° 単位
GRID_CM  = 1       # [cm]
GRID_DEG = 1        # [deg]

POOL = {
    "train": json.loads(Path("room_pool_trainval.json").read_text()),
    "val"  : json.loads(Path("room_pool_trainval.json").read_text()),
    "test" : json.loads(Path("room_pool_test.json").read_text()),
}

def _rand_room(split):
    return random.choice(POOL[split])
def _snap(val: float, grid: float) -> float:
    """val を grid 単位に丸めて返す"""
    return round(val / grid) * grid

# ──────────────────────── 位置サンプリング ─────────────────────────
def _rand_position(
    split    : str,
    ctr      : np.ndarray,
    dist_rng : tuple[float,float] | None = None,
    el_rng   : tuple[float,float] | None = None,
    dims     : tuple[float,float,float] | None = None,   # ← 追加
    margin   : float = 0.1                               # ← 追加
) -> tuple[np.ndarray,float,float,float]:
    """
    壁マージンに抵触した場合は while ループを continue して
    “距離・方位・仰角の量子化セル” を変えずに **再サンプリング** する版。
    """
    if dist_rng is None or el_rng is None:
        rng_cfg = cfg["TEST"] if split == "test" else cfg["TRAINVAL"]
        dist_min, dist_max = rng_cfg["DIST_MIN"], rng_cfg["DIST_MAX"]
        el_min,   el_max   = rng_cfg["EL_MIN"],   rng_cfg["EL_MAX"]
    else:
        dist_min, dist_max = dist_rng
        el_min,   el_max   = el_rng

    if dims is None:
        raise ValueError("room dimensions `dims` must be given")

    w, h, H = dims
    while True:
        # ─── ① 連続乱数を引く
        dist   = random.uniform(dist_min, dist_max)
        az_deg = random.uniform(-180.0, 180.0)
        el_deg = random.uniform(el_min, el_max)

        # ─── ② 量子化（セル偶奇チェックは従来どおり）
        dist_q = _snap(dist, GRID_CM / 100)
        az_q   = _snap(az_deg, GRID_DEG)
        el_q   = _snap(el_deg, GRID_DEG)

        cell_d = int(round(dist_q / (GRID_CM / 100)))
        cell_a = int(round((az_q + 180.0) / GRID_DEG))
        cell_e = int(round((el_q - el_min) / GRID_DEG))
        if ((cell_d + cell_a + cell_e) & 1) != (1 if split == "test" else 0):
            continue

        # ─── ③ 極座標 → 直交座標
        az_rad = math.radians(az_q)
        el_rad = math.radians(el_q)
        src = ctr + dist_q * np.array([
            math.cos(el_rad)*math.cos(az_rad),
            math.cos(el_rad)*math.sin(az_rad),
            math.sin(el_rad)
        ])

        # ─── ④ マージン判定。壁を跨ぐなら「やり直し」
        if ((src[0] < margin) or (src[0] > w-margin) or
            (src[1] < margin) or (src[1] > h-margin) or
            (src[2] < margin) or (src[2] > H-margin)):
            continue   # ← ここでループ先頭に戻る

        # 条件クリア
        return src, dist_q, az_q, el_q

# ──────────────────────── RIR 生成 ─────────────────────────

def gen_one_rir(
    base_dir: Path, 
    id: str,
    split: Literal["train","val","test"],
    room_conf=None
    ):
    fs = 48000
    room_cfg = room_conf if room_conf else _rand_room(split)
    w,h,H = room_cfg["dims"]; alpha = room_cfg["alpha"]
    room = pra.ShoeBox([w,h,H], fs=fs,
                       materials=pra.Material(alpha), max_order=10)
    ctr = np.array([w/2,h/2,H/2])

    rng = cfg["TEST"] if split=="test" else cfg["TRAINVAL"]
    src, dist, az_deg, el_deg = _rand_position(
        split, ctr, (rng["DIST_MIN"], rng["DIST_MAX"]),
        (rng["EL_MIN"],   rng["EL_MAX"]),dims=(w,h,H)
    )

    # ─── ソース位置を必ず部屋の内側に収める ──────────

    room.add_source(src.tolist())
    # 正四面体マイク
    r=0.05; v=r/math.sqrt(3)
    tet = np.array([[ v,  v,  v],           # LFU
                [ v, -v, -v],           # RFD
                [-v,  v, -v],           # RBU
                [-v, -v,  v]], dtype=float).T    # LBD   (shape = 3×4)

    dirs = []
    for x, y, z in tet.T:                   # 列ごとに取り出す
        az  = math.degrees(math.atan2(y, x)) % 360            # 0–360°
        col = math.degrees(math.acos(z / r))                  # 0°=真上, 180°=真下
        dirs.append(
            CardioidFamily(
                orientation=DirectionVector(azimuth=az, colatitude=col, degrees=True),
                p=0.5,
                gain=1.0,
            )
        )
        # ---------- マイクロホンアレイを作成（指向性付き） ----------
    pos_mat = ctr.reshape(3,1) + tet  # 形状は (3, 4)
    mic_array = pra.MicrophoneArray(
        pos_mat,   # 3×4
        fs=fs,
        directivity=dirs,                    # 4 個の Directivity オブジェクト
    )
    room.add_microphone_array(mic_array)
    room.compute_rir()
    rir_4ch = [room.rir[m][0] for m in range(4)]  # 4ch RIR
    Tmax = max(len(o) for o in rir_4ch)
    rir_4ch = [np.pad(o, (0, Tmax - len(o))) for o in rir_4ch] # 4ch RIR をパディングして揃える
    # 4ch RIR をスタックして [
    rir_4ch = np.stack(rir_4ch, axis=1).astype(np.float32)
    data_dir = base_dir/"RIR_data"/f'{split}'
    meta_dir = base_dir/"RIR_meta"/f'{split}'
    # ─── ディレクトリ作成 ──────────
    data_dir.mkdir(exist_ok=True, parents=True)
    meta_dir.mkdir(exist_ok=True, parents=True)

    wav_path = data_dir / f"{id}.wav"
    meta_path = meta_dir / f"{id}.yml"
    sf.write(wav_path, rir_4ch.astype(np.float32), fs)
    # ─── メタデータを保存 ──────────
    room_id = hashlib.md5(json.dumps(room_cfg, sort_keys=True).encode()).hexdigest()[:8]
    meta = dict(
        dims           = room_cfg["dims"],             # [w,h,H]
        area_m2        = room_cfg["area_m2"],
        alpha          = float(alpha),
        fullband_T30_ms= room_cfg["T30_ms"],
        source_distance_m = round(dist, 3),
        azimuth_deg    = round(az_deg, 2),
        elevation_deg  = round(el_deg, 2),
        source_pos_xyz = [round(float(v), 3) for v in src],
        fs             = fs,
        room_id        = room_id,
        split          = split
    )
    meta_path.write_text(yaml.dump(meta, sort_keys=False, allow_unicode=True))
    print(f"RIR {id} generated: {wav_path} and {meta_path}")

    

