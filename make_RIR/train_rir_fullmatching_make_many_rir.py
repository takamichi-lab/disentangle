#!/usr/bin/env python3
# 空カテゴリ('')も“有効”としてカウントし、各コンボあたり n 個ずつ生成する版
from pathlib import Path
import random, yaml, tqdm
import numpy as np
from collections import Counter
from itertools import product

from gen_one_rir import gen_one_rir, _rand_room, _rand_position, POOL  # 既存を利用
from create_csv_for_rir import build_catalog

# ───── 1) マッピング（'' を返すのは dir/elev/dst/rev。sizeは必ず3クラス） ─────
def map_direction(az):
    if -35 <= az <= 35:   return 'front'
    if 55  <= az <= 125:  return 'right'
    if -125<= az <= -55:  return 'left'
    if az >= 145 or az <= -145: return 'back'
    return ''  # ← 有効

def map_elevation(el): return 'up' if el > 40 else 'down' if el < -40 else ''
def map_distance(d):   return 'near' if d < 1 else 'far' if d > 2 else ''
def map_size(a):       return 'small' if a < 50 else 'large' if a > 100 else 'mid-sized'
def map_reverb(t30):   return 'acoustically dampened' if t30 < 200 else \
                        'highly reverberant' if t30 > 1000 else ''  # ← 有効

# ───── 2) 既存メタから“今あるコンボ”の件数を読む（''も数える） ─────
def existing_counts(meta_dir: Path):
    cnt = Counter()
    for yml in meta_dir.glob("*.yml"):
        m = yaml.safe_load(yml.read_text())
        key = (
            map_direction(float(m["azimuth_deg"])),
            map_elevation(float(m["elevation_deg"])),
            map_distance(float(m["source_distance_m"])),
            map_size(float(m["area_m2"])),
            map_reverb(float(m["fullband_T30_ms"]))
        )
        # size は必ず small/mid-sized/large のどれか
        cnt[key] += 1
    return cnt

# ───── 3) “出現し得るラベル”だけで TARGET を作る ─────
def build_allowed_sets(split: str):
    # 3-1) size/reverb は room_pool から実際に出るラベルだけ
    rooms = POOL[split]
    allowed_size = sorted({ map_size(rc["area_m2"]) for rc in rooms })
    allowed_revb = sorted({ map_reverb(rc["T30_ms"]) for rc in rooms })
    # size は '' を含まない/含めない（仕様）
    if '' in allowed_size:
        allowed_size.remove('')

    # 3-2) elev/distance は spatial_ranges.yml から“可能性”を判定
    cfg = yaml.safe_load(Path("spatial_ranges.yml").read_text())
    rng = cfg["TEST"] if split == "test" else cfg["TRAINVAL"]
    el_min, el_max = float(rng["EL_MIN"]), float(rng["EL_MAX"])
    dist_min, dist_max = float(rng["DIST_MIN"]), float(rng["DIST_MAX"])

    allowed_elev = set([''])  # 中間（-40〜40）が1つでも含まれ得るなら '' は常に可能
    if el_max > 40:  allowed_elev.add('up')
    if el_min < -40: allowed_elev.add('down')
    allowed_elev = sorted(allowed_elev)

    allowed_dist = set([''])  # 1〜2m に交差があれば '' があり得る
    if dist_min < 1: allowed_dist.add('near')
    if dist_max > 2: allowed_dist.add('far')
    allowed_dist = sorted(allowed_dist)

    # 3-3) direction は _rand_position が -180〜180 を使うので5クラスすべてあり得る
    allowed_dir = ['front','right','left','back','']

    # 3-4) reverb: room_pool によっては ''（200〜1000ms）が無いこともある点に注意
    allowed_revb = [lab for lab in allowed_revb]  # そのまま（'' が無ければ入らない）

    return allowed_dir, allowed_elev, allowed_dist, allowed_size, allowed_revb

def build_targets(per_combo: int, split: str):
    DIR, ELEV, DIST, SIZE, REV = build_allowed_sets(split)
    tgt = {}
    for key in product(DIR, ELEV, DIST, SIZE, REV):
        tgt[key] = per_combo
    return tgt  # dict: combo -> quota

# ───── 4) メイン ─────
def main():
    BASE  = Path("RIR_dataset")
    SPLIT = "train"      # "train"/"val"/"test"
    SEED  = 40
    PER_COMBO = 2      # ★ 各コンボにつき何個作るか
    MAX_ATTEMPTS = 1_000_0000

    random.seed(SEED)
    data_dir = BASE / "RIR_data" / SPLIT
    meta_dir = BASE / "RIR_meta" / SPLIT
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    target = build_targets(PER_COMBO, SPLIT)
    have   = existing_counts(meta_dir)

    # 既にある分を差し引き
    remaining = {k: max(0, target[k] - have.get(k, 0)) for k in target}
    total_rem = sum(remaining.values())
    print(f"[info] combos={len(target)}  per_combo={PER_COMBO}  remaining={total_rem}")
    if total_rem == 0:
        print("[✓] Nothing to do. Rebuilding CSV...")
        build_catalog(BASE, SPLIT)
        return

    # 既存ファイルと被らない連番ID
    serial = 0
    def next_id():
        nonlocal serial
        rid = f"auto_{serial:06d}"
        serial += 1
        return rid

    pbar = tqdm.tqdm(total=total_rem, desc="Generating RIRs")
    attempts = 0

    while total_rem > 0 and attempts < MAX_ATTEMPTS:
        attempts += 1

        # ---- 前向きフィルタ：部屋→位置のみ（軽量） ----
        rng_state = random.getstate()       # 復元用
        room_cfg  = _rand_room(SPLIT)       # 部屋1つ
        w, h, H   = room_cfg["dims"]
        ctr       = np.array([w/2, h/2, H/2])

        # 位置だけサンプル（RIR合成なし）
        _, dist, az_deg, el_deg = _rand_position(SPLIT, ctr, dims=(w, h, H))

        key = (
            map_direction(az_deg),
            map_elevation(el_deg),
            map_distance(dist),
            map_size(room_cfg["area_m2"]),
            map_reverb(room_cfg["T30_ms"]),
        )

        # この key がターゲット外（=あり得ない/不要）ならスキップ
        if key not in remaining or remaining[key] <= 0:
            continue

        # ---- 受理：乱数状態を戻して本生成（同じ乱数列を再現） ----
        random.setstate(rng_state)
        rid = next_id()
        gen_one_rir(base_dir=BASE, id=rid, split=SPLIT)  # ← gen_one_rir は無改変でOK

        remaining[key] -= 1
        total_rem      -= 1
        pbar.update(1)

    pbar.close()

    if total_rem > 0:
        # 物理的に出にくいコンボが残った場合の警告
        lacking = {k:v for k,v in remaining.items() if v>0}
        print("[!] Could not fill all quotas. Lacking combos:")
        for k, v in list(lacking.items())[:20]:
            print(f"    {k}: {v}")
        if len(lacking) > 20:
            print(f"    ... and {len(lacking)-20} more")

    print("Rebuilding CSV...")
    build_catalog(BASE, SPLIT)
    print("[✓] Done.")

if __name__ == "__main__":
    main()
