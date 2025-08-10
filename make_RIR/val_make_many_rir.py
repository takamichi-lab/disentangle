# val_make_many_rir_fast.py  ← 好きな名前で保存
from pathlib import Path
import random, yaml, tqdm
import numpy as np                     # ★ 追加
from gen_one_rir import gen_one_rir, _rand_room, _rand_position

# ───── 1. マッピング関数 ─────
def map_direction(az):
    if -35 <= az <= 35:   return 'front'
    if 55  <= az <= 125:  return 'right'
    if -125<= az <= -55:  return 'left'
    return 'back' if az >= 145 or az <= -145 else ''

def map_elevation(el): return 'up'   if el > 40  else 'down' if el < -40 else ''
def map_distance(d):   return 'near' if d < 1    else 'far'  if d > 2   else ''
def map_size(a):       return 'small' if a < 50  else 'large' if a > 100 else 'mid-sized'
def map_reverb(t30):   return 'acoustically dampened' if t30 < 200 else \
                        'highly reverberant' if t30 > 1000 else ''

# ───── 2. 目標 96 コンボ ─────
DIR  = ['front','right','left','back']
ELEV = ['up','down']; DIST = ['near','far']
SIZE = ['small','mid-sized','large']
REV  = ['acoustically dampened','highly reverberant']
TARGET = {(d,e,di,s,r) for d in DIR for e in ELEV for di in DIST for s in SIZE for r in REV}

# ───── 3. 既存メタ確認 ─────
def existing_combos(meta_dir: Path):
    combos = set()
    for yml in meta_dir.glob("*.yml"):
        m = yaml.safe_load(yml.read_text())
        key = (
            map_direction(m["azimuth_deg"]),
            map_elevation(m["elevation_deg"]),
            map_distance(m["source_distance_m"]),
            map_size(m["area_m2"]),
            map_reverb(m["fullband_T30_ms"])
        )
        if '' not in key:
            combos.add(key)
    return combos

# ───── 4. メイン ─────
def main():
    BASE  = Path("RIR_dataset")
    split = "val"
    random.seed(42)

    data_dir = BASE / "RIR_data" / split
    meta_dir = BASE / "RIR_meta" / split
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    combos = existing_combos(meta_dir)
    print(f"[info] Found {len(combos)} / 96 combos already present.")

    pbar = tqdm.tqdm(total=len(TARGET) - len(combos), desc="Generating RIRs")
    i = 0
    while combos != TARGET:
        rid   = f"auto_{i:06d}"

        # ---------- 前向きフィルタ (軽い判定) ----------
        rng_state = random.getstate()            # ★ 現在の乱数状態を保存
        room_cfg  = _rand_room(split)            # 部屋を 1 つサンプル
        w, h, H   = room_cfg["dims"]
        ctr       = np.array([w/2, h/2, H/2])
        # 位置だけサンプル（RIR 合成なし）
        _, dist, az_deg, el_deg = _rand_position(
            split, ctr, dims=(w, h, H))
        key = (
            map_direction(az_deg),
            map_elevation(el_deg),
            map_distance(dist),
            map_size(room_cfg["area_m2"]),
            map_reverb(room_cfg["T30_ms"])
        )
        # ① 既カバー or 曖昧なら捨てて次へ
        if '' in key or key in combos:
            i += 1
            continue
        # ② 未カバーなら乱数状態を戻して“同じ乱数列”で RIR 合成
        random.setstate(rng_state)               # ★ 復元
        gen_one_rir(base_dir=BASE, id=rid, split=split)
        combos.add(key)
        pbar.update(1)
        i += 1
    pbar.close()
    print(f"[✓] Completed 96 / 96 category combinations.")

if __name__ == "__main__":
    main()
