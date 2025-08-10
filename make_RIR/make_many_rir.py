# make_many_rirs.py
from pathlib import Path
import tqdm, random, json
from gen_one_rir import gen_one_rir, _rand_room

BASE = Path("RIR_dataset")
N_RIRS = 1000
split = "val"
BASE.mkdir(exist_ok=True)

# toDo:trainとvalで重ならないようにする。
# ──── 乱数シードの設定 ───────────────────────────────
random.seed(30)  # 固定シードで再現性を確保 trainのときは42
# ──── RIR 生成 ────────────────────────────────
for i in tqdm.trange(N_RIRS, desc="Gen RIRs"):
    rid = f"{i:05d}"
    room_conf = _rand_room(split)
    gen_one_rir(
        id=rid,
        base_dir=BASE,
        split=split,
        room_conf=room_conf
    )
print("Done")