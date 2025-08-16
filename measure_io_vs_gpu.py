# measure_io_vs_gpu_v2.py
import time, torch
from pathlib import Path
from torch.utils.data import DataLoader
import torchaudio
from functools import lru_cache
import torchaudio
from functools import lru_cache
from dataset.audio_rir_dataset_old import foa_to_iv  # 既存のSTFT実装を再利用:contentReference[oaicite:3]{index=3}

# === あなたの実装（現行名） ===
from dataset.audio_rir_dataset import AudioRIRDataset, collate_fn as collate_train  # :contentReference[oaicite:4]{index=4}
from dataset.precomputed_val_dataset import PrecomputedValDataset                    # :contentReference[oaicite:5]{index=5}
from model.delsa_model import DELSA                                                 # :contentReference[oaicite:6]{index=6}

clock = time.perf_counter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 設定（config.yamlと揃えてOK） ======
audio_csv_train = "AudioCaps_csv/train.csv"
rir_csv_train   = "RIR_dataset/rir_catalog_train.csv"
audio_base      = "AudioCaps_mp3"

val_precomp_root = "Spatial_AudioCaps/takamichi09/for_delsa_spatialAudio"
val_index_csv    = "Spatial_AudioCaps/takamichi09/for_delsa_spatialAudio/val_precomputed.csv"
rir_csv_val      = "val_fixed_400x24/rir_fixed.csv"

batch_size  = 8          # train
n_views     = 8
val_batch   = 64         # val
num_workers = 8
n_batches   = 10         # それぞれ何バッチ計測

# --- FOA/IV のSR（現行実装と合わせる） ---
FOA_SR = 48_000
IV_SR  = 16_000

# ====== defer（dry+rir_path）用ヘルパ ======
def _aformat_to_foa(wet):  # wet: [4,T] A-format -> [4,T] FOA (W,Y,Z,X)
    m0, m1, m2, m3 = wet[0], wet[1], wet[2], wet[3]
    W = (m0+m1+m2+m3)/2; X = (m0+m1-m2-m3)/2
    Y = (m0-m1+m2-m3)/2; Z = (m0-m1-m2+m3)/2
    return torch.stack([W, Y, Z, X], dim=0)

@lru_cache(maxsize=4096)
def _load_rir_cpu_cached(path: str):
    rir, sr = torchaudio.load(path)  # [4, Tr]
    if sr != FOA_SR:
        rir = torchaudio.functional.resample(rir, sr, FOA_SR)
    return rir

def _build_audio_from_defer(audio_list, device):
    """
    Datasetが {dry, rir_path} を返したときに、GPUで
      畳み込み→A->B(FOA)→16k resample→(i_act,i_rea) まで作って
    DELSA が受け取る dict に整形して返す:contentReference[oaicite:7]{index=7}。
    """
    # dry: [B,1,T]
    drys = torch.stack([a["dry"].squeeze(0) for a in audio_list]).unsqueeze(1).to(device, non_blocking=True)
    T = drys.shape[-1]

    # rir（CPUキャッシュ → pad整形 → GPU）
    rirs_cpu = []
    Tr_list = []
    for a in audio_list:
        rir = _load_rir_cpu_cached(a["rir_path"])
        rirs_cpu.append(rir); Tr_list.append(rir.shape[-1])
    Tr_max = max(Tr_list)
    pad_rirs = []
    for rir in rirs_cpu:
        pad = Tr_max - rir.shape[-1]
        if pad > 0:
            rir = torch.nn.functional.pad(rir, (0, pad))  # → [4, Tr_max]
        pad_rirs.append(rir)
    rirs = torch.stack(pad_rirs).to(device, non_blocking=True)  # [B,4,Tr_max]

    # 共有 n_fft で周波数領域畳み込み
    n = T + Tr_max - 1
    n_fft = 1 << (n - 1).bit_length()
    D = torch.fft.rfft(drys, n_fft)
    R = torch.fft.rfft(rirs, n_fft)
    wet_a = torch.fft.irfft(D * R, n_fft)[..., :n]
    wet_a = wet_a[..., :T]  # [B,4,T]

    # A->B(FOA)
    foa = torch.stack([_aformat_to_foa(w) for w in wet_a], dim=0)  # [B,4,T]
    omni_48k = foa[:, 0, :]                                       # [B,T]

    # 16kへ（環境によりtorchaudio.resampleはCPU実装のことがあるので安全にCPUで）
    foa_16k = torchaudio.functional.resample(foa.detach().cpu(), orig_freq=FOA_SR, new_freq=IV_SR)

    # i_act, i_rea（既存の foa_to_iv をここで呼ばず、forward側と同条件に合わせたいなら
    # ここは省略して forward 内で作る実装に合わせてもOK）
    # 今回は計測簡略化のため、forward想定の「すでに I_act/I_rea を持っている」形で返す:
    # → ここではダミーでゼロを返し、GPU前処理の“畳み込み＋FOAまで”の時間を測る場合に使う。
    # （実運用ではあなたの foa_to_iv を import してここで計算してください）
    B, _, _ = foa_16k.shape
    # ダミーの (B,3,F,Tfrm) を返す（forwardの入力キーを満たすため）:contentReference[oaicite:8]{index=8}
    i_act = torch.zeros(B, 3, 201, 1, dtype=torch.float32)
    i_rea = torch.zeros(B, 3, 201, 1, dtype=torch.float32)

    return {
        "i_act": i_act.to(device, non_blocking=True),
        "i_rea": i_rea.to(device, non_blocking=True),
        "omni_48k": omni_48k,  # [B,T] on device
    }
def _is_defer_batch(batch):
    al = batch["audio"]
    return len(al)>0 and ("dry" in al[0] and "rir_path" in al[0])
# ====== 共通の計測ヘルパ ======
def measure_io_time(loader, n_batches: int):
    """DataLoaderから次バッチを取り出す時間（__getitem__+collateを含む）"""
    times = []
    n_samples = 0
    it = iter(loader)
    for _ in range(n_batches):
        t0 = clock()
        batch = next(it)
        t1 = clock()
        times.append(t1 - t0)
        # AudioRIRDatasetは n_views 分 texts が増える前提:contentReference[oaicite:9]{index=9}
        n_samples += len(batch["texts"])
    avg_b = sum(times) / len(times)
    avg_s = sum(times) / n_samples if n_samples else float("nan")
    return avg_b, avg_s

@torch.no_grad()
def gpu_forward_time_from_batch(model: torch.nn.Module, batch: dict, iters: int = 3):
    # defer かどうかを自動判定
    if _is_defer_batch(batch):
        audio = _build_audio_from_defer(batch["audio"], device)
    else:
        audio = {
            "i_act":    torch.stack([d["i_act"]    for d in batch["audio"]]).to(device, non_blocking=True),
            "i_rea":    torch.stack([d["i_rea"]    for d in batch["audio"]]).to(device, non_blocking=True),
            "omni_48k": torch.stack([d["omni_48k"] for d in batch["audio"]]).to(device, non_blocking=True),
        }
    texts = batch["texts"]

    # warmup
    _ = model(audio, texts)  # DELSAは上3キーを読む:contentReference[oaicite:4]{index=4}
    if device.type == "cuda": torch.cuda.synchronize()

    ts = []
    for _ in range(iters):
        t0 = clock()
        _ = model(audio, texts)
        if device.type == "cuda": torch.cuda.synchronize()
        t1 = clock()
        ts.append(t1 - t0)
    return sum(ts) / len(ts)

def make_train_loader():
    ds = AudioRIRDataset(
        csv_audio=audio_csv_train,
        base_dir=audio_base,
        csv_rir=rir_csv_train,
        n_views=n_views,
        split="train",
        batch_size=batch_size
    )
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_train,  # collateは現行のCPU版:contentReference[oaicite:11]{index=11}
        pin_memory=True, persistent_workers=(num_workers > 0)
    )
    return dl

def make_val_loader():
    if not Path(val_index_csv).exists():
        raise SystemExit(f"val_index_csv not found: {val_index_csv}")
    ds = PrecomputedValDataset(index_csv=val_index_csv, rir_meta_csv=rir_csv_val, root=val_precomp_root)  # :contentReference[oaicite:12]{index=12}
    dl = DataLoader(
        ds, batch_size=val_batch, shuffle=False,
        num_workers=num_workers, collate_fn=collate_train,
        pin_memory=True, persistent_workers=(num_workers > 0)
    )
    return dl

def is_defer_batch(batch):
    al = batch["audio"]
    return len(al) > 0 and ("dry" in al[0] and "rir_path" in al[0])

@torch.no_grad()
def gpu_preproc_time_from_defer(batch, iters=3):
    """deferバッチから GPU 畳み込み前処理（dry×RIR→FOAまで）時間を測る"""
    # warmup
    _ = _build_audio_from_defer(batch["audio"], device)
    if device.type == "cuda": torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = clock()
        _ = _build_audio_from_defer(batch["audio"], device)
        if device.type == "cuda": torch.cuda.synchronize()
        t1 = clock()
        ts.append(t1 - t0)
    return sum(ts) / len(ts)

def main():
    # === A) TRAIN I/O ===
    print("== Measuring I/O (AudioRIRDataset: load + RIR/foa_to_iv path as implemented) ==")
    train_dl = make_train_loader()
    io_b_tr, io_s_tr = measure_io_time(train_dl, n_batches)
    print(f"[TRAIN I/O] {io_b_tr:.6f} s/batch   ({io_s_tr:.6f} s/sample)  "
          f"bs={batch_size}, n_views={n_views}, workers={num_workers}")

    # === B) VAL I/O ===
    print("== Measuring I/O (PrecomputedValDataset: load .pt + FOA wav) ==")
    val_dl = make_val_loader()
    io_b_val, io_s_val = measure_io_time(val_dl, n_batches)
    print(f"[VAL I/O]   {io_b_val:.6f} s/batch   ({io_s_val:.6f} s/sample)  "
          f"bs={val_batch}, workers={num_workers}")

    # === C) GPU forward（実バッチで） ===
    print("== Measuring GPU forward with real batches ==")
    model = DELSA(audio_encoder_cfg={}, text_encoder_cfg={}).to(device).eval()  # 入力キーは i_act/i_rea/omni_48k:contentReference[oaicite:13]{index=13}

    # train-like
    train_batch = next(iter(make_train_loader()))
    t_gpu_train = gpu_forward_time_from_batch(model, train_batch, iters=3)
    print(f"[GPU fwd/TRAIN-like] {t_gpu_train:.6f} s/batch  (bs={batch_size}, n_views={n_views})")

    # val-like
    val_batch0 = next(iter(make_val_loader()))
    t_gpu_val = gpu_forward_time_from_batch(model, val_batch0, iters=3)
    print(f"[GPU fwd/VAL-like]   {t_gpu_val:.6f} s/batch  (bs={val_batch})")

    # === D) （任意）defer前処理のGPU時間 ===
    if is_defer_batch(train_batch):
        t_gpu_pre = gpu_preproc_time_from_defer(train_batch, iters=3)
        print(f"[GPU preproc (defer)] {t_gpu_pre:.6f} s/batch  (dry×RIR→FOA)")
    else:
        print("[GPU preproc] defer形式のバッチではなかったためスキップ")

    # === E) 速度比まとめ（I/O ÷ GPU） ===
    def ratio(a, b): return (float("inf") if b == 0 else a / b)
    print("\n=== Summary (I/O ÷ GPU, bigger = I/O-limited) ===")
    print(f"TRAIN: {ratio(io_b_tr,  t_gpu_train):.1f}×")
    print(f"VAL:   {ratio(io_b_val, t_gpu_val):.1f}×")

if __name__ == "__main__":
    main()
