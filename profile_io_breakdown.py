# profile_io_breakdown.py
import time, math, argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torchaudio

clock = time.perf_counter

# === あなたのパス ===
AUDIO_CSV = "AudioCaps_csv/train.csv"
RIR_CSV   = "RIR_dataset/rir_catalog_train.csv"
AUDIO_ROOT= "AudioCaps_mp3"         # 例: AudioCaps_mp3/train/{audiocap_id}.mp3
FOA_SR = 48_000
IV_SR  = 16_000
MAX_SEC = 10.0
N_FFT = 400
HOP   = 100

def a_to_b_foa(a4: torch.Tensor):
    m0,m1,m2,m3=a4[0],a4[1],a4[2],a4[3]
    W=(m0+m1+m2+m3)/2; X=(m0+m1-m2-m3)/2; Y=(m0-m1+m2-m3)/2; Z=(m0-m1-m2+m3)/2
    return torch.stack([W,Y,Z,X],0)

def foa_to_iv(foa: torch.Tensor, n_fft=N_FFT, hop=HOP, eps=1e-6):
    win=torch.hann_window(n_fft, device=foa.device)
    spec=torch.stft(foa.view(4,-1), n_fft=n_fft, hop_length=hop,
                    window=win, center=True, return_complex=True).view(4, n_fft//2+1, -1)
    W,Y,Z,X=spec[0],spec[1],spec[2],spec[3]; conjW=W.conj()
    I_act=torch.stack([(conjW*Y).real,(conjW*Z).real,(conjW*X).real],0)
    I_rea=torch.stack([(conjW*Y).imag,(conjW*Z).imag,(conjW*X).imag],0)
    norm=torch.linalg.norm(I_act, dim=0, keepdim=True)
    I_act=torch.where(norm>eps, I_act/norm, I_act); I_rea=torch.where(norm>eps, I_rea/norm, I_rea)
    return I_act.float(), I_rea.float()  # [3,F,T] ×2

def load_dry_mp3(audiocap_id:int, split="train"):
    p = Path(AUDIO_ROOT)/split/f"{audiocap_id}.mp3"
    wav, sr = torchaudio.load(p)
    if sr != FOA_SR:
        wav = torchaudio.functional.resample(wav, sr, FOA_SR)
    T = int(FOA_SR*MAX_SEC)
    if wav.shape[-1] < T:
        rpt=math.ceil(T/wav.shape[-1]); wav=wav.repeat(1,rpt)[:,:T]
    else:
        wav=wav[:,:T]
    # A-format 4ch（dryが1chなら複製）
    if wav.size(0)==1: wav4 = wav.repeat(4,1)
    else:              wav4 = wav[:4]
    return wav4  # [4, T]

def conv_cpu(dry4: torch.Tensor, rir: torch.Tensor):
    wet = torchaudio.functional.fftconvolve(dry4, rir)
    return wet[..., :dry4.size(-1)]  # [4,T]

def conv_gpu(dry4: torch.Tensor, rir: torch.Tensor, device="cuda"):
    # CUDAで周波数畳み込み（RIRのrfftは都度; 実運用はキャッシュ推奨）
    T = dry4.size(-1)
    n = 1 << (T-1).bit_length()  # 次の2の冪
    X = torch.fft.rfft(dry4.to(device, non_blocking=True), n=n)
    H = torch.fft.rfft(rir.to(device, non_blocking=True), n=n)
    Y = X * H
    y = torch.fft.irfft(Y, n=n)[..., :T].to("cpu")
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=50, help="計測に使うdry数")
    ap.add_argument("--split", default="train")
    ap.add_argument("--gpu_preproc", action="store_true", help="畳み込みとSTFTをGPUで実行（参考値）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    df_a = pd.read_csv(AUDIO_CSV)
    df_r = pd.read_csv(RIR_CSV)
    rirs = df_r["rir_path"].tolist()

    # ランダムに dry を抽出
    aids = [int(x) for x in df_a["audiocap_id"].tolist()]
    rng.shuffle(aids); aids = aids[:args.samples]

    t_load = []; t_conv = []; t_iv = []
    use_gpu = args.gpu_preproc and torch.cuda.is_available()

    for aid in aids:
        rp = rng.choice(rirs)
        rir, _ = torchaudio.load(rp)

        # 1) ロード（mp3→tensor + resample + pad/trim）
        t0 = clock()
        dry4 = load_dry_mp3(aid, split=args.split)  # [4,T] @48k
        t1 = clock()

        # 2) 畳み込み（A→B変換含む）
        if use_gpu:
            wet4 = conv_gpu(dry4, rir, device="cuda")
        else:
            wet4 = conv_cpu(dry4, rir)
        foa = a_to_b_foa(wet4)              # [4,T] @48k
        t2 = clock()

        # 3) 16kへdownsample + STFT→I_act/I_rea
        foa_16k = torchaudio.functional.resample(foa, FOA_SR, IV_SR)
        _i_act, _i_rea = foa_to_iv(foa_16k.to("cuda" if use_gpu else "cpu"))
        if use_gpu and torch.cuda.is_available(): torch.cuda.synchronize()
        t3 = clock()

        t_load.append(t1 - t0)
        t_conv.append(t2 - t1)
        t_iv.append(t3 - t2)

    def avg(x): return (sum(x)/len(x)) if x else float("nan")
    print(f"[{args.samples} samples] gpu_preproc={use_gpu}")
    print(f"  mp3->tensor(+resample): {avg(t_load):.4f} s/sample")
    print(f"  convolve(+A->B FOA):    {avg(t_conv):.4f} s/sample")
    print(f"  16k+STFT (i_act/rea):   {avg(t_iv):   .4f} s/sample")
    print(f"  TOTAL per sample:       {avg([a+b+c for a,b,c in zip(t_load,t_conv,t_iv)]):.4f} s/sample")

if __name__ == "__main__":
    main()
