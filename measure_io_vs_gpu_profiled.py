import time, torch
from pathlib import Path
from torch.utils.data import DataLoader
from collections import defaultdict

from dataset.audio_rir_dataset_profiled import AudioRIRDatasetProfiled, collate_fn_profiled
from dataset.precomputed_val_dataset_profiled import PrecomputedValDatasetProfiled
from model.delsa_model import DELSA

clock = time.perf_counter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 設定 ======
audio_csv_train = "AudioCaps_csv/train.csv"
rir_csv_train   = "RIR_dataset/rir_catalog_train.csv"
audio_base      = "AudioCaps_mp3"
val_precomp_root= "Spatial_AudioCaps/.../for_delsa_spatialAudio"
val_index_csv   = f"{val_precomp_root}/val_precomputed.csv"
rir_csv_val     = "val_fixed_400x24/rir_fixed.csv"

batch_size=8; n_views=2; val_batch=64; num_workers=8; n_batches=5

def measure_io_time(loader,n_batches:int):
    times=[]; n_samples=0; it=iter(loader)
    for _ in range(n_batches):
        batch=next(it)
        if batch is None: continue
        t0=clock(); t1=clock()
        times.append(t1-t0); n_samples+=len(batch["texts"])
    return sum(times)/len(times), (sum(times)/n_samples if n_samples else float("nan"))

def _sum_prof_in_batch(batch):
    sums=defaultdict(float); cnt=0
    for d in batch.get("_prof_list",[]):
        cnt+=1
        for k,v in d.items(): sums[k]+=float(v)
    return sums,cnt

def measure_io_breakdown(loader,n_batches:int):
    sums=defaultdict(float); n_samples=0; it=iter(loader)
    for _ in range(n_batches):
        batch=next(it); 
        if batch is None: continue
        part,ns=_sum_prof_in_batch(batch)
        for k,v in part.items(): sums[k]+=v
        n_samples+=ns
    if n_samples==0: return {}
    return {k:{"total_s":tot,"per_sample_s":tot/n_samples} for k,tot in sums.items()}

def make_train_loader():
    ds=AudioRIRDatasetProfiled(audio_csv_train,audio_base,rir_csv_train,
        n_views=n_views,split="train",batch_size=batch_size,profile_io=True)
    return DataLoader(ds,batch_size=batch_size,shuffle=True,
        num_workers=num_workers,collate_fn=collate_fn_profiled,
        pin_memory=True,persistent_workers=True)

def make_val_loader():
    ds=PrecomputedValDatasetProfiled(index_csv=val_index_csv,rir_meta_csv=rir_csv_val,
        root=val_precomp_root,profile_io=True)
    return DataLoader(ds,batch_size=val_batch,shuffle=False,
        num_workers=num_workers,collate_fn=collate_fn_profiled,
        pin_memory=True,persistent_workers=True)

def gpu_forward_time_from_batch(model,batch,iters=3):
    audio={k: torch.stack([d[k] for d in batch["audio"]]).to(device) for k in ["i_act","i_rea","omni_48k"]}
    texts=batch["texts"]; _=model(audio,texts); torch.cuda.synchronize()
    ts=[]
    for _ in range(iters):
        t0=clock(); _=model(audio,texts); torch.cuda.synchronize(); ts.append(clock()-t0)
    return sum(ts)/len(ts)

def main():
    print("== TRAIN I/O ==")
    tr_dl=make_train_loader(); io_b_tr,io_s_tr=measure_io_time(tr_dl,n_batches)
    print(f"[TRAIN] {io_b_tr:.6f} s/batch ({io_s_tr:.6f} s/sample)")
    bd_tr=measure_io_breakdown(tr_dl,n_batches)
    for k,d in bd_tr.items():
        print(f"  - {k:14s}: {d['per_sample_s']:.6f} s/sample (tot {d['total_s']:.3f})")

    print("\n== VAL I/O ==")
    val_dl=make_val_loader(); io_b_val,io_s_val=measure_io_time(val_dl,n_batches)
    print(f"[VAL] {io_b_val:.6f} s/batch ({io_s_val:.6f} s/sample)")
    bd_val=measure_io_breakdown(val_dl,n_batches)
    for k,d in bd_val.items():
        print(f"  - {k:14s}: {d['per_sample_s']:.6f} s/sample (tot {d['total_s']:.3f})")

    print("\n== GPU forward ==")
    model=DELSA(audio_encoder_cfg={},text_encoder_cfg={}).to(device).eval()
    t_gpu_tr=gpu_forward_time_from_batch(model,next(iter(make_train_loader())))
    print(f"[GPU/TRAIN] {t_gpu_tr:.6f} s/batch")
    t_gpu_val=gpu_forward_time_from_batch(model,next(iter(make_val_loader())))
    print(f"[GPU/VAL]   {t_gpu_val:.6f} s/batch")

if __name__=="__main__": main()
