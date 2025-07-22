# file: train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, distributed
from tqdm import tqdm
import yaml
import os
import wandb 
# Import your custom modules
from model.elsa_model import ELSA
from model.foa_dataset_caption import ELSADataset
from torch.utils.data import ConcatDataset
from model.audiocaps_dataset import AudioCapsDatasetForELSA
from helper_metrics.metrics import get_retrieval_ranks, calculate_recall_at_k,calculate_median_rank
import torch.nn.functional as F
import argparse
import torch
from torch.amp import autocast, GradScaler

scaler = GradScaler()  
from pathlib import Path
def init_dist_and_get_device():
    if torch.cuda.is_available():
        if 'RANK' in os.environ:
            # DDP 環境下（マルチ GPU）
            print("DDP used. Initializing process group.")
            torch.distributed.init_process_group(backend='nccl')
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            device = torch.device(f'cuda:{rank}')
        else:
            # 単一 GPU
            print("Single GPU detected. Running on cuda:0.")
            rank = 0
            world_size = 1
            device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    else:
        # CUDA 利用不可時は CPU
        print("CUDA not available. Running on CPU.")
        rank = 0
        world_size = 1
        device = torch.device('cpu')
    return rank, world_size, device

def validate_and_evaluate(model, dataloader, device, loss_weights, epoch, eval_name: str, is_main_process: bool):
    model.eval()
    total_loss = 0
    all_audio_embeds, all_text_embeds = [], []
    sum_L_dir_orig = sum_L_dist_orig = sum_L_area_orig = 0.0   
    spatial_batches = 0
    print(f"\n--- Running evaluation for '{eval_name}' (Epoch {epoch+1}) ---")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Eval: {eval_name}", disable=not is_main_process):
            if batch is None: continue
            audio_input = {k: v.to(device) for k, v in batch['audio'].items()}
            text_input = batch['text']
            batch_data = {k: v.to(device) for k, v in batch.items() if k not in ['audio', 'text']}
            logit_scale = model.module.logit_scale if hasattr(model, 'module') else model.logit_scale
            with torch.amp.autocast("cuda", dtype=torch.float16):
                model_outputs = model(audio_input, text_input)
            losses, loss_value = calculate_total_loss(model_outputs, batch_data, logit_scale, loss_weights, device)
            total_loss += loss_value.item()
            all_audio_embeds.append(model_outputs['audio_embed'].cpu())
            all_text_embeds.append(model_outputs['text_embed'].cpu())
         
            # ----------- FOA 用の元スケール損失計算 -----------
            if eval_name == "val_foa":
                spatial_mask = batch_data["has_spatial"]
                if spatial_mask.sum() > 0:
                    # 統計量を取得（Dataset 経由 or stats.pt）
                    if hasattr(dataloader.dataset, "area_mean"):
                        am, ast = dataloader.dataset.area_mean, dataloader.dataset.area_std
                        dm, dst = dataloader.dataset.dist_mean, dataloader.dataset.dist_std
           
                    # 正規化値 → 元スケールへ
                    pd_norm = model_outputs["pred_dist"][spatial_mask].squeeze()
                    td_norm = batch_data["distance"][spatial_mask]
                    pa_norm = model_outputs["pred_area"][spatial_mask].squeeze()
                    ta_norm = batch_data["area"][spatial_mask]

                    pd_orig = pd_norm * dst + dm
                    td_orig = td_norm * dst + dm
                    pa_orig = pa_norm * ast + am
                    ta_orig = ta_norm * ast + am

                    pred_dir = model_outputs["pred_dir"][spatial_mask]
                    true_dir = batch_data["direction"][spatial_mask]

                    # 元スケール損失
                    L_dir_orig  = (-1 * F.cosine_similarity(pred_dir, true_dir, dim=1)).mean()
                    L_dist_orig = F.mse_loss(pd_orig, td_orig)
                    L_area_orig = F.mse_loss(pa_orig, ta_orig)
                    # もし正規化してないデータで訓練した場合は以下の行を有効にする
                    # 正規化していないときモデルの生の出力を表すpd_normは実際の物理量である
                    # ただし、データローダーを介すtd_normは正規化されて小さい値となっているのでtd_originを使用
                    # L_dist_orig = F.mse_loss(pd_norm, td_orig) 正規化してないデータで訓練した場合はこれを使う
                    # L_area_orig = F.mse_loss(pa_norm, ta_orig) 正規化してないデータで訓練した場合はこれを使う

                    sum_L_dir_orig  += L_dir_orig.item()
                    sum_L_dist_orig += L_dist_orig.item()
                    sum_L_area_orig += L_area_orig.item()
                    spatial_batches += 1   
    if not is_main_process or not all_audio_embeds: return
    avg_loss = total_loss / len(dataloader)
    audio_embeds = torch.cat(all_audio_embeds, dim=0)
    text_embeds = torch.cat(all_text_embeds, dim=0)
    sim_matrix = F.normalize(text_embeds) @ F.normalize(audio_embeds).T
    N = audio_embeds.size(0)
    assert N == text_embeds.size(0), "audio / text でサンプル数が不一致です"
    assert sim_matrix.shape == (N, N), "sim_matrix が正方行列ではありません"

    # “同 index が真ペア” を確認（最大値までは要求しない）
    row_idx = torch.arange(N)
    assert torch.all(sim_matrix[row_idx, row_idx] == sim_matrix.diag()), "diag 取り出しが不一致"
    t2a_ranks, a2t_ranks = get_retrieval_ranks(sim_matrix)
    k_values = [1, 5, 10]
    t2a_recall = calculate_recall_at_k(t2a_ranks, k_values)
    a2t_recall = calculate_recall_at_k(a2t_ranks, k_values)
    # --- ▼▼▼ 中央値ランク(MedR)を計算 ▼▼▼ ---
    t2a_median_rank = calculate_median_rank(t2a_ranks)
    a2t_median_rank = calculate_median_rank(a2t_ranks)
    # --- ▲▲▲ 修正ここまで ▲▲▲ ---
    
    # wandbに記録
    log_metrics = {
        f"{eval_name}/avg_loss": avg_loss,
        f"{eval_name}/t2a_recall_at_1": t2a_recall.get('r@1', 0),
        f"{eval_name}/t2a_recall_at_5": t2a_recall.get('r@5', 0),
        f"{eval_name}/t2a_recall_at_10": t2a_recall.get('r@10', 0),
        f"{eval_name}/a2t_recall_at_1": a2t_recall.get('r@1', 0),
        f"{eval_name}/a2t_recall_at_5": a2t_recall.get('r@5', 0),
        f"{eval_name}/a2t_recall_at_10": a2t_recall.get('r@10', 0),
        # --- ▼▼▼ 中央値ランクをログに追加 ▼▼▼ ---
        f"{eval_name}/t2a_median_rank": t2a_median_rank,
        f"{eval_name}/a2t_median_rank": a2t_median_rank,
        f"{eval_name}/temperature": logit_scale.exp().item(), # 温度τ
        # --- ▲▲▲ 修正ここまで ▲▲▲ ---
        "epoch": epoch + 1
    }
    if eval_name == "val_foa" and spatial_batches > 0:
            log_metrics.update({
            f"{eval_name}/L_dir_orig":  sum_L_dir_orig  / spatial_batches,
            f"{eval_name}/L_dist_orig": sum_L_dist_orig / spatial_batches,
            f"{eval_name}/L_area_orig": sum_L_area_orig / spatial_batches,
        })

    if wandb.run is not None: wandb.log(log_metrics)

    # --- ▼▼▼ コンソール表示を更新 ▼▼▼ ---
    print(f"--- Results for '{eval_name}': Avg Loss={avg_loss:.4f}, "
          f"Recall@1(T2A)={t2a_recall.get('r@1', 0):.2f}%, MedR(T2A)={t2a_median_rank:.1f} ---")
    # --- ▲▲▲ 修正ここまで ▲▲▲ ---


# --- ▼▼▼ THIS IS THE CORRECTED FUNCTION ▼▼▼ ---
def collate_fn(batch):
    """
    データセットが出力する辞書の全てのキーを正しくバッチ化する関数。
    """
    # エラーでNoneが返されたサンプルを除外
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None

    # バッチの最初のアイテムからキーを取得し、キーごとにデータをまとめる
    keys = batch[0].keys()
    collated = {key: [d[key] for d in batch] for key in keys}

    # データを種類に応じて正しくバッチ化
    final_batch = {}
    for key, values in collated.items():
        if key == 'text':
            final_batch[key] = values  # テキストは文字列のリストのまま
        elif key == 'audio':
            # audioはネストした辞書なので、特別に処理
            final_batch[key] = {
                'i_act': torch.stack([d['i_act'] for d in values]),
                'i_rea': torch.stack([d['i_rea'] for d in values]),
                'omni_48k': torch.stack([d['omni_48k'] for d in values]),
            }
        else:
            # その他のデータ（direction, distance, area, has_spatialなど）はテンソルとしてスタック
            final_batch[key] = torch.stack(values)
            
    return final_batch
# --- ▲▲▲ END OF CORRECTED FUNCTION ▲▲▲ ---

def calculate_clip_loss(audio_emb, text_emb, logit_scale):
    a_norm = F.normalize(audio_emb)
    t_norm = F.normalize(text_emb)
    logits = (a_norm @ t_norm.T) * logit_scale.exp()
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

# --- Loss Function ---
def calculate_total_loss(model_outputs, batch_data, logit_scale, loss_weights, device):
    loss_clip = calculate_clip_loss(model_outputs['audio_embed'], model_outputs['text_embed'], logit_scale)
    
    spatial_mask = batch_data['has_spatial']
    num_spatial_samples = spatial_mask.sum()

    loss_dir, loss_dist, loss_area = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    if num_spatial_samples > 0:
        pred_dir = model_outputs['pred_dir'][spatial_mask]
        true_dir = batch_data['direction'][spatial_mask]
        loss_dir = (-1* F.cosine_similarity(pred_dir, true_dir, dim=1)).mean()

        pred_dist = model_outputs['pred_dist'][spatial_mask].squeeze()
        true_dist = batch_data['distance'][spatial_mask]
        loss_dist = F.mse_loss(pred_dist, true_dist)

        pred_area = model_outputs['pred_area'][spatial_mask].squeeze()
        true_area = batch_data['area'][spatial_mask]
        loss_area = F.mse_loss(pred_area, true_area)

    total_loss = (loss_weights['clip'] * loss_clip +
                  loss_weights['dir'] * loss_dir +
                  loss_weights['dist'] * loss_dist +
                  loss_weights['area'] * loss_area)
                  
    losses = {
        "total_loss": total_loss, "loss_clip": loss_clip.item(), "loss_dir": loss_dir.item(),
        "loss_dist": loss_dist.item(), "loss_area": loss_area.item()
    }
    return losses, total_loss

# --- Training Loop ---
def train_one_epoch(model, dataloader, optimizer, scheduler, epoch, device, loss_weights, is_main_process, accumulation_steps):
    model.train()
    total_epoch_loss = 0.0
    num_batches = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=not is_main_process)
    for i, batch in enumerate(pbar):
        if batch is None: continue

        # --- ▼▼▼ 損失計算に渡すデータを修正 ▼▼▼ ---
        audio_input = {k: v.to(device) for k, v in batch['audio'].items()}
        text_input = batch['text']
        # 'audio'と'text'以外の全てのキーをまとめてデバイスに転送
        batch_data_on_device = {k: v.to(device) for k, v in batch.items() if k not in ['audio', 'text']}
        # --- ▲▲▲ 修正ここまで ▲▲▲ ---

       
        
        logit_scale = model.module.logit_scale if hasattr(model, 'module') else model.logit_scale
        with torch.amp.autocast("cuda", dtype=torch.float16):
            model_outputs = model(audio_input, text_input)
        
        # 修正されたbatch_data_on_deviceを渡す
            loss_dict, loss_value = calculate_total_loss(model_outputs, batch_data_on_device, logit_scale, loss_weights, device)
        loss = loss_value / accumulation_steps
        scaler.scale(loss).backward()
        # 3. 指定したステップ数に達するか、最後のバッチであればパラメータを更新
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad() 
             # 勾配をリセット

        total_epoch_loss += loss_value.item()
        num_batches += 1
        if is_main_process:
            pbar.set_postfix({"Loss": f"{loss_value.item():.4f}", "LR": f"{scheduler.get_last_lr()[0]:.6f}"})
            if wandb.run is not None:
                wandb.log({
                    "train/step_loss": loss_value.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/temperature": logit_scale.exp().item(), # 温度τ
                    **{f"train/{k}": v for k, v in loss_dict.items() if k != 'total_loss'}
                })
    scheduler.step()  # 学習率スケジューラの更新
    avg_epoch_loss = total_epoch_loss / num_batches if num_batches > 0 else 0.0
    return avg_epoch_loss  # 1エポックの平均損失を返す
 
        
# --- Main Function ---
def main(args):
    with open("model/config.yaml", "r") as f: cfg = yaml.safe_load(f)

    # --- ▼▼▼【この引数の設定を修正】▼▼▼ ---
    # nargs='?': 引数を省略可能にする
    # const='initial': 省略した場合のデフォルト値 (パスではないことを示す文字列)
    # --- ▼▼▼ 検証用引数を追加 ▼▼▼ ---
    parser = argparse.ArgumentParser(description="Train or evaluate the ELSA model.")
    parser.add_argument(
        "--eval_only",
        action='store_true', # この引数があればTrueになる
        help="If specified, run evaluation only without training."
    )
    args = parser.parse_args()
    rank, world_size, device = init_dist_and_get_device() # Get the correct device
    checkpoint_path = cfg['train'].get('checkpoint_path')

    is_main_process = rank == 0
    is_ddp = world_size > 1
        # --- ▼▼▼【ここが修正箇所】▼▼▼ ---
    # 3. wandbの初期化 (メインプロセスでのみ実行)
    if is_main_process:
        wandb.init(
            project="ELSA-training", # プロジェクト名 (任意)
            config=cfg # 設定ファイルを丸ごと記録
        )
    # Move model to the correct device
    model = ELSA(cfg['model']['audio_encoder'], cfg['model']['text_encoder']).to(device)
        # 検証のみモードの場合
    # 検証のみモードの場合

    if is_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    # --- ▼▼▼ データセット準備のロジックを修正 ▼▼▼ ---
    mode = cfg.get('dataset_mode', 'foa')
    ### ▼▼▼ 累積勾配のための修正 ▼▼▼ ###
    # configファイルから累積ステップ数を読み込む (なければデフォルトで1)
    accumulation_steps = cfg['train'].get('accumulation_steps', 1)
    if is_main_process:
        print(f"Using gradient accumulation with {accumulation_steps} steps.")
    ### ▲▲▲ 修正ここまで ▲▲▲ ###
    if mode == 'both':
        foa_train_cfg = cfg['train_dataset']
        foa_train = ELSADataset(
            foa_folder=foa_train_cfg['foa_folder'],
            foa_metadata_csv=foa_train_cfg['foa_metadata_csv'],
            split='train'
        )
        
        mono_train_cfg = cfg['train_dataset']
        mono_train = AudioCapsDatasetForELSA(
            mono_folder=mono_train_cfg['mono_folder'],
            mono_metadata_csv=mono_train_cfg['mono_metadata_csv']
        )
        train_dataset = ConcatDataset([foa_train, mono_train])
    elif mode == 'foa':
        train_dataset = ELSADataset(split='train', **cfg['train_dataset'])
    else:
        # 他のモードが必要な場合はここに追加
        raise ValueError(f"Unsupported dataset_mode: {mode}")

    
    print(f"Combined training dataset size: {len(train_dataset)}")
    foa_val_cfg = cfg['val_dataset']
    if args.eval_only is not None:
        foa_val_dataset = ELSADataset(
                foa_folder=foa_val_cfg['foa_folder'],
                foa_metadata_csv=foa_val_cfg['foa_metadata_csv'],
                split='test'
            )
    else:foa_val_dataset = ELSADataset(
            foa_folder=foa_val_cfg['foa_folder'],
            foa_metadata_csv=foa_val_cfg['foa_metadata_csv'],
            split='val'
        )

    mono_val_cfg = cfg['val_dataset']
    mono_val_dataset = AudioCapsDatasetForELSA(
            mono_folder=mono_val_cfg['mono_folder'],
            mono_metadata_csv=mono_val_cfg['mono_metadata_csv']
        )
    print(f"Validation FOA dataset size: {len(foa_val_dataset)}")    
    print(f"Validation mono dataset size: {len(mono_val_dataset)}")
 

    # --- ▲▲▲ 修正ここまで ▲▲▲ ---
    train_sampler = distributed.DistributedSampler(train_dataset) if is_ddp else None
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['train']['batch_size'],
        sampler=train_sampler, num_workers=cfg['train']['num_workers'],
        collate_fn=collate_fn, pin_memory=False if device.type == 'cuda' else False,
        shuffle=(train_sampler is None), drop_last=True
    )
    #val_sampler = distributed.DistributedSampler(val_dataset) if is_ddp else None
    mono_val_loader = DataLoader(
        mono_val_dataset, batch_size=cfg['train']['batch_size'] , # 検証時は大きめのバッチサイズでも可
        num_workers=cfg['train']['num_workers'], collate_fn=collate_fn, shuffle=False
    )
    foa_val_loader = DataLoader(
        foa_val_dataset, batch_size=cfg['train']['batch_size'] , # 検証時は大きめのバッチサイズでも可
        num_workers=cfg['train']['num_workers'], collate_fn=collate_fn, shuffle=False)
    opt_cfg = cfg['optimizer']
    optimizer_name = opt_cfg.get('name', 'AdamW') # デフォルトはAdamW
    
    params_to_optimize = model.parameters()

    if optimizer_name.lower() == 'adam':
        print("Using Adam optimizer.")
        optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=opt_cfg['lr'],
            # Adamのweight_decayはL2正則化として機能
            weight_decay=opt_cfg.get('weight_decay', 0.0) 
        )
    elif optimizer_name.lower() == 'adamw':
        print("Using AdamW optimizer.")
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=opt_cfg['lr'],
            # AdamWのweight_decayは重要な役割を持つ
            weight_decay=opt_cfg.get('weight_decay', 0.01) 
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # スケジューラはオプティマイザの後に定義
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'])
    print(f"Using CosineAnnealingLR scheduler with T_max={cfg['train']['epochs']} epochs.")
    loss_weights = cfg.get('loss_weights', {
        'clip': 1.0, 'dir': 1.0, 'dist': 1.0, 'area': 1.0
    }) # 重み設定がない場合のデフォルト値    
    print(f"Training started on device: {device}")
 
    # --- ▼▼▼ 検証のみ実行するロジックを追加 ▼▼▼ ---
    if args.eval_only is not None:
        ckpt_path = checkpoint_path
        if ckpt_path:
            print(f"Loading model from checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model_state = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(model_state)
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print(f"Loaded model state from {ckpt_path}")

        print("\n--- RUNNING IN EVALUATION-ONLY MODE ---")

        if is_main_process:
            validate_and_evaluate(model, foa_val_loader, device, cfg['loss_weights'], epoch=0, eval_name="val_foa", is_main_process=is_main_process)
            validate_and_evaluate(model, mono_val_loader, device, cfg['loss_weights'], epoch=0, eval_name="val_mono", is_main_process=is_main_process)
        if wandb.run is not None:
            #wandb.save(save_path)
            wandb.finish()
        print("\nEvaluation Finished")
        return

    for epoch in range(cfg['train']['epochs']):
        if is_ddp: train_sampler.set_epoch(epoch)
        epoch_loss = train_one_epoch(model, train_loader, optimizer, scheduler, epoch, device, loss_weights, is_main_process, accumulation_steps)

    #if is_main_process:
        validate_and_evaluate(model, foa_val_loader, device, cfg['loss_weights'], epoch, "val_foa", is_main_process)
        validate_and_evaluate(model, mono_val_loader, device, cfg['loss_weights'], epoch, "val_mono", is_main_process)
   
        save_path = cfg['train']['check_point_folder_path']
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)  
        model_state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': epoch_loss,
        }, f"{save_path}/elsa_epoch_{epoch+1}.pt")
     
        if wandb.run is not None:
            wandb.save(save_path)

    # --- ▼▼▼【ここが修正箇所】▼▼▼ ---
    # 4. wandbの終了処理
    if rank == 0:
        wandb.finish()
    # --- ▲▲▲【修正はここまで】▲▲▲ ---

    if is_ddp: torch.distributed.destroy_process_group()
    print("Training finished.")
if __name__ == '__main__':
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="Train or evaluate the ELSA model.")
    # (他の引数は変更なし)
   
    # --- ▼▼▼【この引数の設定を修正】▼▼▼ ---
    # nargs='?': 引数を省略可能にする
    # const='initial': 省略した場合のデフォルト値 (パスではないことを示す文字列)
    parser.add_argument(
        "--eval_only",
        nargs='?',
        const='initial', 
        default=None,
        help="Run evaluation only. Optionally provide a path to a checkpoint. If no path, evaluates on an initial model."
    )
    # --- ▲▲▲【修正ここまで】▲▲▲ ---
    
    args = parser.parse_args()
    main(args)