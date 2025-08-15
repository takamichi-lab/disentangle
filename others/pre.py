'''
元のAudioCapsのcsvと

RIR用のcsvが必要
- RIRデータのパス名
- メタデータ
    - 部屋の大きさ
    - 部屋の面積
    - マイクの位置（おそらく部屋の中心)
    - 音源の位置
    - それらの距離
    - 残響時間
    - 方向

以下のコードで行う大事な考え方
- ミニバッチを作るときに、N個のドライ音源に対して、M個のRIRをランダムに選ぶ
- つまり、N×Mのサイズのミニバッチを作る
- 教師ありcontrastive lossでsourceとspaceそれぞれに応じた挙動
'''


## 1. dataset　クラス
#Todo : textについても畳み込むときに、ドライ音源とRIRから、GPT-4.1nanoに生成させる。
class AudioTextContrastiveDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 ir_dict: dict,
                 tokenizer,
                 n_views: int = 1,         # ← 追加: 1ドライあたりのビュー数
                 use_on_the_fly_ir: bool = True):
        import pandas as pd
        df = pd.read_csv(csv_path)
        self.rows = df.to_dict(orient="records")
        self.ir_dict = ir_dict
        self.tokenizer = tokenizer
        self.n_views = n_views
        self.use_ir = use_on_the_fly_ir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        base_wave, sr = torchaudio.load(row["audio_path"])  # [C, T]
        # テキストのトークナイズ
        encoding = self.tokenizer(
            row["full_text"], return_tensors="pt",
            padding="longest", truncation=True
        )
        input_ids     = encoding["input_ids"].squeeze(0)  
        attention_mask= encoding["attention_mask"].squeeze(0)
        src_id = source_label_to_id(row["source_label"])
        spc_id = space_label_to_id(row["space_label"])

        # 複数ビュー生成
        views = []
        for _ in range(self.n_views):
            wave = base_wave.clone()
            if self.use_ir:
                ir_path = random.choice(self.ir_dict[row["space_label"]])
                ir_wave, _ = torchaudio.load(ir_path)
                wave = torchaudio.functional.fftconvolve(wave, ir_wave)
                # 必要なら pad_or_crop(wave, target_length)
            views.append(wave)

        return {
            "waveforms":      views,       # list of Tensors, len = n_views
            "input_ids":      input_ids,   # Tensor[L]
            "attention_mask": attention_mask,
            "source_id":      torch.long(src_id),
            "space_id":       torch.long(spc_id),
        }

# 2. Collate 関数と DataLoader
from torch.nn.utils.rnn import pad_sequence

def multiview_collate(batch):
    """
    batch: list of dicts, len=N (バッチ内のドライ種数)
    各要素['waveforms'] は長さ M のリストになっているので、
    それを flatten して N*M サンプルに展開します。
    """
    flat_waves = []
    flat_ids   = []
    flat_masks = []
    flat_src   = []
    flat_spc   = []

    for elem in batch:
        for wave in elem["waveforms"]:
            # wave: [C, T] → [T,] なら transpose を入れるかそのまま pad する
            flat_waves.append(wave.squeeze(0).t())
            flat_ids.append(elem["input_ids"])
            flat_masks.append(elem["attention_mask"])
            flat_src.append(elem["source_id"])
            flat_spc.append(elem["space_id"])

    # 音声のパディング
    waves_padded = pad_sequence(flat_waves, batch_first=True).transpose(1,2)
    # テキストのパディング
    ids_padded   = pad_sequence(flat_ids,   batch_first=True, padding_value=0)
    masks_padded = pad_sequence(flat_masks, batch_first=True, padding_value=0)
    # ラベル
    src_labels = torch.stack(flat_src)
    spc_labels = torch.stack(flat_spc)

    return {
        "waveform":       waves_padded,   # [N*M, C, T_max]
        "input_ids":      ids_padded,     # [N*M, L_max]
        "attention_mask": masks_padded,   # [N*M, L_max]
        "source_id":      src_labels,     # [N*M]
        "space_id":       spc_labels,     # [N*M]
    }

# N=4 (batch_size), M=3 (n_views)
dataset = AudioTextContrastiveDataset(
    csv_path="data/dataset.csv", # 元のaudioCapsのcsv
    ir_dict=ir_dict,   #RIRのパス名・部屋の大きさ、RT60, 面積、音源の座標、距離、などが入ったcsv
    tokenizer=roberta_tokenizer,
    n_views=3,                # M=3
    use_on_the_fly_ir=True
)
loader = DataLoader(
    dataset,
    batch_size=4,             # N=4
    shuffle=True,
    collate_fn=multiview_collate,
    num_workers=4,
    pin_memory=True,
)

# 教師ありcontrastive lossの計算する関数

import torch
import torch.nn.functional as F

def supervised_contrastive_loss(a_emb: torch.Tensor,
                                b_emb: torch.Tensor,
                                labels: torch.Tensor,
                                temperature: float = 0.07) -> torch.Tensor:
    """
    Cross-modal supervised contrastive loss:
      - a_emb: [B, D] （例: audio_space_emb）
      - b_emb: [B, D] （例: text_space_emb）
      - labels: [B] の整数ラベル（例: space_id）
    正例は同じラベルを持つインスタンス全ペア、負例は異なるラベル全ペア。
    """
    B, D = a_emb.size()
    # 正規化
    a_norm = F.normalize(a_emb, dim=1)   # [B, D]
    b_norm = F.normalize(b_emb, dim=1)   # [B, D]
    # 類似度行列 [B, B]
    logits = torch.matmul(a_norm, b_norm.t()) / temperature

    # ラベルマスク: same_label[i,j]=1 if labels[i]==labels[j] else 0
    label_eq = labels.unsqueeze(1) == labels.unsqueeze(0)  # [B, B]
    # 対角は自分自身なので除外
    mask = label_eq.fill_diagonal_(False)

    # 各行 i について： positives = {j | mask[i,j]=True}, negs = rest
    # 分母はすべての j≠i で exp(logits[i,j])
    exp_logits = torch.exp(logits) * (~torch.eye(B, dtype=bool, device=logits.device))
    denom = exp_logits.sum(dim=1, keepdim=True)  # [B,1]

    # 各 positive に対する loss を平均
    # L_i = - (1/|P_i|) * sum_{j in P_i} log( exp(logits[i,j]) / denom[i] )
    # where P_i = {j | mask[i,j]}
    pos_exp = exp_logits * mask.float()  # [B,B] positives only
    # sum_{j in P_i} exp_logits
    pos_sum = pos_exp.sum(dim=1)          # [B]
    # sum of log probs:
    # sum_{j in P_i} log( exp(logits[i,j]) / denom[i] )
    #   = sum_{j in P_i} (log exp_logits[i,j] - log denom[i])
    #   = sum_{j in P_i}( logits[i,j] ) - |P_i| * log denom[i]
    logits_pos = (logits * mask.float()).sum(dim=1)  # sum logits over P_i
    P_count = mask.sum(dim=1).clamp(min=1).float()   # avoid div0

    loss_i = - ( logits_pos - P_count * torch.log(denom.squeeze()) ) / P_count
    return loss_i.mean()


# train one epoch
for batch in loader:
    # ===== データ取得 =====
    wave = batch["waveform"]
    ids  = batch["input_ids"]
    mask = batch["attention_mask"]
    src_labels = batch["source_id"]  # [B]
    spc_labels = batch["space_id"]   # [B]

    # ===== 埋め込み計算 =====
 　 audio_space_emb, audio_source_emb, text_space_emb, text_source_emb = model(...)
    # 各テンソルは [B, D]

    # ===== 対照損失 =====
    loss_space  = supervised_contrastive_loss(
                      audio_space_emb,
                      text_space_emb,
                      spc_labels,
                      temperature=0.07
                   )
    loss_source = supervised_contrastive_loss(
                      audio_source_emb,
                      text_source_emb,
                      src_labels,
                      temperature=0.07
                   )
    loss = alpha * loss_space + (1-alpha) * loss_source

    # ===== 更新 =====
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
