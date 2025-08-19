import torch, soundfile as sf, librosa, numpy as np
from transformers import ClapProcessor, ClapAudioModelWithProjection, ClapAudioModel
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from typing import List
# auioencoder : HTSATとSpatial Encoderで構成される

# -------------------- 1.HTSATモデルとプロセッサ-----------------
# ----------------------------------------------------------------
import torch
import torch.nn as nn
from typing import List
import numpy as np
# 必要なのはClapAudioModelWithProjectionとClapProcessorだけ
from transformers import ClapProcessor, ClapAudioModelWithProjection

class HTSAT(nn.Module):
    def __init__(self, ckpt: str = "laion/clap-htsat-fused"):
        """
        ckpt: ClapAudioModelWithProjection の checkpoint 名
        """
        super().__init__()
        self.processor = ClapProcessor.from_pretrained(ckpt)

        # ▼▼▼【変更点】▼▼▼
        # 完全なモデルを一度ロードする
        full_model = ClapAudioModelWithProjection.from_pretrained(ckpt)
        # その中から、HTSAT本体（768次元出力）だけをクラスのプロパティとして保持する
        self.model = full_model.audio_model
        # ▲▲▲ 変更ここまで ▲▲▲
   
    def forward(self, omni_wave: torch.Tensor) -> torch.Tensor:
        """
        omni_wave: Tensor の形状は (B, L)
        戻り値: Tensor of shape (B, 768)
        """
        omni_cpu = omni_wave.detach().to("cpu")  # CPUに移動
        # ... (ここから下のforwardメソッドの中身は一切変更ありません) ...
        batch_size = omni_wave.size(0)
        raw_list: List[np.ndarray] = []
        for i in range(batch_size):
            wave_i = omni_cpu[i].numpy()
            raw_list.append(wave_i)

        inputs = self.processor(
            audios=raw_list,
            sampling_rate=48_000,
            return_tensors="pt",
            padding="repeatpad",
            truncation="max_length",
            max_length=480_000
        )

        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        
        return outputs.pooler_output
# ===== spatial_branchに対応

class AddCoords2D(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device), indexing="ij")
        coords = torch.stack([yy, xx]).unsqueeze(0).repeat(B, 1, 1, 1)
        return torch.cat([x, coords], dim=1)

# def conv_block(in_ch, out_ch):
#     return nn.Sequential(
#         AddCoords2D(),
#         nn.Conv2d(in_ch + 2, out_ch, 3, padding=1),
#         nn.BatchNorm2d(out_ch),
#         nn.MaxPool2d(2),
#         nn.ELU()
#     )


class Branch(nn.Module):
    """6‑block CNN, AddCoords2D は最初のブロックのみに適用。
    出力は (B,16,3,25) → Flatten 1200 で論文図に合わせる"""
    def __init__(self):
        super().__init__()
        layers = []
        # --- Block‑1 (AddCoords2D + Conv 5→16) ---
        layers.append(nn.Sequential(
            AddCoords2D(),
            nn.Conv2d(3 + 2, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.ELU())
        )
        # --- Block‑2 〜 Block‑6 (Conv 16→16) ---
        for _ in range(5):
            layers.append(nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.MaxPool2d(2),
                nn.ELU())
            )
        self.cnn = nn.Sequential(*layers)
        self.flat = nn.Sequential(nn.Flatten(), nn.Dropout(0.3))

    def forward(self, x):
        z = self.cnn(x)          # (B,16,3,25)
        return self.flat(z)      # (B,1200)


class SpatialAttributesBranch(nn.Module):
    """図 A.F.2 準拠 (Bx1200 → concat Bx2400 → 3‑layer MLP)"""
    def __init__(self, hidden1=128, hidden2=32, out_dim=44):
        super().__init__()
        self.act = Branch()
        self.rea = Branch()
        self.mlp = nn.Sequential(
            nn.Linear(1200 * 2, hidden1), nn.ELU(), 
            nn.Linear(hidden1, hidden2), nn.ELU(),
            nn.Linear(hidden2, out_dim)
        )
    
    def forward(self, I_act, I_rea):
        z = torch.cat([self.act(I_act), self.rea(I_rea)], dim=1)  # (B,2400)
        return self.mlp(z)




class AudioEncoder(nn.Module):
    def __init__(self, hidden1 =768,out_dim = 512):
        super().__init__()
        self.htsat= HTSAT()
        self.spatial_branch = SpatialAttributesBranch()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.spatial_branch.load_state_dict(torch.load(
            "/home/takamichi-lab-pc09/elsa/ckpt/takamichi09/SpatialLibriSpeech/check/ckpt_ep100.pt",
            map_location=device
        ))

        self.spatial_to_elsa = nn.Sequential(nn.ELU(),nn.Linear(44,192))
        self.audio_projection = nn.Sequential(
            nn.Linear(192+768,hidden1),nn.ELU(),
            nn.Linear(hidden1,out_dim)
        )   
    def forward(self, I_act, I_rea, Omni):
    
        concat = torch.cat([self.spatial_to_elsa(self.spatial_branch(I_act,I_rea)),self.htsat(Omni)],dim=1)
        audio_embeds = self.audio_projection(concat)
        return audio_embeds