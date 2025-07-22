# file: models/delsa_model.py

import torch
import torch.nn as nn
from .shared_audio_encoder import AudioEncoder
from .shared_text_encoder import TextEncoder
import numpy as np

# DELSAでは一旦、物理損失および物理量用のMLPは無視

class DELSA(nn.Module):
    """
    論文のアーキテクチャ全体を統合したELSAモデル。
    """
    def __init__(self, audio_encoder_cfg: dict, text_encoder_cfg: dict):
        super().__init__()
        # Audio/Textエンコーダをインスタンス化
        shared_dim = 512
        out_space_dim = 256 
        out_source_dim = 512
        # 共有用のaudio Encoder
        self.audio_encoder = AudioEncoder(**audio_encoder_cfg)
        # ---- ② 共有 -> 空間用 ----        
        self.audio_space_head = nn.Sequential(
            nn.ELU(),
            nn.Linear(shared_dim, out_space_dim)
        )
        # ---- ③ 共有 -> 音源用 ----
        self.audio_source_head = nn.Sequential(
            nn.ELU(),
            nn.Linear(shared_dim, out_source_dim)
        )
        # 共有用のtext Encoder
        self.text_encoder = TextEncoder(**text_encoder_cfg)
        # ---- ④ 共有 -> 空間用 ----
        self.text_space_head = nn.Sequential(
            nn.ELU(),
            nn.Linear(shared_dim, out_space_dim)
        )
        # ---- ⑤ 共有 -> 音源用 ----
        self.text_source_head = nn.Sequential(
            nn.ELU(),
            nn.Linear(shared_dim, out_source_dim)
        )
        # 論文の図に基づき、最終的な埋め込み次元は512
        # embedding_dim = 512

        # # --- ▼▼▼ このように書くのがおすすめです ▼▼▼ ---
        # 学習可能な温度パラメータ (CLIPの初期値)
        initial_value = np.log(1 / 0.07)
        self.logit_scale = nn.Parameter(torch.tensor(initial_value, dtype=torch.float32))


    def forward(self, audio_data: torch.Tensor, text_data: torch.Tensor):
        """
        音声とテキストを入力とし、すべての出力を辞書形式で返す。
        """
        # audio_shared_embedsはspaceとsourceの要素をconcatしてmlpに通されたaudio_embeds

        audio_shared_embeds = self.audio_encoder(
            I_act=audio_data['i_act'],
            I_rea=audio_data['i_rea'],
            Omni=audio_data['omni_48k']
        )
        # --- 音の埋め込み分岐 ---
        z_space_a  = self.audio_space_head(audio_shared_embeds)     # 空間ベクトル [B, out_space_dim]
        z_source_a = self.audio_source_head(audio_shared_embeds)    # 音源ベクトル [B, out_source_dim]
        # text_shared_embedsはspaceとsourceの要素をconcatしてmlpに通されたtext_embeds
        text_shared_embeds = self.text_encoder(text_data)   # 512次元のテキスト埋め込み
        # --- テキストの埋め込み分岐 ---
        z_space_t  = self.text_space_head(text_shared_embeds)     # 空間ベクトル [B, out_space_dim]
        z_source_t = self.text_source_head(text_shared_embeds)    # 音源ベクトル [B, out_source_dim]
        return {
            "audio_space_emb": z_space_a,          # 音の空間埋め込み (B, out_space_dim)
            "audio_source_emb": z_source_a,         # 音のソース埋め込み(B, out_source_dim)
            "text_space_emb": z_space_t,           # テキストの空間埋め込み (B, out_space_dim)
            "text_source_emb": z_source_t,          # テキストのソース埋め込み (B, out_source_dim)
            "logit_scale": self.logit_scale   # 学習可能な温度パラメータ

        }