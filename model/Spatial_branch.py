# ===== spatial_branch.py =====
import torch
import torch.nn as nn


class AddCoords2D(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device), indexing="ij")
        coords = torch.stack([yy, xx]).unsqueeze(0).repeat(B, 1, 1, 1)
        return torch.cat([x, coords], dim=1)

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        AddCoords2D(),
        nn.Conv2d(in_ch + 2, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.MaxPool2d(2),
        nn.ELU()
    )


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

