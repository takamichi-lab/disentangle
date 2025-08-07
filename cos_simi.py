import torch
import torch.nn.functional as F


#4. バッチ単位での計算
#    テンソル形状は [batch_size, feature_dim]
vec_a4 = torch.tensor([
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, -1.0],
])
vec_b4 = torch.tensor([
    [0.0, 1.0],
    [1.0, 2.0],
    [1.0, 0.0],
])
# dim=1 で各行ごとに計算し、サイズは (3,) のテンソルになる
cos4 = F.cosine_similarity(vec_a4, vec_b4, dim=1)
print("Test 4 (batched):")
for i, val in enumerate(cos4, 1):
    print(f"  batch {i}: {val:.3f}")
