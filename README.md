
# DELSA プロジェクト前提まとめ（ファイル構成込み・ChatGPT向けブリーフ）

最終更新: 自動生成

---

## ゴール
- Dry 音 × RIR から **FOA (4ch: W,Y,Z,X)** を生成し、**事前計算した Validation セット**で「毎エポック同一」の検証を行い、メトリクスを安定化。
- 必要に応じて RIR は **全部**または **代表サブセット（層化）** を使用。

## 主要ハイライト
- **SR:** FOA 48k (`FOA_SR`), IV 16k (`IV_SR`)
- **事前計算出力:** `foa/*.wav`（16-bit PCM）/ `feat/*.pt`（`i_act,i_rea`）/ `val_precomputed.csv`
- **学習タスク:** 空間埋め込みとソース埋め込みの **クロスモーダル SupCon**、および **物理量回帰（方向/距離/面積/残響）**
- **Val は固定**（seed固定＋RIR順序固定）。

---

## データフロー（Train/Val共通の想定）
1. Dry (48k, mono) を読み込み → RIR(FOA 4ch) と **FFT 畳み込み** → FOA (4ch, 48k) を得る  
2. FOA→16k にリサンプルして `foa_to_iv()` で **(i_act, i_rea)** を計算  
3. Audio エンコーダ（HTSAT/CLAP 系）→ **shared 埋め込み**  
4. MLP で **space/source** の 2 ヘッドへ分岐  
5. Text エンコーダ（RoBERTa）→ shared → **space/source 2 ヘッド**  
6. **SupCon (space, source)** + **物理損失** を合算して最適化  
7. Val は **前計算済み**の FOA/IV/CSV を読み込む

---

## ファイル別の要点

### `dataset/audio_rir_dataset.py`
- **定数:** `MAX_DURATION_SEC=10.0`, `FOA_SR=48000`, `IV_SR=16000`
- **主関数:**
  - `rewrite_caption(orig, meta)` … 距離/方位/仰角/部屋サイズ/残響をルールベースで字幕に反映
  - `foa_to_iv(foa_wave, n_fft, hop)` … **(i_act, i_rea)** を返す（**タプル**）。形状は概ね `(B, 3, F, T)`
- **`AudioRIRDataset`**
  - `__init__(csv_audio, base_dir, csv_rir, n_views=1, split="train", n_fft=400, share_rir=True, batch_size=None, stats_path=..., hop=100)`
  - `__getitem__(idx)` → 下記を返す（**1サンプル＝複数 view**）
    ```python
    {
      "audio": [{"i_act":(3,F,T), "i_rea":(3,F,T), "omni_48k":(T)}, ... n_views],
      "texts": [caption_spatial for each view],
      "source_id": LongTensor[n_views],
      "space_id":  LongTensor[n_views],
      "rir_meta":  [dict per view]  # azimuth/elevation など
    }
    ```
    - `rir_meta["direction_vec"] = tensor([rad(az), rad(el)])`
    - **注意:** `_apply_rir()` 内に `sf.write('foa.wav', ...)` の**デバッグ書き出し**が残っていると I/O コスト増。不要なら削除推奨。
  - **RIR 選択ロジック**
    - `share_rir=True` なら **バッチ単位**で `random.sample` した RIR セットを共有
    - `share_rir=False` なら **サンプルごと**に `random.sample`
  - **`collate_fn(batch)`** … `rir_meta` を **キーごとにスタック**し、学習が扱いやすい形にまとめる

### `shared_text_encoder.py`
- **`TextEncoder`**
  - `pretrained_model_name="roberta-base"` を使用
  - Tokenizer + RoBERTa → MLP（`mlp_hidden_size=640` → `output_dim=512`）
  - `forward(texts: List[str]) -> Tensor[B, 512]`

### `shared_audio_encoder.py`
- **CLAP 系**の `ClapAudioModelWithProjection` と `ClapProcessor` を用い、オムニ音声を埋め込みへ（`outputs.pooler_output` を返却）。
- 将来的に **Spatial 分枝**（`I_act/I_rea` の利用）も考慮された構成。

### `RegressionHead.py`
- 物理量回帰の小さな MLP（`Linear → ReLU → Linear`）  
  - 方向（2）、面積（1）、距離（1）、残響（1）などに流用。

### `delsa_model.py`
- **DELSA** 本体。オーディオ/テキストの共有エンコーダ → **space/source** 2 ヘッド。
- `logit_scale`（学習可能温度, 0.07 初期化）を保持。
- 物理量ヘッド: `direction(2), area(1), distance(1), reverb(1)` を出力。
- `forward(audio_data, text_data)` 返却例：
  ```python
  {
    "audio_space_emb": (B, Ds), "audio_source_emb": (B, Dq),
    "text_space_emb":  (B, Ds), "text_source_emb":  (B, Dq),
    "logit_scale": scalar,
    "direction": (B, 2), "area": (B, 1), "distance": (B, 1), "reverb": (B, 1)
  }
  ```

### `train.py`
- **設定**: `load_config()` で YAML をロード（デフォルトは最小: `split/batch_size/n_views/epochs/lr/device/wandb/proj/run_name`）  
  → `audio_csv_train/rir_csv_train/audio_csv_val/rir_csv_val/base_dir` などは **YAML 側で設定**する想定。
- **ロス**
  - `sup_contrast(a, b, labels, logit_scale, symmetric=True, exclude_diag=False)`  
    - **InfoNCE 型の SupCon**。`labels` 一致がポジティブ。両方向（Text→Audio/Audio→Text）平均可。
  - `physical_loss(model_output, batch_data, isNorm=True, dataloader=None)`  
    - 方向: **`-cosine_similarity`**（`direction_vec` と）  
    - 面積/距離/残響: **MSE**。  
    - `isNorm=True` の場合は **正規化空間**で比較（`stats.pt` の mean/std を利用）。
- **トレーニングループ**
  - `AudioRIRDataset(..., n_views, share_rir)` を train/val で作成（Val ブロックあり）
  - `model(audio_dict, texts)` → 埋め込み/ロス計算 → `wandb.log()` → **1エポックごとに checkpoint 保存**
- **バッチ整形**
  - `audio_dict = {k: stack([d[k] for d in batch["audio"]]) for k in ("i_act","i_rea","omni_48k")}`  
  - `texts = batch["texts"]`、`source_id/space_id` は `.reshape(-1)`

---

## Fixed Validation（毎エポック同一）の作り方

### 1) 固定集合の作成（代表例: Audio 400 × RIR 24）
```bash
python make_val_fixed.py \
  --audio_csv AudioCaps_csv/val.csv \
  --rir_csv   RIR_dataset/rir_catalog_val.csv \
  --out_dir   val_fixed_400x24 \
  --audio_n   400 \
  --rir_n     24 \
  --seed      42
```
**出力:** `audio_fixed.csv` / `rir_fixed.csv` / `pairs_fixed.csv` / `summary.txt`

### 2) 前計算（RIR 全部なら `--n_views 0`）
```bash
python scripts/precompute_val.py \
  --csv_audio val_fixed_400x24/audio_fixed.csv \
  --csv_rir   val_fixed_400x24/rir_fixed.csv \
  --out_dir   data/val_precomp_fixed \
  --n_views   24
```
**出力:** `data/val_precomp_fixed/{foa,feat,val_precomputed.csv}`

### 3) 学習側の読み込み（例）
- Val は **前計算データセット**（`PrecomputedValDataset`）を使用（1 view/行、`collate_fn`互換）
- DataLoader は `shuffle=False, drop_last=False`。

---

## 再現性と注意点
- `random.seed(42); torch.manual_seed(42)` をスクリプト先頭に
- `ds.rir_paths` は **`sorted(...)`** で固定順（前計算ではこれで決定論的に）
- **長さ規約**: 畳み込み後は `(len(dry)+len(rir)-1)` → **トリム/Pad**を train/val で統一（本実装は 10 s にクロップ）
- **スケーリング/クリップ**: train/val で揃える
- **チャンネル順**: FOA の **W,Y,Z,X** を維持
- **RIRリーク**: Train/Val の RIR CSV は **物理的に分離**
- **デバッグ書き出し**: `_apply_rir()` の `sf.write('foa.wav', ...)` は不要なら削除

---

## RIR の選択確認
```bash
python inspect_rir_selection.py \
  --csv val_fixed_400x24/rir_fixed.csv \
  --out val_fixed_400x24/report
```
- RIR 一覧 / 部屋・角度分布 / T30・距離などの統計を自動出力。  
- 前計算済みの実績から確認する場合は `val_precomputed.csv` を渡す。

---

## 追加で聞きたいことの例
- PrecomputedValDataset の実装詳細（今の `collate_fn` 互換で）
- メトリクス（view→clip→全体の二段平均、ランキング系の定義）
- CLAP/HTSAT や RoBERTa の **前計算/キャッシュ**戦略
- バリデーションの **サブセット設計** や **フル評価の運用**（節目だけ等）
