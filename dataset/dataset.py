import random, torchaudio, torch, pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

# audiocap_id,youtube_id,start_time,caption
# 91139,r1nicOVtvkQ,130,A woman talks nearby as water pours
#.venv) takamichi-lab-pc09@takamichi-lab-pc09:~/DELSA$ python3 dataset/check_dataset.py     --audio_csv  /home/takamichi-lab-pc09/DELSA/AudioCaps_csv/train.csv     
# --rir_csv    /home/takamichi-lab-pc09/DELSA/RIR_dataset/rir_catalog_train.csv     --n_views    2     --batch_size 4  --audio_base AudioCaps_mp3
class AudioRIRDataset(Dataset):
    """
    csv_audio: audiocaps_train.csv (audio_path, caption)  
    csv_rir: rir_catalog_train.csv(rir_path,, metadata)

    戻り値:
       waveform: audiocaps*rirで生成された音[C,T]
       text: Spatial caption
       source_id: 音源のID
       spatial_id: RIRのID

    """
    def __init__ (self,
                  csv_audio: str,
                  base_dir: str,
                  csv_rir: str,
                  n_views: int = 1,
                  split: str = "train"
                  
                  ):
        # audio_csv読み込み
        self.audio_df = pd.read_csv(csv_audio)

        # source_id = 行番号( 0....... len(audio_df) - 1 )
        self.source_map = {idx: idx for idx in range(len(self.audio_df))}
        # rir_csv読み込み
        rir_df = pd.read_csv(csv_rir)
        # rir_id = 行番号( 0....... len(rir_df) - 1
        self.space_map = {p: i for i,p in enumerate(rir_df["rir_path"].unique())}

        # rir_dict: Path -> [Path] 1:1でも辞書にする
        self.rir_dict = defaultdict(list)
        for p in rir_df["rir_path"]:
            self.rir_dict[p].append(p)
         #dry音源1つに対するRIRの数
        self.n_views = n_views
        self.base_dir = Path(base_dir)
        self.split = split
        # 読み込んだあとの audio_df にフィルタをかける
        self.audio_df = self.audio_df[
        self.audio_df["audiocap_id"].apply(
        lambda i: (self.base_dir/self.split/f"{i}.mp3").exists()
         )].reset_index(drop=True)        
    def __len__(self):
        return len(self.audio_df)
    
    def __getitem__(self, idx: int):
        row = self.audio_df.iloc[idx]

        waves, texts, src_ids, spa_ids = [], [], [], []
        for _ in range(self.n_views):
            #1 )ドライ音を読み込み
            mp3 = self.base_dir / self.split / f"{row['audiocap_id']}.mp3"
            wav, sr = torchaudio.load(mp3)
            #2 ) ランダムRIRを選択
            rir_path =random.choice(list(self.rir_dict.keys()))
            rir_wav, rir_sr = torchaudio.load(rir_path)
            
            #3 ) RIRを適用
            sp_audio = torchaudio.functional.convolve(wav, rir_wav)
            # ToDo: A-format to B-format

            print(sp_audio.shape, sp_audio.max(), sp_audio.min())
            # labels
            src_ids.append(self.source_map[idx])
            spa_ids.append(self.space_map[rir_path])
            waves.append(sp_audio)
            #print(max(waves[0][0]))
            texts.append(row["caption"])
            print(waves, texts, src_ids, spa_ids)
        
        return{
            "waves": waves,
            "texts": texts,
            "source_ids": src_ids,
            "space_ids": spa_ids

        }

def collate_fn(batch):
    wave_list ,text_list, src_list, spa_list = [], [], [], []

    for b in batch:
        wave_list += b["waves"]
        text_list += b["texts"]
        src_list += b["source_ids"]
        spa_list += b["space_ids"]
    # time方向だけゼロパティング
    max_len = max(w.shape[1] for w in wave_list)
    padded_waves = []
    for w in wave_list:
        if w.shape[-1] < max_len:
            pad = torch.nn.functional.pad(w, (0, max_len - w.shape[-1]))
            padded_waves.append(pad)
        else:
            padded_waves.append(w)

    waves_padded = torch.stack(padded_waves)
    return{
        "waves": waves_padded,
        "texts": text_list,
        "source_ids": torch.stack(src_list),
        "space_ids": torch.stack(spa_list)
    }
