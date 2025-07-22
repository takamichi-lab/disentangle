# generate_rir.py
import random
import numpy as np
from pathlib import Path

# your_module は実際のモジュール名に置き換えてください
from gen_one_rir import gen_one_rir

def main():
    # 再現性のためシード固定
    random.seed(42)
    np.random.seed(42)

    # 出力先ディレクトリ（プロジェクト直下に作られる）
    out_dir = Path("./rir_output_real")



    # RIR 生成
    gen_one_rir(out_dir, "test")

    # 出力されたファイル一覧を表示
    print(f"=== {out_dir} の中身 ===")
    for p in sorted(out_dir.iterdir()):
        print(p.name)

if __name__ == "__main__":
    main()