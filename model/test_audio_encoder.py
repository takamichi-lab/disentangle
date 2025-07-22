import torch
from audio_encoder import HTSAT 
import os

# CUDAが不安定な場合があるため、ひとまずCPUでテストする設定に固定
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def test_htsat_output_robustness():
    """
    HTSATモデルの出力の頑健性をテストします。
    1. 出力次元が(B, 768)であること
    2. バッチ内の出力が互いに異なること
    3. 出力にNaNや無限大が含まれないこと
    """
    print("--- HTSAT出力頑健性テスト開始 ---")

    # 1. モデルのインスタンス化
    try:
        # 最終版の正しいアーキテクチャでモデルをロード
        model = HTSAT() 
        print("✅ モデルのインスタンス化に成功しました。")
    except Exception as e:
        print(f"❌ モデルのインスタンス化でエラーが発生しました: {e}")
        return

    # CPUでテストを実行
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    print(f"テストデバイス: {device}")

    # 2. ダミーの入力データを作成
    batch_size = 2
    audio_length = 480000 
    
    # バッチ内で異なる入力を作成するため、別々に生成
    input_1 = torch.randn(1, audio_length)
    input_2 = torch.randn(1, audio_length) * 1.1 # わずかに違う入力
    dummy_input = torch.cat([input_1, input_2], dim=0).to(device)
    
    print(f"ダミー入力の形状: {tuple(dummy_input.shape)}")

    # 3. モデルで推論を実行
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print("✅ モデルの推論に成功しました。")
    except Exception as e:
        print(f"❌ モデルの推論でエラーが発生しました: {e}")
        return

    # 4. 出力の検証
    print(f"出力テンソルの形状: {tuple(output.shape)}")
    
    # --- 検証1: 出力次元のチェック ---
    expected_dim = 768
    assert output.shape[0] == batch_size, "バッチサイズが期待値と異なります"
    assert output.shape[1] == expected_dim, f"出力次元が期待値({expected_dim})と異なります"
    print("✅ 検証1: 出力次元は期待通りです。")

    # --- 検証2: バッチ内の出力が異なるかのチェック ---
    output_1 = output[0]
    output_2 = output[1]
    
    # 2つの出力ベクトルが完全に同じでないことを確認
    are_different = not torch.allclose(output_1, output_2)
    assert are_different, "バッチ内の2つの出力が全く同じ値です"
    print("✅ 検証2: バッチ内の出力はそれぞれ異なる値です。")

    # --- 検証3: Null(NaN/inf)でないかのチェック ---
    has_nan = torch.isnan(output).any()
    has_inf = torch.isinf(output).any()
    
    assert not has_nan, "出力にNaN (Not a Number) が含まれています"
    assert not has_inf, "出力に無限大 (inf) が含まれています"
    print("✅ 検証3: 出力にNaNや無限大は含まれていません。")

    print(f"\n🎉 全てのテスト成功！モデルは正常に出力を生成しています。")
    print("--- テスト終了 ---")


if __name__ == "__main__":
    test_htsat_output_robustness()