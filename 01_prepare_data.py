"""
SECOM データセットのダウンロードと前処理

半導体製造ラインのセンサーデータ（590センサー × 1567サンプル）
- 正常: 1463件 (ラベル -1)
- 異常: 104件  (ラベル +1)
"""

import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SECOM_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
LABELS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"


def download_if_needed(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  既存ファイルを使用: {dest}")
        return
    print(f"  ダウンロード中: {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  保存完了: {dest}")


def main():
    print("=== 1. データダウンロード ===")
    download_if_needed(SECOM_URL,  DATA_DIR / "secom.data")
    download_if_needed(LABELS_URL, DATA_DIR / "secom_labels.data")

    print("\n=== 2. 読み込み ===")
    X = pd.read_csv(DATA_DIR / "secom.data", sep=" ", header=None)
    y_df = pd.read_csv(DATA_DIR / "secom_labels.data", sep=" ", header=None)
    y = y_df.iloc[:, 0].values  # -1: 正常, +1: 異常

    print(f"  全サンプル数  : {len(X)}")
    print(f"  センサー数    : {X.shape[1]}")
    print(f"  正常 (y=-1)  : {(y == -1).sum()}")
    print(f"  異常 (y= 1)  : {(y ==  1).sum()}")

    print("\n=== 3. 前処理 ===")
    # 欠損率 > 40% の列を削除
    missing_rate = X.isnull().mean()
    cols_to_drop = missing_rate[missing_rate > 0.4].index
    X = X.drop(columns=cols_to_drop)
    print(f"  欠損率 > 40% で削除した列数 : {len(cols_to_drop)}")

    # 残りの欠損値を列の中央値で補完
    X = X.fillna(X.median())

    # 分散ゼロの列を削除
    zero_var_cols = X.columns[X.std() == 0]
    X = X.drop(columns=zero_var_cols)
    print(f"  分散ゼロで削除した列数     : {len(zero_var_cols)}")

    print(f"  最終的なセンサー数         : {X.shape[1]}")

    # 正常データだけで StandardScaler をフィット（テストデータへの情報漏洩防止）
    X_normal = X[y == -1]
    scaler = StandardScaler()
    scaler.fit(X_normal)
    X_scaled = scaler.transform(X)

    print("\n=== 4. 保存 ===")
    np.save(DATA_DIR / "X_scaled.npy", X_scaled.astype(np.float32))
    np.save(DATA_DIR / "y.npy", y)
    with open(DATA_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    # スケーラーのパラメータをONNX推論用にも保存
    np.save(DATA_DIR / "scaler_mean.npy",  scaler.mean_.astype(np.float32))
    np.save(DATA_DIR / "scaler_scale.npy", scaler.scale_.astype(np.float32))

    print(f"  data/X_scaled.npy  : shape={X_scaled.shape}")
    print(f"  data/y.npy         : shape={y.shape}")
    print(f"  data/scaler.pkl    : 標準化パラメータ")
    print(f"  data/scaler_mean.npy / scaler_scale.npy : ONNX推論用")
    print("\n完了。次のステップ: python 02_train.py")


if __name__ == "__main__":
    main()
