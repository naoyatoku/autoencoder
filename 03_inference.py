"""
ONNX Runtime による推論スクリプト

このファイルは PyTorch 不要。onnxruntime だけで動作します。
PyInstaller で .exe 化すれば Python環境のない Windows PC でも実行できます。

使い方:
  python 03_inference.py                    # デモ（SECOMテストデータで検証）
  python 03_inference.py --csv your_data.csv  # CSVファイルで推論
"""

import argparse
import numpy as np
import onnxruntime as ort
from pathlib import Path

MODEL_DIR = Path("models")
DATA_DIR  = Path("data")


def load_runtime():
    sess = ort.InferenceSession(
        str(MODEL_DIR / "autoencoder.onnx"),
        providers=["CPUExecutionProvider"],
    )
    threshold    = float(np.load(MODEL_DIR / "threshold.npy")[0])
    scaler_mean  = np.load(DATA_DIR / "scaler_mean.npy")
    scaler_scale = np.load(DATA_DIR / "scaler_scale.npy")
    return sess, threshold, scaler_mean, scaler_scale


def predict(sess, x_raw: np.ndarray, scaler_mean, scaler_scale, threshold):
    """
    x_raw : shape (N, n_sensors) の生センサー値
    戻り値: (anomaly_flags, scores)
      anomaly_flags : bool array, True = 異常
      scores        : float array, 再構成誤差
    """
    x_scaled = ((x_raw - scaler_mean) / scaler_scale).astype(np.float32)
    x_hat = sess.run(None, {"sensor_input": x_scaled})[0]
    scores = ((x_scaled - x_hat) ** 2).mean(axis=1)
    return scores > threshold, scores


def demo_mode(sess, threshold, scaler_mean, scaler_scale):
    """学習に使っていないサンプルで精度を確認"""
    print("=== デモモード: SECOMデータで検証 ===\n")
    X = np.load(DATA_DIR / "X_scaled.npy")
    y = np.load(DATA_DIR / "y.npy")

    # スケーラー適用済みデータをそのまま使う（デモ用）
    x_hat = sess.run(None, {"sensor_input": X})[0]
    scores = ((X - x_hat) ** 2).mean(axis=1)
    flags  = scores > threshold

    y_true = (y == 1).astype(int)
    tp = int(((flags) & (y_true == 1)).sum())
    fp = int(((flags) & (y_true == 0)).sum())
    fn = int(((~flags) & (y_true == 1)).sum())
    tn = int(((~flags) & (y_true == 0)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    print(f"  閾値          : {threshold:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

    print("\n  サンプル出力（先頭10件）:")
    print(f"  {'#':>4}  {'スコア':>8}  {'判定':>6}  {'正解':>6}")
    print("  " + "-" * 32)
    for i in range(10):
        label  = "異常" if y[i] == 1 else "正常"
        result = "異常" if flags[i] else "正常"
        mark   = "OK" if (flags[i] == (y[i] == 1)) else "NG"
        print(f"  {i:>4}  {scores[i]:>8.4f}  {result:>6}  {label:>6} {mark}")


def csv_mode(sess, threshold, scaler_mean, scaler_scale, csv_path: str):
    """CSVファイルを読み込んで推論"""
    import csv

    print(f"=== CSV推論モード: {csv_path} ===\n")
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append([float(v) if v.strip() else 0.0 for v in row])

    x_raw = np.array(rows, dtype=np.float32)
    if x_raw.shape[1] != len(scaler_mean):
        print(f"エラー: CSVの列数({x_raw.shape[1]})がモデルの入力次元({len(scaler_mean)})と一致しません")
        return

    flags, scores = predict(sess, x_raw, scaler_mean, scaler_scale, threshold)

    print(f"  {'行':>4}  {'スコア':>10}  {'判定':>6}")
    print("  " + "-" * 26)
    for i, (flag, score) in enumerate(zip(flags, scores)):
        label = "【異常】" if flag else "  正常"
        print(f"  {i:>4}  {score:>10.6f}  {label}")


def main():
    parser = argparse.ArgumentParser(description="オートエンコーダー異常検知 (ONNX Runtime)")
    parser.add_argument("--csv", type=str, default=None, help="推論対象のCSVファイルパス")
    args = parser.parse_args()

    print("ONNX モデル読み込み中...")
    sess, threshold, scaler_mean, scaler_scale = load_runtime()
    print(f"  入力次元: {scaler_mean.shape[0]}  閾値: {threshold:.4f}\n")

    if args.csv:
        csv_mode(sess, threshold, scaler_mean, scaler_scale, args.csv)
    else:
        demo_mode(sess, threshold, scaler_mean, scaler_scale)


if __name__ == "__main__":
    main()
