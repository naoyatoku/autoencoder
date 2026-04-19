"""
LSTM-AE による時系列異常検知推論スクリプト (ONNX Runtime)

PyTorch 不要。onnxruntime だけで動作します。

使い方:
  python 06_inference_lstm.py              # デモ（合成テストデータで検証）
  python 06_inference_lstm.py --csv data.csv  # CSVファイルで推論

CSVフォーマット:
  1行 = 1タイムステップ（100ms）
  列数 = センサー数（学習時と同じ）
  ヘッダー行なし、カンマ区切り
"""

import argparse
import numpy as np
import onnxruntime as ort
from pathlib import Path

MODEL_DIR = Path("models")
DATA_DIR  = Path("data")
WINDOW    = 50    # 学習時と合わせる（5秒分）
BATCH     = 256   # 推論バッチサイズ


def load_runtime():
    sess      = ort.InferenceSession(str(MODEL_DIR / "lstm_ae.onnx"),
                                     providers=["CPUExecutionProvider"])
    threshold = float(np.load(MODEL_DIR / "ts_threshold.npy")[0])
    mean      = np.load(DATA_DIR / "ts_mean.npy")
    scale     = np.load(DATA_DIR / "ts_scale.npy")
    return sess, threshold, mean, scale


def compute_scores(sess, X_scaled: np.ndarray) -> np.ndarray:
    """
    各時刻の異常スコア（再構成誤差）を計算
    X_scaled: (T, n_sensors) — 正規化済み
    戻り値  : (T,) の異常スコア配列
    """
    T = len(X_scaled)
    scores = np.zeros(T, dtype=np.float32)
    counts = np.zeros(T, dtype=np.float32)

    # 全時刻の窓を一括作成
    n_windows = T - WINDOW + 1
    if n_windows <= 0:
        print(f"エラー: データ長 ({T}) が窓幅 ({WINDOW}) より短いです")
        return scores

    for start in range(0, n_windows, BATCH):
        end   = min(start + BATCH, n_windows)
        batch = np.stack([X_scaled[i:i + WINDOW] for i in range(start, end)])
        recon = sess.run(None, {"windows": batch.astype(np.float32)})[0]
        err   = ((batch - recon) ** 2).mean(axis=2)   # (B, window)
        for j, e in enumerate(err):
            t0 = start + j
            scores[t0:t0 + WINDOW] += e
            counts[t0:t0 + WINDOW] += 1

    return scores / np.maximum(counts, 1)


def print_summary(scores, flags, y_true=None, threshold=0.0):
    print(f"\n  総時刻数    : {len(scores)}")
    print(f"  閾値        : {threshold:.4f}")
    print(f"  異常検知数  : {flags.sum()} 点 ({flags.mean()*100:.1f}%)")

    if y_true is not None:
        tp = int(((flags) & (y_true==1)).sum())
        fp = int(((flags) & (y_true==0)).sum())
        fn = int(((~flags) & (y_true==1)).sum())
        tn = int(((~flags) & (y_true==0)).sum())
        pr = tp / (tp + fp + 1e-9)
        rc = tp / (tp + fn + 1e-9)
        f1 = 2 * pr * rc / (pr + rc + 1e-9)
        print(f"\n  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"  Precision={pr:.3f}  Recall={rc:.3f}  F1={f1:.3f}")

    # 検知した異常区間を表示
    print("\n  検知した異常区間 (先頭10件):")
    print(f"  {'開始[s]':>8}  {'終了[s]':>8}  {'長さ[s]':>8}  {'最大スコア':>10}")
    print("  " + "-" * 46)
    in_anom = False
    seg_start = 0
    shown = 0
    for t, f in enumerate(flags):
        if f and not in_anom:
            in_anom = True; seg_start = t
        elif not f and in_anom:
            in_anom = False
            length = t - seg_start
            max_s  = scores[seg_start:t].max()
            if shown < 10:
                print(f"  {seg_start/10:>8.1f}  {t/10:>8.1f}  {length/10:>8.1f}  {max_s:>10.4f}")
            shown += 1
    if shown >= 10:
        print(f"  ... 合計 {shown} 区間")


def demo_mode(sess, threshold, mean, scale):
    print("=== デモモード: 合成テストデータで検証 ===")
    X_test = np.load(DATA_DIR / "ts_X_test.npy")   # すでに正規化済み
    y_test = np.load(DATA_DIR / "ts_y_test.npy")

    print(f"  データ: {X_test.shape[0]}点 × {X_test.shape[1]}センサー")
    print("  スコア計算中...")
    scores = compute_scores(sess, X_test)
    flags  = scores > threshold
    print_summary(scores, flags, y_test, threshold)


def csv_mode(sess, threshold, mean, scale, csv_path: str):
    print(f"=== CSV推論モード: {csv_path} ===")
    import csv
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            rows.append([float(v) if v.strip() else 0.0 for v in row])

    X_raw = np.array(rows, dtype=np.float32)
    if X_raw.shape[1] != len(mean):
        print(f"エラー: CSVの列数 ({X_raw.shape[1]}) != センサー数 ({len(mean)})")
        return

    X_scaled = ((X_raw - mean) / scale).astype(np.float32)
    print(f"  データ: {X_scaled.shape[0]}点 × {X_scaled.shape[1]}センサー")
    print("  スコア計算中...")
    scores = compute_scores(sess, X_scaled)
    flags  = scores > threshold
    print_summary(scores, flags, threshold=threshold)


def main():
    parser = argparse.ArgumentParser(description="LSTM-AE 時系列異常検知 (ONNX Runtime)")
    parser.add_argument("--csv", type=str, default=None, help="推論対象のCSVファイルパス")
    args = parser.parse_args()

    print("ONNX モデル読み込み中...")
    sess, threshold, mean, scale = load_runtime()
    print(f"  センサー数: {len(mean)}  窓幅: {WINDOW}点 ({WINDOW*100}ms)  閾値: {threshold:.4f}")

    if args.csv:
        csv_mode(sess, threshold, mean, scale, args.csv)
    else:
        demo_mode(sess, threshold, mean, scale)


if __name__ == "__main__":
    main()
