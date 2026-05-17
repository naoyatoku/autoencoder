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
WINDOW    = 200   # 学習時と合わせる（20秒分）
BATCH     = 256   # 推論バッチサイズ


def load_runtime():
    sess           = ort.InferenceSession(str(MODEL_DIR / "lstm_ae.onnx"),
                                          providers=["CPUExecutionProvider"])
    threshold      = float(np.load(MODEL_DIR / "ts_threshold.npy")[0])
    mean           = np.load(DATA_DIR / "ts_mean.npy")
    scale          = np.load(DATA_DIR / "ts_scale.npy")
    sensor_stats   = np.load(MODEL_DIR / "sensor_err_stats.npy")  # (2, sensors)
    groups         = np.load(DATA_DIR / "ts_groups.npy")          # (5, 4)
    return sess, threshold, mean, scale, sensor_stats, groups


def eval_single_sensor(X_scaled: np.ndarray, y_true: np.ndarray, roll_w: int = 200):
    """
    ローリング平均偏差ベースの単センサー監視との比較。
    roll_w 点（デフォルト=20秒）のウィンドウ内平均からの最大偏差をスコアに使う。
    decorrelation 異常は平均も分散も変わらないため、このスコアが上がらない。
    """
    T, S = X_scaled.shape

    # 累積和でローリング平均を O(T) で計算
    padded = np.vstack([np.zeros((1, S), dtype=np.float32), X_scaled])  # (T+1, S)
    cumsum = np.cumsum(padded, axis=0)
    idx_end   = np.arange(1, T + 1)
    idx_start = np.maximum(0, idx_end - roll_w)
    counts    = (idx_end - idx_start).reshape(-1, 1)
    roll_mean = (cumsum[idx_end] - cumsum[idx_start]) / counts   # (T, S)

    max_dev = np.abs(X_scaled - roll_mean).max(axis=1)   # 全センサー中の最大偏差 (T,)

    print("\n  --- 単センサーローリング監視との比較（窓幅20秒）---")
    print(f"  {'閾値(σ)':>8}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}")
    print("  " + "-" * 42)
    for thr in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        flagged = max_dev > thr
        tp = int((flagged  & (y_true == 1)).sum())
        fp = int((flagged  & (y_true == 0)).sum())
        fn = int((~flagged & (y_true == 1)).sum())
        pr = tp / (tp + fp + 1e-9)
        rc = tp / (tp + fn + 1e-9)
        f1 = 2 * pr * rc / (pr + rc + 1e-9)
        print(f"  {thr:>8.1f}  {pr:>10.3f}  {rc:>8.3f}  {f1:>6.3f}")
    print("  ※ decorrelation 異常は各センサーの平均・分散が変わらないため F1~=0 が期待値")


def compute_scores(sess, X_scaled: np.ndarray, sensor_stats: np.ndarray,
                   groups: np.ndarray) -> np.ndarray:
    """
    各時刻の異常スコアを計算。
    グループ内z-スコア平均 → グループ間最大値 を使い、
    異常センサー2本の信号が残り18本で希釈されるのを防ぐ。
    """
    T = len(X_scaled)
    scores = np.zeros(T, dtype=np.float32)
    counts = np.zeros(T, dtype=np.float32)

    s_mean = sensor_stats[0]   # (sensors,)
    s_std  = sensor_stats[1]   # (sensors,)

    n_windows = T - WINDOW + 1
    if n_windows <= 0:
        print(f"エラー: データ長 ({T}) が窓幅 ({WINDOW}) より短いです")
        return scores

    for start in range(0, n_windows, BATCH):
        end    = min(start + BATCH, n_windows)
        batch  = np.stack([X_scaled[i:i + WINDOW] for i in range(start, end)])
        recon  = sess.run(None, {"windows": batch.astype(np.float32)})[0]
        sq_err   = (batch - recon) ** 2                              # (B, W, S)
        z        = np.clip((sq_err - s_mean) / s_std, 0, None)    # z-スコア化、負は0
        # グループ内（全timestep × 全sensor）を平均 → グループ間最大 → 窓スコア (B,)
        g_means  = np.stack([z[:, :, g].mean(axis=(1, 2)) for g in groups], axis=-1)  # (B, G)
        win_sc   = g_means.max(axis=-1)                            # (B,)
        for j, sc in enumerate(win_sc):
            t0 = start + j
            scores[t0:t0 + WINDOW] += sc
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


def demo_mode(sess, threshold, mean, scale, sensor_stats, groups):
    print("=== デモモード: 合成テストデータで検証 ===")
    X_test = np.load(DATA_DIR / "ts_X_test.npy")
    y_test = np.load(DATA_DIR / "ts_y_test.npy")

    print(f"  データ: {X_test.shape[0]}点 × {X_test.shape[1]}センサー")
    print(f"  異常ラベル比率: {y_test.mean()*100:.1f}%")
    print("\n  [LSTM-AE]")
    print("  スコア計算中...")
    scores = compute_scores(sess, X_test, sensor_stats, groups)
    flags  = scores > threshold
    print_summary(scores, flags, y_test, threshold)
    eval_single_sensor(X_test, y_test)


def csv_mode(sess, threshold, mean, scale, sensor_stats, groups, csv_path: str):
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
    scores = compute_scores(sess, X_scaled, sensor_stats, groups)
    flags  = scores > threshold
    print_summary(scores, flags, threshold=threshold)


def main():
    parser = argparse.ArgumentParser(description="LSTM-AE 時系列異常検知 (ONNX Runtime)")
    parser.add_argument("--csv", type=str, default=None, help="推論対象のCSVファイルパス")
    args = parser.parse_args()

    print("ONNX モデル読み込み中...")
    sess, threshold, mean, scale, sensor_stats, groups = load_runtime()
    print(f"  センサー数: {len(mean)}  窓幅: {WINDOW}点 ({WINDOW*100}ms)  閾値: {threshold:.4f}")

    if args.csv:
        csv_mode(sess, threshold, mean, scale, sensor_stats, groups, args.csv)
    else:
        demo_mode(sess, threshold, mean, scale, sensor_stats, groups)


if __name__ == "__main__":
    main()
