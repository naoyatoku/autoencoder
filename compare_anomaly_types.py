"""
異常タイプ別の検出能力比較スクリプト

既存の学習済みモデル (models/lstm_ae.onnx) を使って
spike / level_shift / volatility / stuck を個別に評価し
1枚のグラフにまとめる

正常ベース: 学習済み正規化済みデータ (ts_X_normal.npy) の末尾3000点を流用
異常注入 : 正規化済みスケール (std≈1) で注入
"""

import numpy as np
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Meiryo'
from pathlib import Path

MODEL_DIR  = Path("models")
DATA_DIR   = Path("data")
WINDOW     = 50
BATCH      = 256
RNG        = np.random.default_rng(7)

N_TEST     = 3000    # 5分
ANOM_START = 800
ANOM_LEN   = 400     # 40秒
SAMPLE_HZ  = 10


def inject_anomaly(X_sc: np.ndarray, kind: str) -> np.ndarray:
    """正規化済みデータ (std≈1) のセンサー0に異常を注入"""
    X = X_sc.copy()
    start, end = ANOM_START, ANOM_START + ANOM_LEN

    if kind == "spike":
        idx = RNG.integers(start, end, size=8)
        X[idx, 0] += RNG.choice([-1, 1], size=8) * RNG.uniform(6, 10)

    elif kind == "level_shift":
        X[start:end, 0] += 6.0

    elif kind == "volatility":
        X[start:end, 0] += RNG.normal(0, 4.0, end - start)

    elif kind == "stuck":
        X[start:end, 0] = X[start, 0]   # センサー値が固着

    return X


def compute_scores(sess, X_scaled: np.ndarray, sensor_stats: np.ndarray) -> np.ndarray:
    T      = len(X_scaled)
    scores = np.zeros(T, dtype=np.float32)
    counts = np.zeros(T, dtype=np.float32)
    s_mean = sensor_stats[0]
    s_std  = sensor_stats[1]

    for start in range(0, T - WINDOW + 1, BATCH):
        end    = min(start + BATCH, T - WINDOW + 1)
        batch  = np.stack([X_scaled[i:i + WINDOW] for i in range(start, end)])
        recon  = sess.run(None, {"windows": batch.astype(np.float32)})[0]
        sq_err = (batch - recon) ** 2
        z      = np.clip((sq_err - s_mean) / s_std, 0, None)
        err    = z.mean(axis=2)
        for j, e in enumerate(err):
            t0 = start + j
            scores[t0:t0 + WINDOW] += e
            counts[t0:t0 + WINDOW] += 1

    return scores / np.maximum(counts, 1)


def main():
    sess         = ort.InferenceSession(str(MODEL_DIR / "lstm_ae.onnx"),
                                        providers=["CPUExecutionProvider"])
    mean         = np.load(DATA_DIR / "ts_mean.npy")
    scale        = np.load(DATA_DIR / "ts_scale.npy")
    sensor_stats = np.load(MODEL_DIR / "sensor_err_stats.npy")

    # 学習済み正規化済み正常データの末尾3000点をベースに使う
    X_normal_sc = np.load(DATA_DIR / "ts_X_normal.npy")
    base_sc     = X_normal_sc[-N_TEST:].copy()   # (3000, 20)

    print(f"モデル読み込み完了")
    print(f"ベースデータ shape: {base_sc.shape}")

    kinds = ["spike", "level_shift", "volatility", "stuck"]
    kind_label = {
        "spike":       "spike（瞬間スパイク）",
        "level_shift": "level_shift（平均値シフト）",
        "volatility":  "volatility（振動増大）",
        "stuck":       "stuck（センサー固着）",
    }

    labels = np.zeros(N_TEST, dtype=np.int32)
    labels[ANOM_START:ANOM_START + ANOM_LEN] = 1
    t_axis = np.arange(N_TEST) / SAMPLE_HZ

    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle("異常タイプ別 検出能力比較（センサー0に注入、学習済みモデルで評価）", fontsize=13)

    for row, kind in enumerate(kinds):
        print(f"\n--- {kind} ---")
        X_sc   = inject_anomaly(base_sc, kind)
        scores = compute_scores(sess, X_sc, sensor_stats)

        # 異常前の正常区間からその場で閾値を設定（80パーセンタイル）
        normal_scores = scores[:ANOM_START]
        threshold     = float(np.percentile(normal_scores, 80))

        flags  = scores > threshold

        tp = int(((flags) & (labels == 1)).sum())
        fp = int(((flags) & (labels == 0)).sum())
        fn = int(((~flags) & (labels == 1)).sum())
        tn = int(((~flags) & (labels == 0)).sum())
        pr = tp / (tp + fp + 1e-9)
        rc = tp / (tp + fn + 1e-9)
        f1 = 2 * pr * rc / (pr + rc + 1e-9)
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"  Precision={pr:.3f}  Recall={rc:.3f}  F1={f1:.3f}")

        # 生値に戻してセンサー0を表示
        X_raw_s0 = X_sc[:, 0] * scale[0] + mean[0]

        # 左列: センサー0波形
        ax_w = axes[row, 0]
        ax_w.plot(t_axis, X_raw_s0, linewidth=0.7, color="steelblue")
        ax_w.axvspan(ANOM_START / SAMPLE_HZ, (ANOM_START + ANOM_LEN) / SAMPLE_HZ,
                     alpha=0.25, color="red", label="異常区間（正解）")
        ax_w.set_title(f"{kind_label[kind]} ― センサー0波形", fontsize=10)
        ax_w.set_ylabel("センサー値")
        ax_w.legend(fontsize=8, loc="upper right")
        ax_w.grid(True, alpha=0.4)

        # 右列: 異常スコア
        ax_s = axes[row, 1]
        ax_s.plot(t_axis, scores, linewidth=0.7, color="darkorange", label="異常スコア")
        ax_s.axhline(threshold, color="red", linestyle="--", linewidth=1.2,
                     label=f"閾値 {threshold:.2f}")
        ax_s.fill_between(t_axis, 0, max(scores.max(), threshold) * 1.15,
                          where=(labels == 1), alpha=0.2, color="red")
        result_color = "green" if rc > 0.5 else "red"
        metrics_txt  = (f"閾値={threshold:.3f}（正常区間の80%ile）\n"
                        f"Recall={rc:.3f}  Precision={pr:.3f}  F1={f1:.3f}\n"
                        f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        ax_s.text(0.02, 0.97, metrics_txt, transform=ax_s.transAxes, fontsize=9,
                  verticalalignment='top', color=result_color,
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        ax_s.set_title("異常スコア vs 閾値", fontsize=10)
        ax_s.set_ylabel("スコア")
        ax_s.legend(fontsize=8, loc="upper right")
        ax_s.grid(True, alpha=0.4)

    for ax in axes[-1]:
        ax.set_xlabel("時間 [秒]")

    fig.tight_layout()
    out = MODEL_DIR / "compare_anomaly_types.png"
    fig.savefig(out, dpi=150)
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
