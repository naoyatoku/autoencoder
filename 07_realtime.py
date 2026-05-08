"""
LSTM-AE リアルタイム異常検知デモ (ONNX Runtime)

ts_X_test.npy のデータをリアルタイムストリームとして再生し、
LSTM-AE ONNX モデルで異常をリアルタイム検出します。

注意:
  リアルタイムでは各時刻のスコア = 直前 WINDOW 点の再構成誤差のみ使用。
  05_train の評価（複数窓の平均）とは若干異なるため、
  スコアの絶対値は高くなる傾向があります。

使い方:
  python 07_realtime.py           # 実時間 (10 Hz = 100 ms/ステップ)
  python 07_realtime.py --fast    # 速度制限なし（全データを即時処理）
  python 07_realtime.py --plot    # 結果グラフを models/rt_result.png に保存
"""

import argparse
import collections
import sys
import time

import numpy as np
import onnxruntime as ort
from pathlib import Path

MODEL_DIR = Path("models")
DATA_DIR  = Path("data")

WINDOW = 50
HZ     = 10
STEP_S = 1.0 / HZ


# ---------------------------------------------------------------------------
# モデル読み込み
# ---------------------------------------------------------------------------

def load_runtime():
    sess      = ort.InferenceSession(str(MODEL_DIR / "lstm_ae.onnx"),
                                     providers=["CPUExecutionProvider"])
    threshold = float(np.load(MODEL_DIR / "ts_threshold.npy")[0])
    return sess, threshold


def score_window(sess, buf: np.ndarray) -> float:
    """buf: (WINDOW, n_sensors) → 再構成誤差 (MSE)"""
    x     = buf[np.newaxis].astype(np.float32)   # (1, W, S)
    recon = sess.run(None, {"windows": x})[0]
    return float(((x - recon) ** 2).mean())


# ---------------------------------------------------------------------------
# 真の異常区間を抽出
# ---------------------------------------------------------------------------

def find_segments(y: np.ndarray):
    """連続した y==1 区間を (start, end) のリストで返す（両端 inclusive）"""
    segs, in_seg, s = [], False, 0
    for t, v in enumerate(y):
        if v and not in_seg:
            in_seg, s = True, t
        elif not v and in_seg:
            in_seg = False
            segs.append((s, t - 1))
    if in_seg:
        segs.append((s, len(y) - 1))
    return segs


# ---------------------------------------------------------------------------
# ストリーム処理
# ---------------------------------------------------------------------------

def run(sess, threshold, X, y, realtime: bool) -> np.ndarray:
    T, n_sensors = X.shape
    buf    = collections.deque(maxlen=WINDOW)
    scores = np.full(T, np.nan, dtype=np.float32)
    t_wall = time.time()

    print(f"\n{'時刻[s]':>8}  {'スコア':>10}  {'判定':>6}  メモ")
    print("-" * 55)

    prev_det   = False
    prev_label = 0

    for t in range(T):
        buf.append(X[t])
        if len(buf) == WINDOW:
            scores[t] = score_window(sess, np.array(buf))

        det   = bool(not np.isnan(scores[t]) and scores[t] > threshold)
        label = int(y[t])

        # 状態変化時 または 5 ステップごとに表示
        state_change = (det != prev_det) or (label != prev_label)
        if t % 5 == 0 or state_change:
            t_s   = t / HZ
            s_str = f"{scores[t]:>10.5f}" if not np.isnan(scores[t]) else "  (ウォームアップ)"
            d_str = "[異常]" if det else "[ 正常]"

            note = ""
            if label == 1 and prev_label == 0:
                note = ">> 異常区間 開始"
            elif label == 0 and prev_label == 1:
                note = "<< 異常区間 終了"
            if det and not prev_det:
                note += "  [検出!]"
            elif not det and prev_det:
                note += "  (正常に復帰)"

            print(f"{t_s:>8.1f}  {s_str}  {d_str}  {note}")
            sys.stdout.flush()

        prev_det   = det
        prev_label = label

        if realtime:
            deadline = t_wall + (t + 1) * STEP_S
            slack    = deadline - time.time()
            if slack > 0.001:
                time.sleep(slack)

    return scores


# ---------------------------------------------------------------------------
# 結果サマリー
# ---------------------------------------------------------------------------

def print_summary(scores: np.ndarray, y: np.ndarray, threshold: float, hz: int):
    segs = find_segments(y)

    # 各区間の初検出タイムを求める
    results = []
    for seg_s, seg_e in segs:
        det_t = None
        for t in range(seg_s, seg_e + 1):
            if not np.isnan(scores[t]) and scores[t] > threshold:
                det_t = t
                break
        results.append((seg_s, seg_e, det_t))

    n_detected = sum(1 for _, _, d in results if d is not None)
    n_missed   = sum(1 for _, _, d in results if d is None)

    # 正常区間での誤報率
    valid       = ~np.isnan(scores)
    flags       = scores > threshold
    normal_mask = y == 0
    fp_pts      = int((flags & valid & normal_mask).sum())
    norm_pts    = int((valid & normal_mask).sum())
    fpr         = fp_pts / (norm_pts + 1e-9)

    print("\n" + "=" * 62)
    print("  検出結果サマリー")
    print("=" * 62)
    print(f"  異常区間 (真値)   : {len(segs)} 件")
    print(f"  検出              : {n_detected} 件  ({n_detected / max(len(segs), 1) * 100:.0f}%)")
    print(f"  見逃し            : {n_missed} 件")
    print(f"  正常時 誤報率     : {fpr * 100:.2f}%  ({fp_pts}/{norm_pts} 点)")
    print()
    print(f"  {'#':>3}  {'開始[s]':>8}  {'終了[s]':>8}  {'長さ[s]':>7}  {'初検出[s]':>9}  {'遅延[s]':>7}")
    print("  " + "-" * 52)
    for i, (s, e, d) in enumerate(results):
        length = (e - s + 1) / hz
        if d is not None:
            det_s = f"{d / hz:>9.2f}"
            delay = f"{(d - s) / hz:>7.2f}"
        else:
            det_s = "   見逃し"
            delay = "      -"
        print(f"  {i + 1:>3}  {s / hz:>8.2f}  {e / hz:>8.2f}  {length:>7.2f}  {det_s}  {delay}")
    print("=" * 62)

    # 検出した区間の平均遅延
    delays = [(d - s) / hz for s, _, d in results if d is not None]
    if delays:
        print(f"\n  平均検出遅延: {np.mean(delays):.2f} 秒"
              f"  (最小 {np.min(delays):.2f}s / 最大 {np.max(delays):.2f}s)")


# ---------------------------------------------------------------------------
# 結果グラフ
# ---------------------------------------------------------------------------

def save_plot(scores: np.ndarray, y: np.ndarray, threshold: float, hz: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "Meiryo"

    valid  = ~np.isnan(scores)
    t_axis = np.arange(len(scores)) / hz
    s_max  = float(scores[valid].max()) if valid.any() else 1.0

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # --- 上段: 異常スコア ---
    axes[0].plot(t_axis[valid], scores[valid], linewidth=0.6, color="steelblue",
                 label="異常スコア (リアルタイム)")
    axes[0].axhline(threshold, color="red", linestyle="--", linewidth=1.0,
                    label=f"閾値 = {threshold:.5f}")
    axes[0].fill_between(t_axis, 0, s_max,
                         where=(y == 1), alpha=0.2, color="red", label="真の異常区間")
    axes[0].set_ylabel("異常スコア (MSE)")
    axes[0].set_title("リアルタイム異常スコアと真の異常区間")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linewidth=0.3)

    # --- 下段: 検知イベント ---
    flags   = scores > threshold
    tp_mask = flags & valid & (y == 1)
    fp_mask = flags & valid & (y == 0)
    fn_mask = ~flags & valid & (y == 1)

    axes[1].scatter(t_axis[tp_mask], np.ones(tp_mask.sum()),
                    color="tomato",  s=6, label=f"正検知 (TP) {tp_mask.sum()}", zorder=3)
    axes[1].scatter(t_axis[fp_mask], np.zeros(fp_mask.sum()),
                    color="orange",  s=4, label=f"誤報 (FP) {fp_mask.sum()}", zorder=3)
    axes[1].scatter(t_axis[fn_mask], np.full(fn_mask.sum(), 0.5),
                    color="gray",    s=4, label=f"見逃し (FN) {fn_mask.sum()}", zorder=2)
    axes[1].set_yticks([0, 0.5, 1])
    axes[1].set_yticklabels(["正常時誤報", "異常を見逃し", "異常検知"])
    axes[1].set_xlabel("時刻 [s]")
    axes[1].set_title("検知イベントの分布")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linewidth=0.3)

    fig.tight_layout()
    out = MODEL_DIR / "rt_result.png"
    fig.savefig(out, dpi=150)
    print(f"\n  グラフ保存: {out}")


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LSTM-AE リアルタイム異常検知デモ")
    parser.add_argument("--fast", action="store_true",
                        help="速度制限なし（全データを即時処理）")
    parser.add_argument("--plot", action="store_true",
                        help="結果グラフを models/rt_result.png に保存")
    args = parser.parse_args()

    print("ONNX モデル読み込み中...")
    sess, threshold = load_runtime()
    print(f"  閾値: {threshold:.5f}  窓幅: {WINDOW} 点 ({WINDOW * 100} ms)")

    X = np.load(DATA_DIR / "ts_X_test.npy")
    y = np.load(DATA_DIR / "ts_y_test.npy")
    T = len(X)
    print(f"  データ: {T} 点 ({T / HZ:.0f} 秒 = {T / HZ / 60:.1f} 分)  センサー: {X.shape[1]}")
    print(f"  真の異常区間: {len(find_segments(y))} 件  "
          f"異常率: {y.mean() * 100:.1f}%")

    if args.fast:
        print("\n[高速モード: 速度制限なし]")
    else:
        est = T / HZ
        print(f"\n[リアルタイムモード: {HZ} Hz]  完走まで約 {est:.0f} 秒")
        print("  速く試したい場合は --fast を付けてください。")

    scores = run(sess, threshold, X, y, realtime=not args.fast)
    print_summary(scores, y, threshold, HZ)

    if args.plot:
        save_plot(scores, y, threshold, HZ)


if __name__ == "__main__":
    main()
