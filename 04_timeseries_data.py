"""
半導体製造装置センサーに近い合成時系列データを生成（グループ相関構造付き）

グループ構造:
  - 5グループ × 4センサー = 20センサー
  - 同グループのセンサーは共通の潜在因子（group_factor）で連動（正常時）
  - sensor_signal = base + coupling * group_factor + 個別振動 + ノイズ

異常パターン（decorrelation）:
  - グループの後半2センサーの group_factor を同振幅の独立ノイズに置換
  - 各センサーの平均・分散は変わらない → 単センサー監視では検知不可
  - グループ内の相関だけが崩れる       → LSTM-AE で初めて検知可能
"""

import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

RNG            = np.random.default_rng(42)
N_SENSORS      = 20
N_GROUPS       = 5
GROUP_SIZE     = 4
GROUPS         = [list(range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE)) for g in range(N_GROUPS)]
SAMPLE_RATE_HZ = 10
NORMAL_MINUTES = 120
N_NORMAL       = NORMAL_MINUTES * 60 * SAMPLE_RATE_HZ   # 72,000点
N_ANOMALY_EVENTS = 30


def make_group_factor(n: int) -> np.ndarray:
    """グループ共通の緩やかな振動因子"""
    t     = np.linspace(0, n / SAMPLE_RATE_HZ, n)
    freq  = RNG.uniform(0.005, 0.05)
    amp   = RNG.uniform(2.0, 5.0)
    phase = RNG.uniform(0, 2 * np.pi)
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


def make_sensor_signal(n: int, group_factor: np.ndarray, coupling: float) -> np.ndarray:
    """グループ共通因子 + 個別振動 + ノイズ"""
    t     = np.linspace(0, n / SAMPLE_RATE_HZ, n)
    base  = RNG.uniform(20, 80)
    amp   = RNG.uniform(0.5, 1.5)
    freq  = RNG.uniform(0.5, 2.0)
    noise = RNG.uniform(0.1, 0.3)
    trend = RNG.uniform(-0.001, 0.001)
    return (base
            + coupling * group_factor
            + amp * np.sin(2 * np.pi * freq * t + RNG.uniform(0, np.pi))
            + trend * t
            + RNG.normal(0, noise, n)).astype(np.float32)


def make_correlated_data(n: int):
    """
    全センサーデータを生成。
    Returns:
      X              : (n, N_SENSORS)
      group_factors  : list of N_GROUPS arrays of shape (n,)
      coupling_coefs : (N_SENSORS,)  各センサーの group_factor への結合強度
    """
    group_factors  = [make_group_factor(n) for _ in range(N_GROUPS)]
    coupling_coefs = np.empty(N_SENSORS, dtype=np.float32)
    columns = []
    for g_id, group in enumerate(GROUPS):
        for s_id in group:
            c = float(RNG.uniform(0.7, 1.0))
            coupling_coefs[s_id] = c
            columns.append(make_sensor_signal(n, group_factors[g_id], c))
    return np.column_stack(columns), group_factors, coupling_coefs


def inject_decorrelation(X: np.ndarray, group: list,
                         group_factor: np.ndarray, coupling_coefs: np.ndarray,
                         start: int, length: int) -> np.ndarray:
    """
    グループ後半センサーの group_factor を、同振幅・同周波数帯のランダム位相サイン波に置換。

    白色ノイズではなくサイン波を使う理由:
      白色ノイズは高周波成分を持つため、ローリング平均からの偏差が増加してしまい
      単センサー監視でも検知可能になってしまう。サイン波なら時系列特性が保たれる。

    置換後の効果:
      ・各センサーの平均・分散・時系列の自己相関は変わらない
      ・グループ内センサー間の相関だけが崩れる（異なる位相で動く）
    """
    s      = X.copy()
    end    = min(start + length, len(s))
    n      = len(group_factor)
    gf_std = float(group_factor.std())
    amp    = gf_std * np.sqrt(2)          # sin波の std = amp/sqrt(2) となるよう振幅を設定
    t_full = np.linspace(0, n / SAMPLE_RATE_HZ, n)

    for s_id in group[len(group) // 2:]:   # 後半 2 センサーに適用
        c     = float(coupling_coefs[s_id])
        freq  = RNG.uniform(0.005, 0.05)   # 元の group_factor と同じ周波数帯
        phase = RNG.uniform(0, 2 * np.pi)  # ランダム位相 → 元と独立
        alt_gf = (amp * np.sin(2 * np.pi * freq * t_full + phase)).astype(np.float32)
        s[start:end, s_id] -= c * group_factor[start:end]
        s[start:end, s_id] += c * alt_gf[start:end]

    return s


def main():
    print("=== 正常データ生成（グループ相関構造あり）===")
    X_normal, _, _ = make_correlated_data(N_NORMAL)
    print(f"  shape: {X_normal.shape}  ({N_NORMAL / SAMPLE_RATE_HZ / 60:.0f}分)")
    for g_id, group in enumerate(GROUPS):
        corr     = np.corrcoef(X_normal[:, group].T)
        avg_corr = (corr.sum() - len(group)) / (len(group) * (len(group) - 1))
        print(f"  グループ{g_id} センサー{group}: グループ内平均相関係数={avg_corr:.3f}")

    print("\n=== テストデータ生成（decorrelation 異常のみ）===")
    TEST_MINUTES = 30
    N_TEST       = TEST_MINUTES * 60 * SAMPLE_RATE_HZ
    X_test, group_factors_test, coupling_coefs = make_correlated_data(N_TEST)
    labels = np.zeros(N_TEST, dtype=np.int32)

    for event_i in range(N_ANOMALY_EVENTS):
        length = int(RNG.uniform(100, 400))
        start  = int(RNG.integers(200, N_TEST - length - 200))
        g_id   = int(RNG.integers(0, N_GROUPS))
        X_test = inject_decorrelation(
            X_test, GROUPS[g_id], group_factors_test[g_id], coupling_coefs, start, length
        )
        labels[start:start + length] = 1
        affected = GROUPS[g_id][len(GROUPS[g_id]) // 2:]
        print(f"  異常{event_i + 1:2d}: グループ{g_id} センサー{affected} group_factor→独立ノイズ  "
              f"t={start / SAMPLE_RATE_HZ:.1f}s  長さ={length / SAMPLE_RATE_HZ:.1f}s")

    print(f"\n  shape: {X_test.shape}  異常ラベル比率: {labels.mean() * 100:.1f}%")

    print("\n=== 正規化（正常データの統計を使用）===")
    mean  = X_normal.mean(axis=0).astype(np.float32)
    scale = X_normal.std(axis=0).clip(min=1e-6).astype(np.float32)
    X_normal_sc = ((X_normal - mean) / scale).astype(np.float32)
    X_test_sc   = ((X_test   - mean) / scale).astype(np.float32)

    # 正規化後の異常区間での各センサー平均ずれを確認
    print("\n=== 異常区間での単センサー変化量確認 ===")
    anom_mask = labels == 1
    norm_mask = labels == 0
    for g_id in range(N_GROUPS):
        for s_id in GROUPS[g_id][len(GROUPS[g_id]) // 2:]:
            diff = abs(X_test_sc[anom_mask, s_id].mean() - X_test_sc[norm_mask, s_id].mean())
            print(f"  センサー{s_id:2d}: 異常区間の平均ずれ={diff:.4f}σ  "
                  f"（これが単センサー監視で見える変化量）")

    print("\n=== 保存 ===")
    np.save(DATA_DIR / "ts_X_normal.npy",    X_normal_sc)
    np.save(DATA_DIR / "ts_X_test.npy",      X_test_sc)
    np.save(DATA_DIR / "ts_y_test.npy",      labels)
    np.save(DATA_DIR / "ts_mean.npy",        mean)
    np.save(DATA_DIR / "ts_scale.npy",       scale)
    np.save(DATA_DIR / "ts_groups.npy",      np.array(GROUPS, dtype=np.int32))
    np.save(DATA_DIR / "ts_coupling.npy",    coupling_coefs)
    print(f"  ts_X_normal.npy : {X_normal_sc.shape}")
    print(f"  ts_X_test.npy   : {X_test_sc.shape}")
    print("\n完了。次のステップ: python 05_train_lstm_ae.py")


if __name__ == "__main__":
    main()
