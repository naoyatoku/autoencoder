"""
半導体製造装置センサーに近い合成時系列データを生成（グループ相関構造付き）

グループ構造:
  - 5グループ × 4センサー = 20センサー
  - 同グループのセンサーは共通の潜在因子（group_factor）で連動（正常時）
  - sensor_signal = base + coupling * group_factor + 個別振動 + ノイズ

異常パターン（Decorrelation のみ）:
  - グループ後半2センサーの group_factor を独立サイン波（同振幅・同周波数帯）に置換
  - 各センサーの平均・分散・自己相関は変わらない → 単センサー監視では検知不可
  - グループ内センサー間の相関だけが崩れる     → LSTM-AE で初めて検知可能

改善点（v2）:
  - group_factor の周波数を 0.05〜0.10 Hz に変更
    （窓20秒に1〜2周期収まり、LSTM-AE が相関パターンを学習しやすくなる）
  - 異常イベントの時間重複を排除（同一期間に複数イベントが重ならない）
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

TEST_MINUTES     = 30
N_TEST           = TEST_MINUTES * 60 * SAMPLE_RATE_HZ   # 18,000点
N_ANOMALY_EVENTS = 15   # decorrelation のみ
EVENT_MIN_LEN    = 150  # 15秒
EVENT_MAX_LEN    = 500  # 50秒
EVENT_MARGIN     = 100  # データ端のマージン


# ---------------------------------------------------------------------------
# 正常データ生成
# ---------------------------------------------------------------------------

def make_group_factor(n: int) -> np.ndarray:
    """
    グループ共通の振動因子。
    周波数 0.05〜0.10 Hz → 20秒窓に 1〜2 周期収まり LSTM-AE が相関を学習しやすい。
    （旧: 0.005〜0.05 Hz → 窓5秒では相関が見えなかった）
    """
    t     = np.linspace(0, n / SAMPLE_RATE_HZ, n)
    freq  = RNG.uniform(0.05, 0.10)
    amp   = RNG.uniform(2.0, 5.0)
    phase = RNG.uniform(0, 2 * np.pi)
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


def make_sensor_signal(n: int, group_factor: np.ndarray, coupling: float) -> np.ndarray:
    """グループ共通因子 + 個別振動（0.5〜2 Hz）+ ノイズ"""
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
    group_factors  = [make_group_factor(n) for _ in range(N_GROUPS)]
    coupling_coefs = np.empty(N_SENSORS, dtype=np.float32)
    columns = []
    for g_id, group in enumerate(GROUPS):
        for s_id in group:
            c = float(RNG.uniform(0.7, 1.0))
            coupling_coefs[s_id] = c
            columns.append(make_sensor_signal(n, group_factors[g_id], c))
    return np.column_stack(columns), group_factors, coupling_coefs


# ---------------------------------------------------------------------------
# 異常イベントのスケジューリング（重複なし）
# ---------------------------------------------------------------------------

def schedule_events(n_total: int, n_events: int, rng) -> list:
    """
    重複しない (start, length) のリストを返す。
    最大試行回数内で確保できた件数だけ返す。
    """
    occupied = np.zeros(n_total, dtype=bool)
    events   = []
    for _ in range(n_events * 50):
        if len(events) == n_events:
            break
        length = int(rng.integers(EVENT_MIN_LEN, EVENT_MAX_LEN + 1))
        start  = int(rng.integers(EVENT_MARGIN, n_total - length - EVENT_MARGIN))
        if not occupied[start:start + length].any():
            occupied[start:start + length] = True
            events.append((start, length))
    return sorted(events, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# 異常1: Decorrelation（正規化前の raw データに注入）
# ---------------------------------------------------------------------------

def inject_decorrelation(X: np.ndarray, group: list,
                         group_factor: np.ndarray, coupling_coefs: np.ndarray,
                         start: int, length: int) -> np.ndarray:
    """
    グループ後半2センサーの group_factor を独立サイン波に置換。
    境界でクロスフェードして不連続ジャンプ（見かけのレベルシフト）を防ぐ。
    """
    s      = X.copy()
    end    = min(start + length, len(s))
    seg_len = end - start
    n      = len(group_factor)
    gf_std = float(group_factor.std())
    amp    = gf_std * np.sqrt(2)
    t_full = np.linspace(0, n / SAMPLE_RATE_HZ, n)

    # 境界のフェード長: イベント長の25%または最大50サンプル
    fade = min(50, seg_len // 4)
    weight = np.ones(seg_len, dtype=np.float32)
    weight[:fade]  = np.linspace(0.0, 1.0, fade)
    weight[-fade:] = np.linspace(1.0, 0.0, fade)

    for s_id in group[len(group) // 2:]:
        c      = float(coupling_coefs[s_id])
        freq   = RNG.uniform(0.05, 0.10)
        phase  = RNG.uniform(0, 2 * np.pi)
        alt_gf = (amp * np.sin(2 * np.pi * freq * t_full + phase)).astype(np.float32)
        # 窓内 DC オフセットを group_factor に揃える → 見かけのレベルシフトを除去
        alt_seg = alt_gf[start:end].copy()
        alt_seg += group_factor[start:end].mean() - alt_seg.mean()
        diff   = alt_seg - group_factor[start:end]
        s[start:end, s_id] += c * weight * diff
    return s


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    # 正常 + テストを一括生成 → 同じセンサーパラメータ（base, coupling, freq）を共有
    print("=== データ生成（正常 + テスト）===")
    N_TOTAL = N_NORMAL + N_TEST
    X_all, group_factors_all, coupling_coefs = make_correlated_data(N_TOTAL)
    X_normal = X_all[:N_NORMAL]
    X_test   = X_all[N_NORMAL:].copy()
    group_factors_test = [gf[N_NORMAL:] for gf in group_factors_all]
    print(f"  正常: {X_normal.shape}  ({NORMAL_MINUTES}分)")
    print(f"  テスト: {X_test.shape}  ({TEST_MINUTES}分)")

    for g_id, group in enumerate(GROUPS):
        corr     = np.corrcoef(X_normal[:, group].T)
        avg_corr = (corr.sum() - len(group)) / (len(group) * (len(group) - 1))
        print(f"  グループ{g_id} センサー{group}: グループ内平均相関係数={avg_corr:.3f}")

    print("\n=== テストデータ異常注入 ===")
    labels = np.zeros(N_TEST, dtype=np.int32)

    # 正規化統計（decorrelation 注入前に計算）
    mean  = X_normal.mean(axis=0).astype(np.float32)
    scale = X_normal.std(axis=0).clip(min=1e-6).astype(np.float32)

    # イベントスケジューリング（重複なし）
    all_events = schedule_events(N_TEST, N_ANOMALY_EVENTS, RNG)

    print(f"\n  イベント合計: {len(all_events)} 件  (decorrelation のみ)")

    # --- Decorrelation 注入（raw データ段階）---
    print("\n  [Decorrelation 注入]")
    for i, (start, length) in enumerate(all_events):
        g_id   = int(RNG.integers(0, N_GROUPS))
        X_test = inject_decorrelation(
            X_test, GROUPS[g_id], group_factors_test[g_id], coupling_coefs, start, length
        )
        labels[start:start + length] = 1
        affected = GROUPS[g_id][len(GROUPS[g_id]) // 2:]
        print(f"    {i+1:2d}: グループ{g_id} センサー{affected}  "
              f"t={start/SAMPLE_RATE_HZ:.1f}〜{(start+length)/SAMPLE_RATE_HZ:.1f}s")

    # --- 正規化 ---
    X_normal_sc = ((X_normal - mean) / scale).astype(np.float32)
    X_test_sc   = ((X_test   - mean) / scale).astype(np.float32)

    anom_ratio = labels.mean() * 100
    print(f"\n  異常ラベル比率: {anom_ratio:.1f}%")

    # --- 保存 ---
    print("\n=== 保存 ===")
    np.save(DATA_DIR / "ts_X_normal.npy",    X_normal_sc)
    np.save(DATA_DIR / "ts_X_test.npy",      X_test_sc)
    np.save(DATA_DIR / "ts_y_test.npy",      labels)
    np.save(DATA_DIR / "ts_mean.npy",        mean)
    np.save(DATA_DIR / "ts_scale.npy",       scale)
    np.save(DATA_DIR / "ts_groups.npy",      np.array(GROUPS, dtype=np.int32))
    np.save(DATA_DIR / "ts_coupling.npy",    coupling_coefs)
    print("  完了。次のステップ: python 05_train_lstm_ae.py")


if __name__ == "__main__":
    main()
