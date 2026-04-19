"""
半導体製造装置センサーに近い合成時系列データを生成

想定仕様:
  - センサー数  : 20種類
  - サンプリング: 100ms間隔
  - 正常運転   : 約2時間分 (72,000点)
  - 異常区間   : ランダムに30箇所挿入

異常パターン:
  - スパイク     : 瞬間的な大きな値の跳ね上がり
  - レベルシフト : 平均値が急に変化して戻る
  - 振動増大    : ノイズ振幅が急増する
  - センサー固着 : 値が一定になる（センサー故障）
"""

import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

RNG      = np.random.default_rng(42)
N_SENSORS      = 20
SAMPLE_RATE_HZ = 10           # 100ms = 10Hz
NORMAL_MINUTES = 120          # 正常運転 2時間
N_NORMAL       = NORMAL_MINUTES * 60 * SAMPLE_RATE_HZ   # 72,000点
N_ANOMALY_EVENTS = 30         # 異常イベント数


def make_normal_signal(n: int, sensor_id: int) -> np.ndarray:
    """センサーごとに異なる正常パターンを生成"""
    t = np.linspace(0, n / SAMPLE_RATE_HZ, n)

    # 各センサーは基準値・周期・ノイズが異なる
    base    = RNG.uniform(20, 80)
    amp1    = RNG.uniform(1, 5)
    amp2    = RNG.uniform(0.5, 2)
    freq1   = RNG.uniform(0.01, 0.1)
    freq2   = RNG.uniform(0.5, 2.0)
    noise   = RNG.uniform(0.1, 0.5)
    trend   = RNG.uniform(-0.001, 0.001)   # ゆっくりしたドリフト

    signal = (base
              + amp1 * np.sin(2 * np.pi * freq1 * t)
              + amp2 * np.sin(2 * np.pi * freq2 * t + RNG.uniform(0, np.pi))
              + trend * t
              + RNG.normal(0, noise, n))
    return signal.astype(np.float32)


def inject_anomaly(signal: np.ndarray, kind: str, start: int, length: int) -> np.ndarray:
    s = signal.copy()
    end = min(start + length, len(s))
    std = signal.std()

    if kind == "spike":
        # 数点だけ大きなスパイク
        idx = RNG.integers(start, end, size=3)
        s[idx] += RNG.choice([-1, 1]) * RNG.uniform(8, 15) * std

    elif kind == "level_shift":
        # 平均値がシフト
        shift = RNG.choice([-1, 1]) * RNG.uniform(5, 10) * std
        s[start:end] += shift

    elif kind == "volatility":
        # ノイズ振幅が増大
        extra = RNG.normal(0, RNG.uniform(3, 6) * std, end - start)
        s[start:end] += extra

    elif kind == "stuck":
        # 値が固着（センサー故障）
        s[start:end] = signal[start] + RNG.normal(0, 0.01 * std, end - start)

    return s


def main():
    print("=== 正常データ生成 ===")
    # (N_NORMAL, N_SENSORS)
    X_normal = np.stack(
        [make_normal_signal(N_NORMAL, i) for i in range(N_SENSORS)], axis=1
    )
    print(f"  正常データ shape: {X_normal.shape}  ({N_NORMAL/SAMPLE_RATE_HZ/60:.0f}分)")

    print("\n=== テストデータ生成（正常 + 異常区間あり）===")
    TEST_MINUTES = 30
    N_TEST = TEST_MINUTES * 60 * SAMPLE_RATE_HZ   # 18,000点

    X_test_base = np.stack(
        [make_normal_signal(N_TEST, i) for i in range(N_SENSORS)], axis=1
    )
    labels = np.zeros(N_TEST, dtype=np.int32)  # 0=正常, 1=異常

    kinds = ["spike", "level_shift", "volatility", "stuck"]
    for event_i in range(N_ANOMALY_EVENTS):
        kind    = RNG.choice(kinds)
        length  = int(RNG.uniform(50, 300))        # 0.5〜30秒
        start   = int(RNG.integers(100, N_TEST - length - 100))
        # 1〜3個のセンサーに注入
        n_affected = RNG.integers(1, 4)
        sensors = RNG.choice(N_SENSORS, size=n_affected, replace=False)
        for s_id in sensors:
            X_test_base[:, s_id] = inject_anomaly(
                X_test_base[:, s_id], kind, start, length
            )
        labels[start:start + length] = 1
        print(f"  異常{event_i+1:2d}: {kind:15s}  センサー{list(sensors)}  "
              f"t={start/SAMPLE_RATE_HZ:.1f}s  長さ={length/SAMPLE_RATE_HZ:.1f}s")

    print(f"\n  テストデータ shape : {X_test_base.shape}")
    print(f"  異常ラベル比率     : {labels.mean()*100:.1f}%")

    print("\n=== 正規化パラメータを正常データから計算 ===")
    mean  = X_normal.mean(axis=0).astype(np.float32)
    scale = X_normal.std(axis=0).clip(min=1e-6).astype(np.float32)

    X_normal_sc = ((X_normal - mean) / scale).astype(np.float32)
    X_test_sc   = ((X_test_base - mean) / scale).astype(np.float32)

    print("\n=== 保存 ===")
    np.save(DATA_DIR / "ts_X_normal.npy", X_normal_sc)
    np.save(DATA_DIR / "ts_X_test.npy",   X_test_sc)
    np.save(DATA_DIR / "ts_y_test.npy",   labels)
    np.save(DATA_DIR / "ts_mean.npy",     mean)
    np.save(DATA_DIR / "ts_scale.npy",    scale)
    print(f"  ts_X_normal.npy : {X_normal_sc.shape}")
    print(f"  ts_X_test.npy   : {X_test_sc.shape}")
    print(f"  ts_y_test.npy   : {labels.shape}")
    print("\n完了。次のステップ: python 05_train_lstm_ae.py")


if __name__ == "__main__":
    main()
