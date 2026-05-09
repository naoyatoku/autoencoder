"""
各異常パターンの波形を可視化するスクリプト
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Meiryo'

RNG = np.random.default_rng(42)

def make_normal_signal(n: int) -> np.ndarray:
    t = np.linspace(0, n / 10, n)
    signal = (50
              + 3 * np.sin(2 * np.pi * 0.05 * t)
              + 1.5 * np.sin(2 * np.pi * 1.0 * t)
              + RNG.normal(0, 0.3, n))
    return signal.astype(np.float32)

N = 300   # 30秒分
ANOM_START = 100
ANOM_LEN   = 100

t_axis = np.arange(N) / 10  # 秒

base = make_normal_signal(N)

fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
fig.suptitle("異常パターンの波形サンプル（赤帯=異常区間）", fontsize=14)

anomalies = {
    "spike（スパイク）\n瞬間的に値が跳ね上がる": "spike",
    "level_shift（レベルシフト）\n平均値が急に変化する": "level_shift",
    "volatility（振動増大）\nノイズが急に大きくなる": "volatility",
    "stuck（センサー固着）\n値が一定になる（センサー故障）": "stuck",
}

def inject(signal, kind):
    s = signal.copy()
    std = signal.std()
    start, end = ANOM_START, ANOM_START + ANOM_LEN

    if kind == "spike":
        idx = RNG.integers(start, end, size=5)
        s[idx] += RNG.choice([-1, 1], size=5) * RNG.uniform(10, 15) * std
    elif kind == "level_shift":
        s[start:end] += 8 * std
    elif kind == "volatility":
        s[start:end] += RNG.normal(0, 5 * std, end - start)
    elif kind == "stuck":
        s[start:end] = signal[start] + RNG.normal(0, 0.01 * std, end - start)
    return s

for ax, (title, kind) in zip(axes, anomalies.items()):
    s = inject(base, kind)
    ax.plot(t_axis, s, linewidth=0.8, color="steelblue", label="センサー値")
    ax.axvspan(t_axis[ANOM_START], t_axis[ANOM_START + ANOM_LEN - 1],
               alpha=0.25, color="red", label="異常区間")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("値")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.4)

axes[-1].set_xlabel("時間 [秒]")
fig.tight_layout()
fig.savefig("models/anomaly_patterns.png", dpi=150)
print("保存: models/anomaly_patterns.png")
