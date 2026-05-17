"""
正常データ vs 異常データの可視化
各グラフに全20センサーを表示。正常・異常それぞれランダム10例。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.rcParams["font.family"] = "Meiryo"

DATA_DIR  = "data"
OUT_DIR   = "models"
HZ        = 10
WIN_ANOM  = 300  # 異常グラフ1枚に表示するサンプル数（前後込み）

RNG = np.random.default_rng(0)

# ---------- データ読み込み ----------
X_normal = np.load(f"./{DATA_DIR}/ts_X_normal.npy")   # (72000, 20)
X_test   = np.load(f"./{DATA_DIR}/ts_X_test.npy")     # (18000, 20)
y_test   = np.load(f"./{DATA_DIR}/ts_y_test.npy")     # (18000,)
GROUPS   = np.load(f"./{DATA_DIR}/ts_groups.npy")     # (5, 4)
N_SENSORS = X_normal.shape[1]

# グループごとの色（5色 × 各グループの4センサーを濃淡で）
GROUP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

def sensor_color(s_id):
    g_id = s_id // 4
    pos  = s_id % 4
    alpha = 1.0 - pos * 0.18   # 0番センサーが最濃、3番が最淡
    return GROUP_COLORS[g_id], alpha

def find_segments(y):
    segs, in_s, s = [], False, 0
    for t, v in enumerate(y):
        if v and not in_s:  in_s, s = True, t
        elif not v and in_s: in_s = False; segs.append((s, t - 1))
    if in_s: segs.append((s, len(y) - 1))
    return segs

def plot_sensors(ax, X_slice, t_start_s=0.0, anom_mask=None):
    """
    X_slice: (N, 20)
    t_start_s: スライス先頭の絶対時刻 [s]
    anom_mask: X_slice に対応する長さの bool 配列（True=異常）
    """
    t = t_start_s + np.arange(len(X_slice)) / HZ
    for s_id in range(N_SENSORS):
        color, alpha = sensor_color(s_id)
        ax.plot(t, X_slice[:, s_id], color=color, alpha=alpha, linewidth=0.7)
    if anom_mask is not None and anom_mask.any():
        # 連続した True 区間を塗りつぶし
        prev = False
        for i, v in enumerate(anom_mask):
            if v and not prev:
                xs = t[i]
            if not v and prev:
                ax.axvspan(xs, t[i - 1], alpha=0.18, color="red", zorder=0)
            prev = v
        if prev:
            ax.axvspan(xs, t[-1], alpha=0.18, color="red", zorder=0)
    ax.set_ylim(Y_LIM)
    ax.set_ylabel("正規化値", fontsize=7)
    ax.grid(True, linewidth=0.3)
    ax.tick_params(labelsize=7)

# ---------- 凡例パッチ ----------
legend_patches = [
    mpatches.Patch(color=GROUP_COLORS[g], label=f"グループ{g} (センサー{g*4}〜{g*4+3})")
    for g in range(5)
]

# 正常データの範囲を基準に y 軸を統一（外れ値で崩れないよう percentile を使用）
_p1  = float(np.percentile(X_normal, 1))
_p99 = float(np.percentile(X_normal, 99))
_margin = (_p99 - _p1) * 0.1
#Y_LIM = (_p1 - _margin, _p99 + _margin)
#toku 固定値にしてみます。
Y_LIM = (-10, 10)   

# ===========================================================
# 1. 正常データ: ランダム10窓（各200点 = 20秒）
# ===========================================================
WIN = 200   # 20秒
max_start = len(X_normal) - WIN
starts = RNG.integers(0, max_start, size=10)

fig, axes = plt.subplots(10, 1, figsize=(16, 22), sharex=False)
fig.suptitle("正常データ（ランダム10区間）— 全20センサー表示", fontsize=13, y=0.995)

for i, (ax, st) in enumerate(zip(axes, starts)):
    sl = X_normal[st:st + WIN]
    plot_sensors(ax, sl, t_start_s=st / HZ)
    ax.set_title(f"サンプル {i+1}  (t = {st/HZ:.1f}〜{(st+WIN)/HZ:.1f} s)", fontsize=8)

axes[-1].set_xlabel("時刻 [s]", fontsize=9)
fig.legend(handles=legend_patches, loc="upper right",
           bbox_to_anchor=(1.0, 0.995), fontsize=8, framealpha=0.8)
fig.tight_layout(rect=[0, 0, 0.87, 0.995])
out1 = f"{OUT_DIR}/viz_normal.png"
fig.savefig(out1, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"保存: {out1}")

# ===========================================================
# 2. 異常データ: 異常ラベルが付いた点をランダムに10点選び、
#    その点を中心に WIN_ANOM 幅で切り出す（前後に正常区間も含む）
# ===========================================================
# 各異常区間の「開始点」を取得し、ランダム10件選ぶ
all_segs = find_segments(y_test)
BEFORE   = 80    # 異常開始の何サンプル前から表示するか
AFTER    = WIN_ANOM - BEFORE   # 異常開始後に表示するサンプル数

chosen_segs = [all_segs[i] for i in
               RNG.choice(len(all_segs), size=min(10, len(all_segs)), replace=False)]
chosen_segs.sort(key=lambda x: x[0])

fig, axes = plt.subplots(10, 1, figsize=(16, 22), sharex=False)
fig.suptitle("異常データ（ランダム10区間・赤塗り = 真の異常区間）— 全20センサー表示\n"
             "各グラフ: 異常開始8秒前〜開始後22秒 を表示",
             fontsize=11, y=0.998)

for i, (ax, (seg_s, seg_e)) in enumerate(zip(axes, chosen_segs)):
    t_start = max(0, seg_s - BEFORE)
    t_end   = min(len(X_test), seg_s + AFTER)
    sl      = X_test[t_start:t_end]
    mask    = y_test[t_start:t_end].astype(bool)
    plot_sensors(ax, sl, t_start_s=t_start / HZ, anom_mask=mask)
    ax.axvline(seg_s / HZ, color="darkred", linewidth=1.0, linestyle=":", alpha=0.8)
    length_s = (seg_e - seg_s + 1) / HZ
    ax.set_title(
        f"サンプル {i+1}  異常開始 t={seg_s/HZ:.1f}s  "
        f"（異常区間の長さ {length_s:.1f}s）",
        fontsize=8
    )

axes[-1].set_xlabel("時刻 [s]", fontsize=9)

anom_patch = mpatches.Patch(color="red", alpha=0.3, label="異常区間")
fig.legend(handles=legend_patches + [anom_patch], loc="upper right",
           bbox_to_anchor=(1.0, 0.995), fontsize=8, framealpha=0.8)
fig.tight_layout(rect=[0, 0, 0.87, 0.995])
out2 = f"{OUT_DIR}/viz_anomaly.png"
fig.savefig(out2, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"保存: {out2}")

print("\n完了。")
print(f"  正常:  {out1}")
print(f"  異常:  {out2}")

# ===========================================================
# 3. グループ内センサー相関の時系列（ローリング相関）
#    正常時は高相関、decorrelation 異常時に相関が崩れる様子を可視化
# ===========================================================
WIN_CORR = 100   # ローリング相関の窓幅 (10秒)

def rolling_corr(x1: np.ndarray, x2: np.ndarray, window: int) -> np.ndarray:
    """2センサー間のローリング Pearson 相関係数を返す"""
    T = len(x1)
    out = np.full(T, np.nan, dtype=np.float32)
    cs1  = np.cumsum(x1);  cs2  = np.cumsum(x2)
    cs11 = np.cumsum(x1 * x1); cs22 = np.cumsum(x2 * x2); cs12 = np.cumsum(x1 * x2)
    for t in range(window - 1, T):
        s = t - window + 1
        n = window
        s1 = cs1[t]  - (cs1[s-1]  if s > 0 else 0)
        s2 = cs2[t]  - (cs2[s-1]  if s > 0 else 0)
        q1 = cs11[t] - (cs11[s-1] if s > 0 else 0)
        q2 = cs22[t] - (cs22[s-1] if s > 0 else 0)
        q12= cs12[t] - (cs12[s-1] if s > 0 else 0)
        var1 = q1/n - (s1/n)**2; var2 = q2/n - (s2/n)**2
        cov  = q12/n - (s1/n)*(s2/n)
        denom = np.sqrt(max(var1, 0) * max(var2, 0))
        out[t] = cov / denom if denom > 1e-8 else 0.0
    return out

fig3, axes3 = plt.subplots(len(GROUPS), 1, figsize=(16, 12), sharex=True)
fig3.suptitle(
    "グループ内センサー相関の時系列（正常センサー[0]×異常注入センサー[2]）\n"
    "Decorrelation 異常時にのみ相関が崩れる",
    fontsize=11, y=0.998
)

t_full = np.arange(len(X_test)) / HZ

for g_id, (ax, group) in enumerate(zip(axes3, GROUPS)):
    s_normal = int(group[0])   # 正常側（前半）
    s_anom   = int(group[2])   # 異常注入側（後半）
    corr = rolling_corr(X_test[:, s_normal], X_test[:, s_anom], WIN_CORR)
    ax.plot(t_full, corr, linewidth=0.8, color=GROUP_COLORS[g_id],
            label=f"Sensor{s_normal}×Sensor{s_anom}")
    ax.fill_between(t_full, -1, 1,
                    where=(y_test == 1), alpha=0.15, color="red", zorder=0)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel(f"G{g_id}", fontsize=8)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, linewidth=0.3)
    ax.tick_params(labelsize=7)

axes3[-1].set_xlabel("時刻 [s]", fontsize=9)
red_patch = mpatches.Patch(color="red", alpha=0.3, label="異常区間（真値）")
fig3.legend(handles=[red_patch], loc="upper left", bbox_to_anchor=(0.01, 0.998),
            fontsize=8)
fig3.tight_layout(rect=[0, 0, 1.0, 0.995])
out3 = f"{OUT_DIR}/viz_group_corr.png"
fig3.savefig(out3, dpi=130, bbox_inches="tight")
plt.close(fig3)
print(f"  相関時系列: {out3}")

# ===========================================================
# 4. センサーペアの散布図（正常区間 vs 異常区間）
#    相関があれば楕円状、ないなら雲状になる
# ===========================================================
# 最初の異常区間を使用
seg0_s, seg0_e = all_segs[0]
normal_win_end = seg0_s
normal_win_st  = max(0, normal_win_end - 300)  # 直前 300 サンプル
anom_win_st    = seg0_s
anom_win_en    = min(len(X_test), seg0_s + 300)

fig4, axes4 = plt.subplots(len(GROUPS), 4, figsize=(16, 12))
fig4.suptitle(
    f"センサーペア散布図（正常: t={normal_win_st/HZ:.0f}〜{normal_win_end/HZ:.0f}s  "
    f"異常: t={anom_win_st/HZ:.0f}〜{anom_win_en/HZ:.0f}s）\n"
    "Decorrelation 異常では、同グループ内でのみ正常→雲状に変化する",
    fontsize=10, y=0.999
)

for g_id, group in enumerate(GROUPS):
    s_lead = [int(group[0]), int(group[1])]   # 正常センサー（前半）
    s_tail = [int(group[2]), int(group[3])]   # 異常注入センサー（後半）
    pairs  = [(s_lead[0], s_tail[0]),          # 前半×後半（相関が崩れる）
              (s_lead[0], s_tail[1]),
              (s_lead[0], s_lead[1]),          # 前半×前半（崩れない）
              (s_tail[0], s_tail[1])]          # 後半×後半（崩れる）

    for col, (sa, sb) in enumerate(pairs):
        ax = axes4[g_id, col]
        xn = X_test[normal_win_st:normal_win_end, sa]
        yn = X_test[normal_win_st:normal_win_end, sb]
        xa = X_test[anom_win_st:anom_win_en, sa]
        ya = X_test[anom_win_st:anom_win_en, sb]
        ax.scatter(xn, yn, s=3, alpha=0.4, color="steelblue", label="正常")
        ax.scatter(xa, ya, s=3, alpha=0.4, color="tomato",    label="異常")
        cn = float(np.corrcoef(xn, yn)[0, 1])
        ca = float(np.corrcoef(xa, ya)[0, 1])
        label_type = "崩れる" if col < 2 or col == 3 else "正常側"
        ax.set_title(f"G{g_id}: S{sa}×S{sb}  [{label_type}]\n"
                     f"正常r={cn:.2f} / 異常r={ca:.2f}", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, linewidth=0.3)
        if g_id == 0 and col == 0:
            ax.legend(fontsize=7, markerscale=2)

fig4.tight_layout(rect=[0, 0, 1.0, 0.997])
out4 = f"{OUT_DIR}/viz_scatter.png"
fig4.savefig(out4, dpi=130, bbox_inches="tight")
plt.close(fig4)
print(f"  散布図:     {out4}")

print(f"\n全グラフ完了。")
print(f"  正常波形:   {out1}")
print(f"  異常波形:   {out2}")
print(f"  相関時系列: {out3}")
print(f"  散布図:     {out4}")
